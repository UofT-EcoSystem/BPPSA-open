import argparse
import numpy as np
import os
import pandas as pd
import random
import sys
import time
import torch
import scan
import torch.nn as nn
import torch.optim as optim
import torch.utils.data


class RNN(nn.Module):

    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 bias=True,
                 nonlinearity='tanh',
                 mode='blelloch',
                 rnn_type='PyTorch',
                 test_artifacts=None):
        super(RNN, self).__init__()
        self._input_size = input_size
        self._hidden_size = hidden_size
        assert mode in {'normal', 'blelloch', 'normal-nobp'}
        self._mode = mode
        assert rnn_type in {'PyTorch', 'cuDNN'}
        self._rnn_type = rnn_type
        if rnn_type == 'PyTorch':
            self._rnn_cell = nn.RNNCell(
                input_size,
                hidden_size,
                bias=bias,
                nonlinearity=nonlinearity,
            )
            self._rnn = None
        else:
            self._rnn_cell = None
            self._rnn = nn.RNN(
                input_size,
                hidden_size,
                bias=bias,
                nonlinearity=nonlinearity,
                batch_first=False,
            )

        self._linear_y = nn.Linear(hidden_size, output_size, bias=True)
        self._scan_inputs = None
        self._x = None
        self._hx = None
        self._last_hx = None
        self._test_artifacts = test_artifacts

    @property
    def weight_hh(self):
        if self._rnn_type == 'PyTorch':
            return self._rnn_cell.weight_hh
        else:
            return self._rnn.weight_hh_l0

    @property
    def bias_hh(self):
        if self._rnn_type == 'PyTorch':
            return self._rnn_cell.bias_hh
        else:
            return self._rnn.bias_hh_l0

    @property
    def weight_ih(self):
        if self._rnn_type == 'PyTorch':
            return self._rnn_cell.weight_ih
        else:
            return self._rnn.weight_ih_l0

    @property
    def bias_ih(self):
        if self._rnn_type == 'PyTorch':
            return self._rnn_cell.bias_ih
        else:
            return self._rnn.bias_ih_l0

    def _forward_cuDNN(self, x, hx):
        if self.training and self._mode in {'blelloch'}:
            self._hx[-1, :, :] = hx

        hx = hx.view(1, -1, self._hidden_size)
        x = x.view(x.size(0), x.size(1),
                   self._input_size).transpose(0, 1).contiguous()

        output, hx = self._rnn(x, hx.view(1, -1, self._hidden_size))
        if self.training and self._mode in {'blelloch'}:
            scan.reverse_seq(self._hx[:-1, :, :], output[:-1, :, :])
            scan.reverse_seq(self._x, x)
            scan.fill_inputs2(self._scan_inputs, self.weight_hh, output)

        # Debug
        if self._test_artifacts is not None:
            for i in range(output.size(0)):
                self._test_artifacts.add_artifact('hx_{}'.format(i),
                                                  output[i, :, :])
        return hx.view(-1, self._hidden_size)

    def _forward_PyTorch(self, x, hx):
        """ Side-effects:
        1. self._hx
        2. self._x
        3. self._scan_inputs
        4. self._test_artifacts
        """
        for i in range(x.size(1)):
            if self.training and self._mode in {'blelloch'}:
                self._hx[x.size(1) - 1 - i, :, :] = hx
                self._x[x.size(1) - 1 - i, :, :] = x[:, i].view(
                    -1, self._input_size)

            hx = self._rnn_cell(x[:, i].unsqueeze(1), hx)

            if self.training and self._mode in {'blelloch'}:
                scan.fill_inputs(self._scan_inputs, self.weight_hh, hx, i)

            # Debug.
            if self._test_artifacts is not None:
                self._test_artifacts.add_artifact('hx_{}'.format(i), hx)
        return hx

    def forward(self, x):
        hx = torch.zeros((x.size(0), self._hidden_size),
                         device=self._linear_y.weight.device)

        if self.training and self._mode in {'blelloch'}:
            if self._scan_inputs is None:
                # x.size(1) number of fully-connected and activation,
                # 1 for the grad vec.
                scan_length = 2 * x.size(1) + 1
                self._scan_inputs = torch.zeros(
                    (scan_length, x.size(0), self._hidden_size,
                     self._hidden_size),
                    dtype=x.dtype,
                    device=x.device,
                )
            else:
                self._scan_inputs.zero_()

            if self._x is None:
                self._x = torch.zeros(
                    (x.size(1), x.size(0), self._input_size),
                    dtype=x.dtype,
                    device=x.device,
                )
            #else:
            #    self._x.zero_()

            if self._hx is None:
                self._hx = torch.zeros(
                    (x.size(1), x.size(0), self._hidden_size),
                    dtype=hx.dtype,
                    device=hx.device,
                )
            #else:
            #    self._hx.zero_()

        if self._rnn_type == 'PyTorch':
            forward_fn = lambda x, hx: self._forward_PyTorch(x, hx)
        else:
            forward_fn = lambda x, hx: self._forward_cuDNN(x, hx)
        if self.training and self._mode in {'blelloch'}:
            with torch.no_grad():
                hx = forward_fn(x, hx)
                self._last_hx = hx
            hx.requires_grad = True
        else:
            hx = forward_fn(x, hx)

        return self._linear_y(hx)

    def backward_by_scan(self, loss):
        # Figure out the gradient of loss to last_hx
        (self._scan_inputs[0, :, 0, :], self._linear_y.weight.grad,
         self._linear_y.bias.grad) = torch.autograd.grad(
             loss, [self._last_hx, self._linear_y.weight, self._linear_y.bias])

        scan_results = scan.scan(self._scan_inputs)
        dl_dz = scan_results[2::2, :, :].contiguous()

        self.weight_hh.grad = torch.bmm(
            dl_dz.unsqueeze(3).view(-1, self._hidden_size, 1),
            self._hx.unsqueeze(2).view(-1, 1, self._hidden_size)).sum(dim=0)
        self.weight_ih.grad = torch.bmm(
            dl_dz.unsqueeze(3).view(-1, self._hidden_size, 1),
            self._x.unsqueeze(2).view(-1, 1, self._input_size)).sum(dim=0)

        bias_grad = dl_dz.sum(dim=(0, 1))
        self.bias_hh.grad, self.bias_ih.grad = bias_grad, bias_grad

    def update_test_artifacts(self, test_artifacts):
        test_artifacts.add_artifact('rnn.weight_hh', self.weight_hh)
        test_artifacts.add_artifact('rnn.bias_hh', self.bias_hh)
        test_artifacts.add_artifact('rnn.weight_ih', self.weight_ih)
        test_artifacts.add_artifact('rnn.bias_ih', self.bias_ih)
        test_artifacts.add_artifact('_linear_y.weight', self._linear_y.weight)
        test_artifacts.add_artifact('_linear_y.bias', self._linear_y.bias)

        test_artifacts.add_artifact('rnn.weight_hh.grad', self.weight_hh.grad)
        test_artifacts.add_artifact('rnn.bias_hh.grad', self.bias_hh.grad)
        test_artifacts.add_artifact('rnn.weight_ih.grad', self.weight_ih.grad)
        test_artifacts.add_artifact('rnn.bias_ih.grad', self.bias_ih.grad)
        test_artifacts.add_artifact('_linear_y.weight.grad',
                                    self._linear_y.weight.grad)
        test_artifacts.add_artifact('_linear_y.bias.grad',
                                    self._linear_y.bias.grad)


def build_dataloaders(save_dir, train_batch_size, test_batch_size):
    train_X = torch.load(os.path.join(save_dir, 'train_X'))
    train_Y = torch.load(os.path.join(save_dir, 'train_Y'))
    test_X = torch.load(os.path.join(save_dir, 'test_X'))
    test_Y = torch.load(os.path.join(save_dir, 'test_Y'))
    num_classes = (max(train_Y.max(), test_Y.max()) -
                   min(train_Y.min(), test_Y.min()) + 1)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_X, train_Y),
        batch_size=train_batch_size,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_X, test_Y),
        batch_size=test_batch_size,
        shuffle=True,
    )
    return train_loader, test_loader, int(num_classes.item())


class UnitTestArtifacts(object):

    def __init__(self):
        self._record = []

    def new_timeframe(self):
        self._record.append({})

    def add_artifact(self, k, v):
        self._record[-1][k] = v.detach().cpu()

    def assert_allclose(self, expected):
        for i, expected_artifacts in enumerate(expected._record):
            print('************************************************')
            print('****************** epoch = {} ******************'.format(i))
            print('************************************************')
            for k, expected_artifact in expected_artifacts.items():
                print('++++++++++++++ Compare {} ++++++++++++++'.format(k))
                try:
                    np.testing.assert_allclose(self._record[i][k].numpy(),
                                               expected_artifact.numpy(),
                                               rtol=1e-4)
                    print('Done!')
                except AssertionError as e:
                    print(e)


def seed_this_process(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def accuracy(y, y_):
    with torch.no_grad():
        _, indices = y.max(1)
        return (indices == y_).type(dtype=torch.float).mean().item()


def main(args):
    # Redirect stdout and stderr outputs.
    if args.stdout is not None:
        sys.stdout = open(args.stdout, 'w', buffering=1)
    if args.stderr is not None:
        sys.stderr = open(args.stderr, 'w', buffering=1)

    test_artifacts = None if args.unit_test is None else UnitTestArtifacts()
    # Fix a seed.
    seed_this_process(args.seed)
    train_loader, test_loader, num_classes = build_dataloaders(
        args.save_dir, args.train_batch_size, args.test_batch_size)

    rnn = RNN(
        1,
        args.hidden_size,
        num_classes,
        mode=args.mode,
        rnn_type=args.rnn_type,
        test_artifacts=test_artifacts,
    ).cuda()
    loss_fn = nn.CrossEntropyLoss()

    def zero_grad(self):
        r"""Clears the gradients of all optimized :class:`torch.Tensor` s."""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    try:
                        p.grad.detach_()
                        p.grad.zero_()
                    except RuntimeError as e:
                        p.grad = None
                        pass

    torch.optim.Optimizer.zero_grad = zero_grad
    optimizer = optim.Adam(rnn.parameters(), lr=args.learning_rate)

    epoch_latency = {
        'epoch': [],
        'timestamp': [],
        'latency': [],
        'loss': [],
        'accuracy': [],
    }
    epoch_events = []

    def record(epoch, event_start, event_stop, loss, accuracy):
        epoch_latency['epoch'].append(epoch)
        epoch_events.append((event_start, event_stop))
        epoch_latency['loss'].append(loss)
        epoch_latency['accuracy'].append(accuracy)

    def train():
        for i, batch in enumerate(train_loader):
            if args.num_iterations is not None and i >= args.num_iterations:
                break
            if args.unit_test is not None:
                test_artifacts.new_timeframe()

            optimizer.zero_grad()
            x = batch[0].cuda()
            y_ = batch[1].type(dtype=torch.long).cuda()

            y = rnn(x)
            loss = loss_fn(y, y_)

            if args.mode == 'blelloch':
                rnn.backward_by_scan(loss)
            elif args.mode == 'normal':
                loss.backward()
            elif args.mode == 'normal-nobp':
                for param in rnn.parameters():
                    param.grad = torch.zeros_like(param)
            else:
                raise RuntimeError('Impossible to reach here!')

            optimizer.step()

            if args.unit_test is not None:
                rnn.update_test_artifacts(test_artifacts)

    def train_loss():
        running_loss = []
        running_acc = []
        rnn.eval()
        with torch.no_grad():
            for i, batch in enumerate(train_loader):
                if args.num_iterations is not None and i >= args.num_iterations:
                    break

                x = batch[0].cuda()
                y_ = batch[1].type(dtype=torch.long).cuda()

                y = rnn(x)
                loss = loss_fn(y, y_)

                running_loss.append(loss.item())
                running_acc.append(accuracy(y, y_))
        rnn.train()
        return np.mean(running_loss), np.mean(running_acc)

    # The training loop:
    for epoch in range(args.num_epochs):
        epoch_start = torch.cuda.Event(enable_timing=True)
        epoch_stop = torch.cuda.Event(enable_timing=True)
        epoch_start.record()
        train()
        epoch_stop.record()
        epoch_loss, epoch_acc = train_loss() if args.save_loss_acc else (0, 0)
        record(epoch, epoch_start, epoch_stop, epoch_loss, epoch_acc)

    if args.unit_test == 'expected':
        torch.save(test_artifacts, args.unit_test_cache)
    elif args.unit_test == 'actual':
        expected = torch.load(args.unit_test_cache)
        test_artifacts.assert_allclose(expected)

    torch.cuda.synchronize()
    clock = 0.0
    for i, event_pair in enumerate(epoch_events):
        lat = event_pair[0].elapsed_time(event_pair[1]) / 1000
        clock += lat
        epoch_latency['timestamp'].append(clock)
        epoch_latency['latency'].append(lat)
        print('[{}/{} @ {} s] lat: {} s; loss: {}; acc: {}'.format(
            epoch_latency['epoch'][i],
            args.num_epochs,
            epoch_latency['timestamp'][i],
            epoch_latency['latency'][i],
            epoch_latency['loss'][i],
            epoch_latency['accuracy'][i],
        ))
    if args.save_epoch_latency is not None:
        pd.DataFrame(epoch_latency).to_csv(args.save_epoch_latency)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--save-dir',
        type=str,
        required=True,
        default='./syn_data/',
    )
    parser.add_argument('--num-epochs', type=int, required=False, default=50)
    parser.add_argument('--hidden-size', type=int, required=False, default=20)
    parser.add_argument(
        '--num-iterations',
        type=int,
        required=False,
        default=None,
    )
    parser.add_argument(
        '--train-batch-size',
        type=int,
        required=False,
        default=16,
    )
    parser.add_argument(
        '--test-batch-size',
        type=int,
        required=False,
        default=512,
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        required=False,
        default=0.00003,
    )
    parser.add_argument('--print-freq', type=int, required=False, default=100)
    parser.add_argument(
        '--mode',
        type=str,
        required=False,
        default='blelloch',
        choices=['blelloch', 'normal', 'normal-nobp'],
    )
    parser.add_argument(
        '--rnn-type',
        type=str,
        required=False,
        default='cuDNN',
        choices=['PyTorch', 'cuDNN'],
    )
    parser.add_argument('--seed', type=int, required=False, default=4202)
    # For unit-testing:
    parser.add_argument(
        '--unit-test',
        type=str,
        choices=['expected', 'actual'],
        required=False,
        default=None,
    )
    parser.add_argument(
        '--unit-test-cache',
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument('--stdout', type=str, required=False, default=None)
    parser.add_argument('--stderr', type=str, required=False, default=None)
    parser.add_argument(
        '--save-epoch-latency',
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument('--save-loss-acc', action='store_true', required=False)
    main(parser.parse_args())
