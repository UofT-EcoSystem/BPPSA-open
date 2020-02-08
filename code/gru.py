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


class GRU(nn.Module):

    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 bias=True,
                 mode='blelloch',
                 rnn_type='cuDNN',
                 test_artifacts=None):
        super(GRU, self).__init__()
        self._input_size = input_size
        self._hidden_size = hidden_size
        assert mode in {'normal', 'blelloch', 'normal-nobp', 'blelloch-nobp'}
        self._mode = mode
        assert rnn_type in {'cuDNN'}
        self._rnn_type = rnn_type
        self._gru = nn.GRU(
            input_size,
            hidden_size,
            bias=bias,
            batch_first=False,
        )

        self._linear_y = nn.Linear(hidden_size, output_size, bias=True)
        self._scan_inputs = None
        self._last_hx = None
        self._test_artifacts = test_artifacts

        self._h_t_prev = None
        self._x = None

    @property
    def weight_ih(self):
        return self._gru.weight_ih_l0

    @property
    def weight_hh(self):
        return self._gru.weight_hh_l0

    @property
    def bias_ih(self):
        return self._gru.bias_ih_l0

    @property
    def bias_hh(self):
        return self._gru.bias_hh_l0

    @property
    def W_ir(self):
        return self._gru.weight_ih_l0[:self._hidden_size]

    @property
    def W_ir_grad(self):
        return self._gru.weight_ih_l0.grad[:self._hidden_size]

    @property
    def W_iz(self):
        return self._gru.weight_ih_l0[self._hidden_size:2 * self._hidden_size]

    @property
    def W_iz_grad(self):
        return self._gru.weight_ih_l0.grad[self._hidden_size:2 *
                                           self._hidden_size]

    @property
    def W_in(self):
        return self._gru.weight_ih_l0[2 * self._hidden_size:3 *
                                      self._hidden_size]

    @property
    def W_in_grad(self):
        return self._gru.weight_ih_l0.grad[2 * self._hidden_size:3 *
                                           self._hidden_size]

    @property
    def b_ir(self):
        return self._gru.bias_ih_l0[:self._hidden_size]

    @property
    def b_ir_grad(self):
        return self._gru.bias_ih_l0.grad[:self._hidden_size]

    @property
    def b_iz(self):
        return self._gru.bias_ih_l0[self._hidden_size:2 * self._hidden_size]

    @property
    def b_iz_grad(self):
        return self._gru.bias_ih_l0.grad[self._hidden_size:2 *
                                         self._hidden_size]

    @property
    def b_in(self):
        return self._gru.bias_ih_l0[2 * self._hidden_size:3 * self._hidden_size]

    @property
    def b_in_grad(self):
        return self._gru.bias_ih_l0.grad[2 * self._hidden_size:3 *
                                         self._hidden_size]

    @property
    def W_hr(self):
        return self._gru.weight_hh_l0[:self._hidden_size]

    @property
    def W_hr_grad(self):
        return self._gru.weight_hh_l0.grad[:self._hidden_size]

    @property
    def W_hz(self):
        return self._gru.weight_hh_l0[self._hidden_size:2 * self._hidden_size]

    @property
    def W_hz_grad(self):
        return self._gru.weight_hh_l0.grad[self._hidden_size:2 *
                                           self._hidden_size]

    @property
    def W_hn(self):
        return self._gru.weight_hh_l0[2 * self._hidden_size:3 *
                                      self._hidden_size]

    @property
    def W_hn_grad(self):
        return self._gru.weight_hh_l0.grad[2 * self._hidden_size:3 *
                                           self._hidden_size]

    @property
    def b_hr(self):
        return self._gru.bias_hh_l0[:self._hidden_size]

    @property
    def b_hr_grad(self):
        return self._gru.bias_hh_l0.grad[:self._hidden_size]

    @property
    def b_hz(self):
        return self._gru.bias_hh_l0[self._hidden_size:2 * self._hidden_size]

    @property
    def b_hz_grad(self):
        return self._gru.bias_hh_l0.grad[self._hidden_size:2 *
                                         self._hidden_size]

    @property
    def b_hn(self):
        return self._gru.bias_hh_l0[2 * self._hidden_size:3 * self._hidden_size]

    @property
    def b_hn_grad(self):
        return self._gru.bias_hh_l0.grad[2 * self._hidden_size:3 *
                                         self._hidden_size]

    def _get_shapes(self):
        """ Return shapes of W and b to make them broadcastable. """
        W_ih_shape = (1, 1, self._hidden_size, self._input_size)
        b_shape = (1, 1, self._hidden_size, 1)
        W_hh_shape = (1, 1, self._hidden_size, self._hidden_size)
        return W_ih_shape, b_shape, W_hh_shape

    def _compute_gates(self, x, h_t_prev):
        """ Takes x, h_t_prev,
        returns (x, h_t_prev, R, r, M, N, n, Z, z).
        No side-effects.
        """
        seq_len, batch_size = x.size(0), x.size(1)
        x = x.view(seq_len, batch_size, self._input_size, 1)
        h_t_prev = h_t_prev.view(seq_len, batch_size, self._hidden_size, 1)
        W_ih_shape, b_shape, W_hh_shape = self._get_shapes()

        IH = torch.matmul(
            self.weight_ih.view(1, 1, 3 * self._hidden_size, self._input_size),
            x,
        )
        HH = torch.matmul(
            self.weight_hh.view(1, 1, 3 * self._hidden_size, self._hidden_size),
            h_t_prev,
        )

        # R = W_ir * x + b_ir + W_hr * h_t-1 + b_hr
        R = (IH[:, :, :self._hidden_size, :] + self.b_ir.view(*b_shape) +
             HH[:, :, :self._hidden_size, :] + self.b_hr.view(*b_shape))
        # r = sigmoid(R)
        r = torch.sigmoid(R)
        # M = W_hn * h_t-1 + b_hn
        M = (HH[:, :, 2 * self._hidden_size:3 * self._hidden_size, :] +
             self.b_hn.view(*b_shape))
        # N = W_in * x + b_in + r .* M
        N = (IH[:, :, 2 * self._hidden_size:3 * self._hidden_size, :] +
             self.b_in.view(*b_shape) + r * M)
        # n = tanh(N)
        n = torch.tanh(N)
        # Z = W_iz * x + b_iz + W_hz * h_t-1 + b_hz
        Z = (IH[:, :, self._hidden_size:2 * self._hidden_size, :] +
             self.b_iz.view(*b_shape) +
             HH[:, :, self._hidden_size:2 * self._hidden_size, :] +
             self.b_hz.view(*b_shape))
        # z = sigmoid(z)
        z = torch.sigmoid(Z)

        return (x, h_t_prev, R, r, M, N, n, Z, z)

    def _compute_jcbT(self, inner_states):
        """ Takes (x, h_t_prev, R, r, M, N, n, Z, z),
        returns (d h_t/d h_{t-1})^T,
        saves
          [(d h_t/d N_t)^T, (d h_t/d M_t)^T, (d h_t/d R_t)^T, (d h_t/d Z_t)^T]
        in
          [self._jcbT_6_7, self._jcbT_5_6_7, self._jcbT_2_3_6_7,
           self._jcbT_9_10]
        """
        x, h_t_prev, R, r, M, N, n, Z, z = inner_states
        seq_len, batch_size = x.size(0), x.size(1)
        W_ih_shape, b_shape, W_hh_shape = self._get_shapes()
        diag_embed_shape = (seq_len, batch_size, self._hidden_size)

        jcbT_1 = self.W_hr.transpose(0, 1).view(*W_hh_shape)
        jcbT_2_diag = r * (1 - r)
        jcbT_3_diag = M
        jcbT_4 = self.W_hn.transpose(0, 1).view(*W_hh_shape)
        jcbT_5_diag = r
        jcbT_6_diag = (1 - n**2)
        jcbT_7_diag = (1 - z)
        jcbT_8 = self.W_hz.transpose(0, 1).view(*W_hh_shape)
        jcbT_9_diag = z * (1 - z)
        jcbT_10_diag = h_t_prev - n
        jcbT_11 = torch.diag_embed(z.view(*diag_embed_shape))

        self._jcbT_6_7_diag = jcbT_6_diag * jcbT_7_diag
        self._jcbT_5_6_7_diag = jcbT_5_diag * self._jcbT_6_7_diag
        jcbT_2_3_diag = jcbT_2_diag * jcbT_3_diag
        self._jcbT_2_3_6_7_diag = jcbT_2_3_diag * self._jcbT_6_7_diag
        self._jcbT_9_10_diag = jcbT_9_diag * jcbT_10_diag

        jcbT_1_2_3 = jcbT_1 * jcbT_2_3_diag.transpose(2, 3)
        jcbT_4_5 = jcbT_4 * jcbT_5_diag.transpose(2, 3)
        jcbT_8_9_10 = jcbT_8 * self._jcbT_9_10_diag.transpose(2, 3)
        return ((jcbT_1_2_3 + jcbT_4_5) * self._jcbT_6_7_diag.transpose(2, 3) +
                jcbT_8_9_10 + jcbT_11)

    def _forward_cuDNN(self, x, hx):
        """ Side-effects:
        1. self._h_t_prev
        2. self._x
        3. self._scan_inputs
        4. See _compute_jcbT.
        """
        if self.training and self._mode in {'blelloch', 'blelloch-nobp'}:
            self._h_t_prev[0, :, :] = hx

        x = x.view(x.size(0), x.size(1), self._input_size).transpose(0, 1)

        if self.training and self._mode in {'blelloch', 'blelloch-nobp'}:
            self._x = x

        h_t, hx = self._gru(x, hx.view(1, -1, self._hidden_size))

        if self.training and self._mode in {'blelloch', 'blelloch-nobp'}:
            self._h_t_prev[1:, :, :] = h_t[:-1, :, :]

            self._scan_inputs[1:] = torch.flip(
                # Needs to prepare the jcbT of dh_t/dh_{t-1}, as well as for
                # computing the gradients for weights and biases.
                self._compute_jcbT(self._compute_gates(x, self._h_t_prev)),
                (0,))

        # Debug
        if self._test_artifacts is not None:
            for i in range(h_t.size(0)):
                self._test_artifacts.add_artifact(
                    'hx_{}'.format(i),
                    h_t[i, :, :],
                )
        return hx.view(-1, self._hidden_size)

    def forward(self, x):
        hx = torch.zeros((x.size(0), self._hidden_size),
                         device=self._linear_y.weight.device)

        if self.training and self._mode in {'blelloch', 'blelloch-nobp'}:
            seq_len, batch_size = x.size(1), x.size(0)
            if (self._h_t_prev is None or self._h_t_prev.size(0) != seq_len or
                    self._h_t_prev.size(1) != batch_size):
                self._h_t_prev = torch.zeros(
                    (seq_len, batch_size, self._hidden_size),
                    dtype=x.dtype,
                    device=x.device,
                )
            if (self._scan_inputs is None or
                    self._scan_inputs.size(0) != seq_len + 1 or
                    self._scan_inputs.size(1) != batch_size):
                self._scan_inputs = torch.zeros(
                    (seq_len + 1, batch_size, self._hidden_size,
                     self._hidden_size),
                    dtype=x.dtype,
                    device=x.device,
                )

        forward_fn = lambda x, hx: self._forward_cuDNN(x, hx)
        if self.training and self._mode in {'blelloch', 'blelloch-nobp'}:
            with torch.no_grad():
                hx = forward_fn(x, hx)
                self._last_hx = hx
            hx.requires_grad = True
        else:
            hx = forward_fn(x, hx)

        return self._linear_y(hx)

    def backward_by_scan(self, loss):
        seq_len, batch_size = self._x.size(0), self._x.size(1)
        # Figure out the gradient of loss to last_hx
        (self._scan_inputs[0, :, 0, :], self._linear_y.weight.grad,
         self._linear_y.bias.grad) = torch.autograd.grad(
             loss, [self._last_hx, self._linear_y.weight, self._linear_y.bias])

        grad_l_over_h_t = scan.scan(self._scan_inputs)
        grad_l_over_h_t = torch.flip(grad_l_over_h_t[1:, :, :], (0,))
        orig_x_shape = (seq_len, batch_size, self._input_size)
        orig_h_shape = (seq_len, batch_size, self._hidden_size)
        x = self._x
        h_t_prev = self._h_t_prev
        jcbT_2_3_6_7_diag = self._jcbT_2_3_6_7_diag.view(*orig_h_shape)
        jcbT_9_10_diag = self._jcbT_9_10_diag.view(*orig_h_shape)
        jcbT_6_7_diag = self._jcbT_6_7_diag.view(*orig_h_shape)
        jcbT_5_6_7_diag = self._jcbT_5_6_7_diag.view(*orig_h_shape)

        grad_l_over_R = jcbT_2_3_6_7_diag * grad_l_over_h_t
        b_ir_grad = grad_l_over_R.sum(dim=(0, 1))
        b_hr_grad = b_ir_grad
        W_ir_grad = torch.matmul(grad_l_over_R.transpose(1, 2), x).sum(dim=0)
        W_hr_grad = torch.matmul(grad_l_over_R.transpose(1, 2),
                                 h_t_prev).sum(dim=0)

        grad_l_over_Z = jcbT_9_10_diag * grad_l_over_h_t
        b_iz_grad = grad_l_over_Z.sum(dim=(0, 1))
        b_hz_grad = b_iz_grad
        W_iz_grad = torch.matmul(grad_l_over_Z.transpose(1, 2), x).sum(dim=0)
        W_hz_grad = torch.matmul(grad_l_over_Z.transpose(1, 2),
                                 h_t_prev).sum(dim=0)

        grad_l_over_N = jcbT_6_7_diag * grad_l_over_h_t
        grad_l_over_M = jcbT_5_6_7_diag * grad_l_over_h_t
        b_in_grad = grad_l_over_N.sum(dim=(0, 1))
        b_hn_grad = grad_l_over_M.sum(dim=(0, 1))
        W_in_grad = torch.matmul(grad_l_over_N.transpose(1, 2), x).sum(dim=0)
        W_hn_grad = torch.matmul(grad_l_over_M.transpose(1, 2),
                                 h_t_prev).sum(dim=0)

        self.weight_ih.grad = torch.cat((W_ir_grad, W_iz_grad, W_in_grad),
                                        dim=0)
        self.weight_hh.grad = torch.cat((W_hr_grad, W_hz_grad, W_hn_grad),
                                        dim=0)
        self.bias_ih.grad = torch.cat((b_ir_grad, b_iz_grad, b_in_grad), dim=0)
        self.bias_hh.grad = torch.cat((b_hr_grad, b_hz_grad, b_hn_grad), dim=0)

    def update_test_artifacts(self, test_artifacts):
        test_artifacts.add_artifact('rnn.W_ir', self.W_ir)
        test_artifacts.add_artifact('rnn.W_ir.grad', self.W_ir_grad)
        test_artifacts.add_artifact('rnn.W_iz', self.W_iz)
        test_artifacts.add_artifact('rnn.W_iz.grad', self.W_iz_grad)
        test_artifacts.add_artifact('rnn.W_in', self.W_in)
        test_artifacts.add_artifact('rnn.W_in.grad', self.W_in_grad)

        test_artifacts.add_artifact('rnn.W_hr', self.W_hr)
        test_artifacts.add_artifact('rnn.W_hr.grad', self.W_hr_grad)
        test_artifacts.add_artifact('rnn.W_hz', self.W_hz)
        test_artifacts.add_artifact('rnn.W_hz.grad', self.W_hz_grad)
        test_artifacts.add_artifact('rnn.W_hn', self.W_hn)
        test_artifacts.add_artifact('rnn.W_hn.grad', self.W_hn_grad)

        test_artifacts.add_artifact('rnn.b_ir', self.b_ir)
        test_artifacts.add_artifact('rnn.b_ir.grad', self.b_ir_grad)
        test_artifacts.add_artifact('rnn.b_iz', self.b_iz)
        test_artifacts.add_artifact('rnn.b_iz.grad', self.b_iz_grad)
        test_artifacts.add_artifact('rnn.b_in', self.b_in)
        test_artifacts.add_artifact('rnn.b_in.grad', self.b_in_grad)

        test_artifacts.add_artifact('rnn.b_hr', self.b_hr)
        test_artifacts.add_artifact('rnn.b_hr.grad', self.b_hr_grad)
        test_artifacts.add_artifact('rnn.b_hz', self.b_hz)
        test_artifacts.add_artifact('rnn.b_hz.grad', self.b_hz_grad)
        test_artifacts.add_artifact('rnn.b_hn', self.b_hn)
        test_artifacts.add_artifact('rnn.b_hn.grad', self.b_hn_grad)

        test_artifacts.add_artifact('_linear_y.weight.grad',
                                    self._linear_y.weight.grad)
        test_artifacts.add_artifact('_linear_y.bias.grad',
                                    self._linear_y.bias.grad)


def build_dataloaders(save_dir, train_batch_size, test_batch_size):
    train_X = torch.load(os.path.join(save_dir, 'train_X')).cuda()
    X_size = train_X.size()
    train_Y = torch.load(os.path.join(
        save_dir,
        'train_Y',
    )).type(dtype=torch.long).cuda()
    #test_X = torch.load(os.path.join(save_dir, 'test_X'))
    #test_Y = torch.load(os.path.join(save_dir, 'test_Y'))
    #num_classes = (max(train_Y.max(), test_Y.max()) -
    #               min(train_Y.min(), test_Y.min()) + 1)
    num_classes = (train_Y.max() - train_Y.min() + 1)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_X, train_Y),
        batch_size=train_batch_size,
        shuffle=True,
    )
    #test_loader = torch.utils.data.DataLoader(
    #    torch.utils.data.TensorDataset(test_X, test_Y),
    #    batch_size=test_batch_size,
    #    shuffle=True,
    #)
    test_loader = None
    return train_loader, test_loader, int(num_classes.item()), X_size


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
    train_loader, test_loader, num_classes, X_size = build_dataloaders(
        args.save_dir, args.train_batch_size, args.test_batch_size)

    input_size = 1 if len(X_size) == 2 else X_size[2]
    rnn = GRU(
        input_size,
        args.hidden_size,
        num_classes,
        mode=args.mode,
        rnn_type=args.rnn_type,
        test_artifacts=test_artifacts,
    ).cuda()
    loss_fn = nn.CrossEntropyLoss()

    if args.mode in {'blelloch', 'blelloch-nobp'}:
        torch.optim.Optimizer.zero_grad = lambda self: None
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
            x = batch[0]
            y_ = batch[1]

            y = rnn(x)
            loss = loss_fn(y, y_)

            if args.mode == 'blelloch':
                rnn.backward_by_scan(loss)
            elif args.mode == 'normal':
                loss.backward()
            elif args.mode in {'normal-nobp', 'blelloch-nobp'}:
                pass
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

                x = batch[0]
                y_ = batch[1]

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
        choices=['blelloch', 'normal', 'normal-nobp', 'blelloch-nobp'],
    )
    parser.add_argument(
        '--rnn-type',
        type=str,
        required=False,
        default='cuDNN',
        choices=['cuDNN'],
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
