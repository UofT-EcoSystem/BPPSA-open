import argparse
import math
import matplotlib.pyplot as plt
import os
import re
import torch
import torch.utils.data

from torch.distributions.normal import Normal
from random import randrange, randint


def has_pattern(seq, pattern):
    # Cache the compiled regex.
    if not hasattr(has_pattern, 'pattern_db'):
        has_pattern.pattern_db = {}
    if pattern not in has_pattern.pattern_db:
        has_pattern.pattern_db[pattern] = re.compile(pattern)

    return (has_pattern.pattern_db[pattern].search(seq) is not None)


def bit_tensor_to_str(t):
    res = ''
    for bit in t.type(dtype=torch.uint8).tolist():
        res += str(bit)
    return res


def regen_bitvec(seq_len):
    return torch.bernoulli(torch.full((seq_len,), 0.5, dtype=torch.float32))


def mutate_once(X, Y, pattern, option='1to0'):
    num_samples = X.size(0)
    seq_len = X.size(1)

    def cond(bitvec):
        res = has_pattern(bit_tensor_to_str(bitvec), pattern)
        if option == '1to0':
            return (not res)
        else:
            return res

    assert option in {'1to0', '0to1'}
    mutate_idx = randrange(num_samples)
    while cond(X[mutate_idx]):
        mutate_idx = randrange(num_samples)
    y_to_assert = Y[mutate_idx].type(dtype=torch.uint8).item()
    if option == '1to0':
        assert y_to_assert == 1
    else:
        assert y_to_assert == 0
    new_bitvec = regen_bitvec(seq_len)
    while not cond(new_bitvec):
        new_bitvec = regen_bitvec(seq_len)
    X[mutate_idx] = new_bitvec
    Y[mutate_idx] = 0.0 if option == '1to0' else 1.0


def gen_XY1(num_samples, seq_len, pattern):
    X = torch.bernoulli(
        torch.full((num_samples, seq_len), 0.5, dtype=torch.float32))
    Y = torch.zeros((num_samples,), dtype=torch.float32)
    for i in range(X.size(0)):
        seq = bit_tensor_to_str(X[i])
        if has_pattern(seq, pattern):
            Y[i] = 1.0
    # Balance the two class:
    balance = num_samples / 2 - Y.sum(dim=0).type(dtype=torch.int32).item()
    while balance != 0:
        if balance < 0:  # Too many samples have the sequence.
            mutate_once(X, Y, pattern, option='1to0')
            balance += 1
        else:  # Too many samples don't have the sequence.
            mutate_once(X, Y, pattern, option='0to1')
            balance -= 1
    return X, Y


def gen_XY2(num_samples, seq_len):
    # Generate
    X = torch.zeros((num_samples, seq_len), dtype=torch.float32)
    Y = torch.zeros((num_samples,), dtype=torch.float32)
    for i in range(X.size(0)):
        if randint(0, 1) == 1:
            X[i, :] = torch.bernoulli(
                torch.full((seq_len,), 0.4, dtype=torch.float32))
            Y[i] = 1.0
        else:
            X[i, :] = torch.bernoulli(
                torch.full((seq_len,), 0.6, dtype=torch.float32))
    return X, Y


def gen_XY3(num_samples, seq_len):
    # Generate
    X = torch.zeros((num_samples, seq_len), dtype=torch.float32)
    Y = torch.zeros((num_samples,), dtype=torch.float32)
    base_freqs = [math.pi / 10, math.pi / 100]
    freq_noises = []
    for base_freq in base_freqs:
        freq_noises.append(
            Normal(torch.tensor([0.0]), torch.tensor([base_freq * 0.1])))

    for i in range(X.size(0)):
        if randint(0, 1) == 1:
            freq = base_freqs[0] + freq_noises[0].sample()
            Y[i] = 1.0
        #    color = 'r'
        else:
            freq = base_freqs[1] + freq_noises[1].sample()
        #    color = 'b'
        X[i, :] = (torch.arange(seq_len, dtype=torch.float) * freq +
                   torch.rand(1, dtype=torch.float) * 2 * math.pi).cos()
        #if i in {50, 500, 5000, 10000}:
        #    plt.plot(torch.arange(seq_len, dtype=torch.float).numpy(),
        #             X[i, :].numpy(),
        #             color=color)
    #plt.show()
    return X, Y


def gen_XY4(num_samples, seq_len):
    # Generate
    X = torch.zeros((num_samples, seq_len), dtype=torch.float32)
    Y = torch.zeros((num_samples,), dtype=torch.float32)
    for i in range(X.size(0)):
        p = randint(0, 9)
        X[i, :] = torch.bernoulli(
            torch.full((seq_len,), 0.05 + p * 0.1, dtype=torch.float32))
        Y[i] = p

    return X, Y


def main(args):
    if args.dataset == 1:
        gen_XY = gen_XY1
    elif args.dataset == 2:
        gen_XY = gen_XY2
    elif args.dataset == 3:
        gen_XY = gen_XY3
    elif args.dataset == 4:
        gen_XY = gen_XY4
    else:
        raise RuntimeError('Invalid dataset choice!')

    X, Y = gen_XY(args.num_samples, args.seq_len)
    dataset = torch.utils.data.TensorDataset(X, Y)
    train_size = int(args.train_test_ratio * args.num_samples)
    test_size = args.num_samples - train_size
    trainset, testset = torch.utils.data.random_split(dataset,
                                                      [train_size, test_size])
    torch.save(trainset[:][0], os.path.join(args.save_dir, 'train_X'))
    torch.save(trainset[:][1], os.path.join(args.save_dir, 'train_Y'))
    torch.save(testset[:][0], os.path.join(args.save_dir, 'test_X'))
    torch.save(testset[:][1], os.path.join(args.save_dir, 'test_Y'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-samples',
                        type=int,
                        required=False,
                        default=50000)
    parser.add_argument('--seq-len', type=int, required=False, default=1000)
    parser.add_argument('--pattern',
                        type=str,
                        required=False,
                        default='0011100011')
    parser.add_argument('--save-dir',
                        type=str,
                        required=False,
                        default='./syn_data/')
    parser.add_argument('--train-test-ratio',
                        type=float,
                        required=False,
                        default=0.64)
    parser.add_argument('--dataset',
                        type=int,
                        required=True,
                        choices=[1, 2, 3, 4])
    main(parser.parse_args())
