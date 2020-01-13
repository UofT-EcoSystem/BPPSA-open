import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_csv(gpu, mode, seq_len, batch_size):
    return pd.read_csv(
        './rnn-epoch-latency-{}/{}-bernoulli{}_10-batch_size_{}.csv'.format(
            gpu, mode, seq_len, batch_size),
        index_col=0)


def plot_batch_sizes_to_speedups(gpu):
    batch_sizes = list(reversed([2, 4, 8, 16, 32, 64, 128, 256]))
    speedups = []
    speedup_limits = []
    for batch_size in batch_sizes:
        normal_latencies = read_csv(gpu, 'normal', 1000, batch_size)['latency']
        normal_nobp_latencies = read_csv(gpu, 'normal-nobp', 1000, batch_size)['latency']
        blelloch_latencies = read_csv(gpu, 'blelloch', 1000,
                                      batch_size)['latency']
        speedups.append(normal_latencies.mean() / blelloch_latencies.mean())
        speedup_limits.append(normal_latencies.mean() / normal_nobp_latencies.mean())

    print('Max total speedup for {} :  {}'.format(gpu, np.max(speedups)))
    plt.bar(np.arange(len(batch_sizes)),
            speedups,
            color='#5A9BD5',
            width=0.25,
            label=gpu)
    plt.bar(np.arange(len(batch_sizes)),
            speedup_limits,
            color='None',
            edgecolor='#5A9BD5',
            width=0.25,
            label='limit')
    plt.xticks(np.arange(len(batch_sizes)), batch_sizes)
    plt.axhline(y=1.0, color='r', linestyle='--', linewidth=3)
    plt.xlabel('Mini-batch Size')
    plt.ylabel('Speedup over Baseline (PyTorch/cuDNN)')
    plt.legend()
    plt.savefig('fig_10_d.png', bbox_inches='tight', pad_inches=0.0)
    plt.clf()


def plot_seq_lens_to_speedups(gpu):
    seq_lens = [10, 30, 100, 300, 1000, 3000, 10000, 30000]
    speedups = []
    speedup_limits = []
    for seq_len in seq_lens:
        normal_latencies = read_csv(gpu, 'normal', seq_len, 16)['latency']
        normal_nobp_latencies = read_csv(gpu, 'normal-nobp', seq_len, 16)['latency']
        blelloch_latencies = read_csv(gpu, 'blelloch', seq_len, 16)['latency']
        speedups.append(normal_latencies.mean() / blelloch_latencies.mean())
        speedup_limits.append(normal_latencies.mean() / normal_nobp_latencies.mean())

    print('Max total speedup for {} :  {}'.format(gpu, np.max(speedups)))
    plt.bar(np.arange(len(seq_lens)),
            speedups,
            color='#5A9BD5',
            width=0.25,
            label=gpu)
    plt.bar(np.arange(len(seq_lens)),
            speedup_limits,
            color='None',
            edgecolor='#5A9BD5',
            width=0.25,
            label='limit')
    plt.xticks(np.arange(len(seq_lens)),
               ['10', '30', '100', '300', '1k', '3k', '10k', '30k'])
    plt.axhline(y=1.0, color='r', linestyle='--', linewidth=3)
    plt.xlabel('Sequence Length')
    plt.ylabel('Speedup over Baseline (PyTorch/cuDNN)')
    plt.legend()
    plt.savefig('fig_10_b.png', bbox_inches='tight', pad_inches=0.0)
    plt.clf()

def read_training_csv(gpu, mode):
    return pd.read_csv(
        './rnn-epoch-latency-{}/training-curve-{}-bernoulli1000_10-batch_size_16.csv'.format(
            gpu, mode),
        index_col=0)


def plot_training_curve(gpu):
    normal = read_training_csv(gpu, 'normal')
    blelloch = read_training_csv(gpu, 'blelloch')
    print(normal['timestamp'])
    print(normal['loss'])
    plt.plot(normal['timestamp'],
             normal['loss'],
             color='r',
             label='PyTorch/cuDNN')
    plt.plot(blelloch['timestamp'],
             blelloch['loss'],
             color='#5A9BD5',
             label='BPPSA')
    plt.xlabel('Wall-clock Time (s)')
    plt.ylabel('Training Loss')
    plt.legend()
    plt.savefig('fig_9.png',
                bbox_inches='tight',
                pad_inches=0.0)
    plt.clf()


def plot_batch_sizes_to_backward_speedups(gpu):
    batch_sizes = list(reversed([2, 4, 8, 16, 32, 64, 128, 256]))
    speedups = []
    for batch_size in batch_sizes:
        normal_latencies = read_csv(gpu, 'normal', 1000, batch_size)['latency']
        blelloch_latencies = read_csv(gpu, 'blelloch', 1000,
                                      batch_size)['latency']
        normal_nobp_latencies = read_csv(gpu, 'normal-nobp', 1000,
                                         batch_size)['latency']
        speedups.append(
            (normal_latencies.mean() - normal_nobp_latencies.mean()) /
            (blelloch_latencies.mean() - normal_nobp_latencies.mean()))


    print('Max backward pass speedup for {} : {}'.format(gpu, np.max(speedups)))
    plt.bar(np.arange(len(batch_sizes)),
            speedups,
            color='#5A9BD5',
            width=0.25,
            label=gpu)
    plt.xticks(np.arange(len(batch_sizes)), batch_sizes)
    plt.axhline(y=1.0, color='r', linestyle='--', linewidth=3)
    plt.xlabel('Mini-batch Size')
    plt.ylabel('Speedup over Baseline (PyTorch/cuDNN)')
    plt.legend()
    plt.savefig('fig_10_c.png',
                bbox_inches='tight',
                pad_inches=0.0)
    plt.clf()


def plot_seq_lens_to_backward_speedups(gpu):
    seq_lens = [10, 30, 100, 300, 1000, 3000, 10000, 30000]
    speedups = []
    for seq_len in seq_lens:
        normal_latencies = read_csv(gpu, 'normal', seq_len, 16)['latency']
        blelloch_latencies = read_csv(gpu, 'blelloch', seq_len,
                                      16)['latency']
        normal_nobp_latencies = read_csv(gpu, 'normal-nobp', seq_len,
                                         16)['latency']
        speedups.append(
            (normal_latencies.mean() - normal_nobp_latencies.mean()) /
            (blelloch_latencies.mean() - normal_nobp_latencies.mean()))


    print('Max backward pass speedup for {} : {}'.format(gpu, np.max(speedups)))
    plt.bar(np.arange(len(seq_lens)),
            speedups,
            color='#5A9BD5',
            width=0.25,
            label=gpu)
    plt.xticks(np.arange(len(seq_lens)),
               ['10', '30', '100', '300', '1k', '3k', '10k', '30k'])
    plt.axhline(y=1.0, color='r', linestyle='--', linewidth=3)
    plt.xlabel('Sequence Length')
    plt.ylabel('Speedup over Baseline (PyTorch/cuDNN)')
    plt.legend()
    plt.savefig('fig_10_a',
                bbox_inches='tight',
                pad_inches=0.0)
    plt.clf()


def main(args):
    plt.rcParams.update({'font.size': 15})
    plot_batch_sizes_to_speedups(args.gpu)
    plot_seq_lens_to_speedups(args.gpu)
    plt.rcParams.update({'font.size': 12})
    plot_training_curve(args.gpu)
    plt.rcParams.update({'font.size': 15})
    plot_batch_sizes_to_backward_speedups(args.gpu)
    plot_seq_lens_to_backward_speedups(args.gpu)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, required=True)
    main(parser.parse_args())
