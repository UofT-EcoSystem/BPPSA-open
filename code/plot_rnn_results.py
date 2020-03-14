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
    fps = []
    bps = []
    bar_width = 0.25
    for batch_size in batch_sizes:
        normal_latencies = read_csv(gpu, 'normal', 1000, batch_size)['latency']
        blelloch_latencies = read_csv(gpu, 'blelloch', 1000,
                                      batch_size)['latency']
        normal_nobp_latencies = read_csv(gpu, 'normal-nobp', 1000,
                                         batch_size)['latency']
        baseline = normal_latencies.mean()

        fp = normal_nobp_latencies.mean() / baseline
        fps.append(fp)

        bp = ((blelloch_latencies.mean() - normal_nobp_latencies.mean()) /
              baseline)
        bps.append(bp)

        mean_speedup = normal_latencies.mean() / blelloch_latencies.mean()
        speedups.append(mean_speedup)

    print('{}: {}'.format(gpu, np.max(speedups)))
    plt.bar(
        np.arange(len(batch_sizes)),
        fps,
        edgecolor='#5A9BD5',
        color='white',
        hatch='////',
        alpha=0.99,
        width=bar_width,
        label=gpu + ', FP',
    )
    plt.bar(
        np.arange(len(batch_sizes)),
        bps,
        bottom=fps,
        edgecolor='#5A9BD5',
        color='#5A9BD5',
        width=bar_width,
        label=gpu + ', BPPSA',
    )

    plt.xticks(
        np.arange(len(batch_sizes)),
        ['1/{}'.format(b) for b in batch_sizes],
        fontsize=13,
    )
    plt.axhline(y=1.0, color='r', linestyle='--', linewidth=3, label='Baseline')
    plt.xlabel('Fraction of GPU per Sample')
    plt.ylabel('Normalized Runtime Breakdown')
    plt.legend()
    #plt.show()
    plt.savefig('fig_8_f.png', bbox_inches='tight', pad_inches=0.0)
    plt.clf()


def plot_seq_lens_to_speedups(gpu):
    seq_lens = [10, 30, 100, 300, 1000, 3000, 10000, 30000]
    speedups = []
    fps = []
    bps = []
    bar_width = 0.25
    for seq_len in seq_lens:
        normal_latencies = read_csv(gpu, 'normal', seq_len, 16)['latency']
        blelloch_latencies = read_csv(gpu, 'blelloch', seq_len, 16)['latency']
        normal_nobp_latencies = read_csv(gpu, 'normal-nobp', seq_len,
                                         16)['latency']
        baseline = normal_latencies.mean()

        fp = normal_nobp_latencies.mean() / baseline
        fps.append(fp)

        bp = ((blelloch_latencies.mean() - normal_nobp_latencies.mean()) /
              baseline)
        bps.append(bp)

        mean_speedup = normal_latencies.mean() / blelloch_latencies.mean()
        speedups.append(mean_speedup)

    print('{}: {}'.format(gpu, np.max(speedups)))
    plt.bar(
        np.arange(len(seq_lens)),
        fps,
        edgecolor='#5A9BD5',
        color='white',
        hatch='////',
        alpha=0.99,
        width=bar_width,
        label=gpu + ', FP',
    )
    plt.bar(
        np.arange(len(seq_lens)),
        bps,
        bottom=fps,
        edgecolor='#5A9BD5',
        color='#5A9BD5',
        width=bar_width,
        label=gpu + ', BPPSA',
    )

    plt.xticks(np.arange(len(seq_lens)),
               ['10', '30', '100', '300', '1k', '3k', '10k', '30k'])
    plt.axhline(y=1.0, color='r', linestyle='--', linewidth=3, label='Baseline')
    plt.xlabel('Sequence Length')
    plt.ylabel('Normalized Runtime Breakdown')
    plt.legend()
    #plt.show()
    plt.savefig('fig_8_c.png', bbox_inches='tight', pad_inches=0.0)
    plt.clf()


def read_training_csv(gpu, mode):
    return pd.read_csv(
        './rnn-epoch-latency-{}/training-curve-{}-bernoulli1000_10-batch_size_16.csv'
        .format(gpu, mode),
        index_col=0)


def plot_training_curve(gpu):
    normal = read_training_csv(gpu, 'normal')
    blelloch = read_training_csv(gpu, 'blelloch')
    print(normal['timestamp'])
    print(normal['loss'])
    plt.plot(normal['timestamp'],
             normal['loss'],
             color='r',
             label='PyTorch/cuDNN',
             linewidth=5)
    plt.plot(blelloch['timestamp'],
             blelloch['loss'],
             color='#5A9BD5',
             label='BPPSA',
             linewidth=5)
    plt.xlabel('Wall-clock Time (s)')
    plt.ylabel('Training Loss')
    plt.legend()
    plt.savefig('fig_7.png', bbox_inches='tight', pad_inches=0.0)
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
            label='Speedup')
    plt.xticks(np.arange(len(batch_sizes)), batch_sizes)
    plt.axhline(y=1.0, color='r', linestyle='--', linewidth=3, label='Baseline')
    plt.xlabel('Mini-batch Size')
    plt.ylabel('Speedup over Baseline (PyTorch/cuDNN)')
    plt.legend()
    plt.yscale('log')
    plt.savefig('fig_8_e.png', bbox_inches='tight', pad_inches=0.0)
    plt.clf()


def plot_seq_lens_to_backward_speedups(gpu):
    seq_lens = [10, 30, 100, 300, 1000, 3000, 10000, 30000]
    speedups = []
    for seq_len in seq_lens:
        normal_latencies = read_csv(gpu, 'normal', seq_len, 16)['latency']
        blelloch_latencies = read_csv(gpu, 'blelloch', seq_len, 16)['latency']
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
            label='Speedup')
    plt.xticks(np.arange(len(seq_lens)),
               ['10', '30', '100', '300', '1k', '3k', '10k', '30k'])
    plt.axhline(y=1.0, color='r', linestyle='--', linewidth=3, label='Baseline')
    plt.xlabel('Sequence Length')
    plt.ylabel('Speedup over Baseline (PyTorch/cuDNN)')
    plt.legend()
    plt.yscale('log')
    plt.savefig('fig_8_b.png', bbox_inches='tight', pad_inches=0.0)
    plt.clf()


def plot_seq_lens_to_backward_runtimes(gpu):
    seq_lens = [10, 30, 100, 300, 1000, 3000, 10000, 30000]
    runtimes_mean = []
    runtimes_std = []
    bar_width = 0.25
    for seq_len in seq_lens:
        num_iterations = 32000 / 16
        blelloch_latencies = read_csv(
            gpu,
            'blelloch',
            seq_len,
            16,
        )['latency'] / num_iterations * 1000
        normal_nobp_latencies = read_csv(
            gpu,
            'normal-nobp',
            seq_len,
            16,
        )['latency'] / num_iterations * 1000
        runtimes_mean.append(
            (blelloch_latencies - normal_nobp_latencies).mean())
        runtimes_std.append(
            np.sqrt(blelloch_latencies.var() + normal_nobp_latencies.var()))

    plt.bar(
        np.arange(len(seq_lens)),
        runtimes_mean,
        yerr=runtimes_std,
        color='#5A9BD5',
        width=bar_width,
        label=gpu,
    )

    plt.xticks(np.arange(len(seq_lens)),
               ['10', '30', '100', '300', '1k', '3k', '10k', '30k'])
    plt.xlabel('Sequence Length')
    plt.ylabel('BPPSA Runtime (ms) per Iteration')
    plt.legend()
    #plt.yscale('log')
    #plt.show()
    plt.savefig('fig_8_a.png', bbox_inches='tight', pad_inches=0.0)
    plt.clf()


def plot_batch_sizes_to_backward_runtimes(gpu):
    batch_sizes = list(reversed([2, 4, 8, 16, 32, 64, 128, 256]))
    runtimes_mean = []
    runtimes_std = []
    bar_width = 0.25
    for batch_size in batch_sizes:
        num_iterations = 32000 / batch_size
        blelloch_latencies = read_csv(
            gpu,
            'blelloch',
            1000,
            batch_size,
        )['latency'] / num_iterations * 1000
        normal_nobp_latencies = read_csv(
            gpu,
            'normal-nobp',
            1000,
            batch_size,
        )['latency'] / num_iterations * 1000
        runtimes_mean.append(
            (blelloch_latencies - normal_nobp_latencies).mean())
        runtimes_std.append(
            np.sqrt(blelloch_latencies.var() + normal_nobp_latencies.var()))

    plt.bar(
        np.arange(len(batch_sizes)),
        runtimes_mean,
        yerr=runtimes_std,
        color='#5A9BD5',
        width=bar_width,
        label=gpu,
    )

    plt.xticks(
        np.arange(len(batch_sizes)),
        ['1/{}'.format(b) for b in batch_sizes],
        fontsize=13,
    )
    plt.xlabel('Fraction of GPU per Sample')
    plt.ylabel('BPPSA Runtime (ms) per Iteration')
    plt.legend()
    #plt.yscale('log')
    #plt.show()
    plt.savefig('fig_8_d.png', bbox_inches='tight', pad_inches=0.0)
    plt.clf()


def main(args):
    plt.rcParams.update({'font.size': 15})
    plot_batch_sizes_to_speedups(args.gpu)
    plot_seq_lens_to_speedups(args.gpu)
    plt.rcParams.update({'font.size': 23})
    plot_training_curve(args.gpu)
    plt.rcParams.update({'font.size': 15})
    plot_batch_sizes_to_backward_speedups(args.gpu)
    plot_seq_lens_to_backward_speedups(args.gpu)
    plot_seq_lens_to_backward_runtimes(args.gpu)
    plot_batch_sizes_to_backward_runtimes(args.gpu)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, required=True)
    main(parser.parse_args())
