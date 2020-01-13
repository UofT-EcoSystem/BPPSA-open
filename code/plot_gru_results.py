import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_csv(gpu, mode, seq_len, batch_size):
    return pd.read_csv(
        './gru-epoch-latency-{}/{}-IRMASmfcc_{}-batch_size_{}.csv'
        .format(gpu, mode, seq_len, batch_size),
        index_col=0)


def plot_runtime_breakdown(gpu):
    batch_sizes = [16, 32, 64]
    seq_lens = ['s', 'm', 'l']
    gpus = [gpu]
    fig, axs = plt.subplots(
        3,
        3,
        sharex=True,
        sharey=True,
        figsize=(7, 7)
    )
    for ax, batch_size in zip(axs[0], batch_sizes):
        ax.set_title('Batch Size = {}'.format(batch_size), size=15, pad=10)
    for ax, seq_len in zip(axs[:, 0], seq_lens):
        ax.set_ylabel(seq_len.upper(), rotation=0, size=15, labelpad=10)

    max_overall_speedup = 0
    max_bp_speedup = 0
    for i, seq_len in enumerate(seq_lens):
        for j, batch_size in enumerate(batch_sizes):
            axs[i][j].set_xlim((-1, 1))
            axs[i][j].set_ylim((0, 1))
            axs[i][j].set_xticks(range(len(gpus)))
            axs[i][j].set_xticklabels(gpus)
            for k, gpu in enumerate(gpus):
                normal_latencies = read_csv(gpu, 'normal', seq_len,
                                            batch_size)['latency']
                blelloch_latencies = read_csv(gpu, 'blelloch', seq_len,
                                              batch_size)['latency']
                normal_nobp_latencies = read_csv(gpu, 'normal-nobp', seq_len,
                                                 batch_size)['latency']
                blelloch_nobp_latencies = read_csv(gpu, 'blelloch-nobp',
                                                   seq_len,
                                                   batch_size)['latency']
                baseline = normal_latencies.mean()
                fp = (normal_nobp_latencies.mean() / baseline)
                oh = ((blelloch_nobp_latencies.mean() -
                       normal_nobp_latencies.mean()) / baseline)
                bp = ((blelloch_latencies.mean() -
                       blelloch_nobp_latencies.mean()) / baseline)
                max_overall_speedup = max(baseline / blelloch_latencies.mean(),
                                          max_overall_speedup)
                max_bp_speedup = max(
                    (normal_latencies.mean() - normal_nobp_latencies.mean()) /
                    (blelloch_latencies.mean() -
                     blelloch_nobp_latencies.mean()), max_bp_speedup)

                fp_h = axs[i][j].bar(
                    k,
                    fp,
                    width=0.2,
                    color='orange',
                )
                bp_h = axs[i][j].bar(
                    k,
                    bp,
                    bottom=fp,
                    width=0.2,
                    color='royalblue',
                )
                oh_h = axs[i][j].bar(
                    k,
                    oh,
                    bottom=fp + bp,
                    width=0.2,
                    color='limegreen',
                )
                bl_h = axs[i][j].bar(
                    k + 0.2,
                    (1 - fp),
                    bottom=fp,
                    width=0.06,
                    color='red',
                )
                axs[i][j].bar(
                    k + 0.2,
                    fp,
                    width=0.06,
                    color='orange',
                )

    print(max_overall_speedup, max_bp_speedup)
    box = axs[0][0].get_position()
    lgd = fig.legend(
        [fp_h, bp_h, oh_h, bl_h],
        labels=['FP', 'BPPSA', 'FO', 'BP (PyTorch/cuDNN)'],
        loc='upper left',
        bbox_to_anchor=(box.x0 - 0.03, box.y1, 0.1, 0.17),
        ncol=4,
    )
    fig.tight_layout()
    plt.savefig(
        'fig_12.png',
        bbox_extra_artists=(lgd,),
        bbox_inches='tight',
        dpi=300,
    )


def read_training_csv(gpu, mode, seq_len):
    return pd.read_csv(
        './gru-epoch-latency-{}/training-curve-{}-IRMASmfcc_{}-batch_size_16.csv'.format(
            gpu, mode, seq_len),
        index_col=0)


def plot_training_curve(gpu):
    seq_lens = ['s', 'm', 'l']
    fig, axs = plt.subplots(
        1,
        3,
        figsize=(10, 3.5)
    )
    print(axs.shape)
    for ax, seq_len in zip(axs, seq_lens):
        ax.set_title(seq_len.upper(), size=15, pad=10)
        ax.set_xlabel('Wall-clock Time (s)')
    axs[0].set_ylabel('Training Loss')
    for i, seq_len in enumerate(seq_lens):
        bppsa_losses = read_training_csv(gpu, 'blelloch', seq_len)['loss']
        bppsa_timestamps = read_csv(gpu, 'blelloch', seq_len,
                                    '16')['timestamp']

        normal_losses = read_training_csv(gpu, 'normal', seq_len)['loss']
        normal_timestamps = read_csv(gpu, 'normal', seq_len,
                                     '16')['timestamp']

        normal = axs[i].plot(normal_timestamps,
                             normal_losses,
                             color='r',
                             label='PyTorch/cuDNN')
        bppsa = axs[i].plot(bppsa_timestamps,
                            bppsa_losses,
                            color='#5A9BD5',
                            label='BPPSA')

    box = axs[0].get_position()
    print(box)
    lgd = fig.legend(
        [normal, bppsa],
        labels=['PyTorch/cuDNN', 'BPPSA'],
        loc='upper left',
        bbox_to_anchor=(box.x0 - 0.06, box.y1, 0.1, 0.17),
        ncol=2,
    )
    fig.tight_layout()
    plt.savefig(
        'fig_11.png',
        bbox_extra_artists=(lgd,),
        bbox_inches='tight',
        dpi=300,
    )

def main(args):
    import matplotlib
    plt.rcParams.update({'font.size': 14})
    plot_runtime_breakdown(args.gpu)
    plot_training_curve(args.gpu)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, required=True)
    main(parser.parse_args())
