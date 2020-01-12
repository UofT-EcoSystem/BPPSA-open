import argparse
import functools
import io
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io.wavfile as wavfile
import sklearn
import sys
import torch
import torch.utils.data
from multiprocessing import Pool


def to_mfcc(sig, rate, n_mfcc=13, n_fft=2048, hop_length=512):
    mfcc = librosa.feature.mfcc(
        y=sig,
        sr=rate,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    mfcc = mfcc[1:, :]  # Remove the first coeff.
    mfcc = sklearn.preprocessing.scale(mfcc, axis=1)
    return mfcc.astype(np.float32)


def process_file(wav_file_name, instrument_dir, n_mfcc, n_fft, hop_length):
    rate, data = wavfile.read(os.path.join(instrument_dir, wav_file_name))
    data = data.astype(np.float)

    return np.transpose(
        np.concatenate(
            (to_mfcc(data[:, 0],
                     rate,
                     n_mfcc=n_mfcc,
                     n_fft=n_fft,
                     hop_length=hop_length),
             to_mfcc(data[:, 1],
                     rate,
                     n_mfcc=n_mfcc,
                     n_fft=n_fft,
                     hop_length=hop_length)),
            axis=0,
        ))


def main(args):
    assert os.path.isdir(args.data_dir)
    instrument_encoding = {
        "cel": 0,
        "cla": 1,
        "flu": 2,
        "gac": 3,
        "gel": 4,
        "org": 5,
        "pia": 6,
        "sax": 7,
        "tru": 8,
        "vio": 9,
        "voi": 10,
    }
    frames_to_hop = {
        's': 512,
        'm': 256,
        'l': 128,
    }
    frames_to_nfft = {
        's': 4096,
        'm': 2048,
        'l': 1024,
    }
    frames_to_nmfcc = {
        's': 20,
        'm': 13,
        'l': 7,
    }
    X = []
    Y = []
    prev_rate = None
    for instrument in instrument_encoding.keys():
        instrument_dir = os.path.join(args.data_dir, instrument)

        with Pool(10) as p:
            this_X = p.map(
                functools.partial(
                    process_file,
                    instrument_dir=instrument_dir,
                    n_mfcc=frames_to_nmfcc[args.frames],
                    n_fft=frames_to_nfft[args.frames],
                    hop_length=frames_to_hop[args.frames],
                ),
                os.listdir(instrument_dir),
            )
            Y.extend([instrument_encoding[instrument]] * len(this_X))
            X.extend(this_X)

    X, Y = np.stack(X), np.array(Y)
    print(X.shape, Y.shape)
    X, Y = torch.from_numpy(X), torch.from_numpy(Y)
    dataset = torch.utils.data.TensorDataset(X, Y)
    #trainset, testset = torch.utils.data.random_split(dataset, [6704, 1])
    trainset = dataset
    torch.save(trainset[:][0], os.path.join(args.save_dir, 'train_X'))
    torch.save(trainset[:][1], os.path.join(args.save_dir, 'train_Y'))
    #torch.save(testset[:][0], os.path.join(args.save_dir, 'test_X'))
    #torch.save(testset[:][1], os.path.join(args.save_dir, 'test_Y'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='The path to the IRMAS dataset.',
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        required=False,
        default='./IRMASmfcc_512/',
        help='Where to save the parsed data.',
    )
    parser.add_argument(
        '--frames',
        type=str,
        default='m',
        choices=['s', 'm', 'l'],
    )
    main(parser.parse_args())
