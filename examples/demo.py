"""Synthetic or local original-EEG algorithm demonstration."""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from eeg_dataset_explorer.signals import synthetic, signal_array, sampling_rate
from eeg_dataset_explorer.plotting import plot_signals


def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, help="Local NPZ with data and fs arrays")
    parser.add_argument("--figure", type=Path, help="Optional source signal plot (dataset explorer only)")
    args = parser.parse_args()
    if args.sample:
        with np.load(args.sample, allow_pickle=False) as sample:
            data = signal_array(sample["data"])
            fs = sampling_rate(float(sample["fs"]))
        source = "Original local EEG excerpt: " + args.sample.name
    else:
        fs = 256.0
        data = synthetic(fs)
        source = "Synthetic signal: not patient data"
    print(source)

    print("Signal dimensions:", data.shape, "sample rate:", fs, "Hz")
    if args.figure:
        figure = plot_signals(data, fs)
        figure.suptitle(source)
        figure.savefig(args.figure, dpi=140)


if __name__ == "__main__":
    main()
