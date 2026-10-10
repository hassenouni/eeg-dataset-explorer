"""Usage: python -m eeg_dataset_explorer FILE.edf [--seconds 8] [--plot]."""

import argparse
import json
from pathlib import Path
from .headers import read_header
from .io import read_segment, export_sample
from .plotting import plot_signals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("edf", type=Path)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--channels", nargs="+")
    parser.add_argument("--export-npz", type=Path)
    args = parser.parse_args()
    print(json.dumps(read_header(args.edf), indent=2))
    if args.export_npz:
        export_sample(args.edf, args.export_npz, args.start, args.seconds, args.channels)
        print("Local excerpt exported with provenance.")
    if args.plot:
        import matplotlib.pyplot as plt

        data, meta = read_segment(args.edf, start=args.start, duration=args.seconds, channels=args.channels)
        plot_signals(data[:4], meta["fs"], meta["channels"][:4])
        plt.show()


if __name__ == "__main__":
    main()
