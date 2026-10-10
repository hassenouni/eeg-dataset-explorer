"""Bounded EDF reads using MNE; physical amplitudes are returned in volts."""

from __future__ import annotations
from pathlib import Path
import numpy as np
from .signals import sampling_rate


def read_segment(
    path: str | Path, start: float = 0.0, duration: float = 8.0, channels: list[str] | None = None
) -> tuple[np.ndarray, dict]:
    """Read [start, start+duration) without loading the complete EDF."""
    import mne

    if not np.isfinite(start) or not np.isfinite(duration) or start < 0 or duration <= 0:
        raise ValueError("Require finite start >= 0 and duration > 0")
    with mne.io.read_raw_edf(str(path), preload=False, verbose="ERROR") as raw:
        fs = sampling_rate(raw.info["sfreq"])
        first = int(round(start * fs))
        last = int(round((start + duration) * fs))
        if first < 0 or last > raw.n_times or last <= first:
            raise ValueError("Requested segment is outside the recording")
        picks = raw.ch_names if channels is None else channels
        if not picks or len(set(picks)) != len(picks) or any(c not in raw.ch_names for c in picks):
            raise ValueError("Select existing, unique channels")
        data = raw.get_data(picks=picks, start=first, stop=last)
        return data, {
            "recording": Path(path).name,
            "channels": list(picks),
            "fs": fs,
            "start_sample": first,
            "stop_sample": last,
            "units": "V",
        }


def export_sample(
    path: str | Path,
    destination: str | Path,
    start: float = 0.0,
    duration: float = 8.0,
    channels: list[str] | None = None,
    dataset: str = "user-supplied",
) -> dict:
    """Write a local calibrated, unfiltered excerpt plus a source checksum manifest.

    No patient header fields or absolute paths are exported. Data licensing remains
    separate from the code license. Caller must review redistribution terms.
    """
    import hashlib
    import json

    data, metadata = read_segment(path, start, duration, channels)
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    metadata.update(
        {
            "dataset": dataset,
            "source_sha256": digest.hexdigest(),
            "processing": "EDF physical calibration to volts only; no filtering or normalization",
            "labels": "not supplied; load source annotations before seizure labeling",
        }
    )
    target = Path(destination)
    if target.suffix != ".npz":
        raise ValueError("Destination must end with .npz")
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, data=data, fs=metadata["fs"], channels=np.array(metadata["channels"]))
    metadata["excerpt_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    target.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf8")
    return metadata
