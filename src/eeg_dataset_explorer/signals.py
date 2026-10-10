"""Shared signal and sampling contracts. Arrays use (channels, samples), in volts."""

from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike, NDArray


def signal_array(data: ArrayLike, minimum_samples: int = 2) -> NDArray[np.float64]:
    """Validate a real, finite, two-dimensional EEG array."""
    if np.iscomplexobj(data):
        raise ValueError("EEG input must be real-valued")
    array = np.asarray(data, dtype=float)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] < minimum_samples:
        raise ValueError("Expected nonempty (channels, samples) array")
    if not np.isfinite(array).all():
        raise ValueError("Signal contains NaN or infinity")
    return array


def sampling_rate(fs: float) -> float:
    """Require a finite positive sample rate in Hz."""
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("Sampling rate must be finite and positive")
    return float(fs)


def synthetic(
    fs: float = 256.0, seconds: float = 8.0, channels: int = 4, seed: int = 7
) -> NDArray[np.float64]:
    """Deterministic synthetic EEG-like signals, explicitly not patient data."""
    sampling_rate(fs)
    if not np.isfinite(seconds) or seconds <= 0 or channels < 1:
        raise ValueError("Positive duration and channel count required")
    time = np.arange(round(fs * seconds)) / fs
    if len(time) < 2:
        raise ValueError("At least two samples required")
    rng = np.random.default_rng(seed)
    return np.array(
        [
            20e-6 * np.sin(2 * np.pi * (8 + i) * time + 0.2 * i) + rng.normal(0, 2e-6, len(time))
            for i in range(channels)
        ]
    )
