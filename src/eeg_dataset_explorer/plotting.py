"""Optional scientific visualization; importing the package does not require plotting."""

from __future__ import annotations
import numpy as np
from .signals import signal_array, sampling_rate


def plot_signals(data: np.ndarray, fs: float, labels: list[str] | None = None):
    """Return a labeled stacked EEG figure in microvolts without saving a result."""
    import matplotlib.pyplot as plt

    x = signal_array(data)
    sampling_rate(fs)
    if labels is not None and len(labels) != len(x):
        raise ValueError("One label per channel required")
    fig, axes = plt.subplots(len(x), 1, sharex=True, squeeze=False, figsize=(10, 1.8 * len(x)))
    t = np.arange(x.shape[1]) / fs
    for i, ax in enumerate(axes[:, 0]):
        ax.plot(t, x[i] * 1e6, linewidth=0.7)
        ax.set_ylabel((labels[i] if labels else f"Ch {i + 1}") + " (uV)")
    axes[-1, 0].set_xlabel("Time (s)")
    fig.tight_layout()
    return fig
