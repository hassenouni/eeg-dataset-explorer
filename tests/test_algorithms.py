from pathlib import Path
import numpy as np
import pytest
from eeg_dataset_explorer.headers import read_header
from eeg_dataset_explorer.io import read_segment
from eeg_dataset_explorer.signals import signal_array, synthetic


def make_edf(path: Path) -> None:
    def field(value, width):
        return str(value).ljust(width).encode("ascii")

    header = b"".join(
        [
            field("0", 8),
            field("synthetic", 80),
            field("test", 80),
            field("01.01.00", 8),
            field("00.00.00", 8),
            field(768, 8),
            field("", 44),
            field(1, 8),
            field(1, 8),
            field(2, 4),
        ]
    )
    blocks = [
        (["A", "B"], 16),
        (["", ""], 80),
        (["uV", "uV"], 8),
        ([-100, -100], 8),
        ([100, 100], 8),
        ([-32768, -32768], 8),
        ([32767, 32767], 8),
        (["", ""], 80),
        ([16, 16], 8),
        (["", ""], 32),
    ]
    signals = b"".join(field(value, width) for values, width in blocks for value in values)
    samples = np.tile(np.arange(16, dtype="<i2"), 2).tobytes()
    path.write_bytes(header + signals + samples)


def test_edf_header_and_physical_read(tmp_path):
    path = tmp_path / "synthetic.edf"
    make_edf(path)
    meta = read_header(path)
    assert meta["labels"] == ["A", "B"]
    assert meta["sampling_rates_hz"] == [16, 16]
    assert meta["size_matches_header"] is True
    x, info = read_segment(path, 0.25, 0.5, ["B"])
    assert x.shape == (1, 8)
    assert info["start_sample"] == 4 and info["units"] == "V"
    expected = ((np.arange(4, 12) + 32768) * (200 / 65535) - 100) * 1e-6
    np.testing.assert_allclose(x[0], expected, atol=1e-15)
    with pytest.raises(ValueError):
        read_segment(path, 0, 2)


def test_truncated_header(tmp_path):
    path = tmp_path / "broken.edf"
    path.write_bytes(b"0")
    with pytest.raises(ValueError, match="Truncated"):
        read_header(path)


@pytest.mark.parametrize(
    "value", [np.zeros(4), np.array([[np.nan, 1]]), np.ones((0, 4)), np.ones((2, 2)) * 1j]
)
def test_invalid_signal(value):
    with pytest.raises(ValueError):
        signal_array(value)


def test_synthetic_reproducibility():
    np.testing.assert_array_equal(synthetic(), synthetic())


def test_exported_excerpt_manifest(tmp_path):
    import hashlib
    import json
    from eeg_dataset_explorer.io import export_sample

    source = tmp_path / "synthetic.edf"
    make_edf(source)
    destination = tmp_path / "excerpt.npz"
    meta = export_sample(source, destination, duration=0.5, channels=["A"])
    assert meta["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = json.loads(destination.with_suffix(".json").read_text())
    assert str(tmp_path) not in json.dumps(manifest)
    with np.load(destination, allow_pickle=False) as sample:
        assert sample["data"].shape == (1, 8)
        assert sample["fs"] == 16


def test_non_edf_rejected(tmp_path):
    source = tmp_path / "wrong.edf"
    make_edf(source)
    data = source.read_bytes()
    source.write_bytes(b"1" + data[1:])
    with pytest.raises(ValueError, match="version"):
        read_header(source)
