"""Read-only EDF inventory. No signal loading, copying, or model training."""

from __future__ import annotations

from pathlib import Path


def read_header(path: Path) -> dict:
    """Read EDF structural fields without exposing patient/header identifiers."""
    with path.open("rb") as stream:
        header = stream.read(256)
        if len(header) != 256:
            raise ValueError("Truncated EDF header")
        if header[:8].strip() != b"0":
            raise ValueError("Expected ordinary 16-bit EDF version 0")
        size = int(header[184:192])
        records = int(header[236:244])
        duration = float(header[244:252])
        channels = int(header[252:256])
        if channels <= 0 or duration <= 0 or records < -1 or size < 256 + channels * 256:
            raise ValueError("Invalid EDF dimensions")
        signal_header = stream.read(channels * 256)
        if len(signal_header) != channels * 256:
            raise ValueError("Truncated signal header")

    def fields(offset: int, width: int) -> list[str]:
        return [
            signal_header[offset + i * width : offset + (i + 1) * width].decode("ascii").strip()
            for i in range(channels)
        ]

    labels = fields(0, 16)
    samples = [int(v) for v in fields(channels * 216, 8)]
    if any(value <= 0 for value in samples):
        raise ValueError("Each EDF signal needs a positive sample count per record")
    expected = None if records < 0 else size + 2 * records * sum(samples)
    return {
        "channels": channels,
        "labels": labels,
        "units": fields(channels * 96, 8),
        "sampling_rates_hz": [v / duration for v in samples],
        "duration_seconds": None if records < 0 else records * duration,
        "file_bytes": path.stat().st_size,
        "expected_bytes": expected,
        "size_matches_header": None if expected is None else path.stat().st_size == expected,
    }


def audit(root: Path) -> dict:
    if not root.is_dir():
        raise ValueError(f"Dataset directory does not exist: {root}")
    report = {}
    for dataset in sorted(p for p in root.iterdir() if p.is_dir()):
        paths = sorted(dataset.rglob("*.edf"))
        if not paths:
            continue
        entries = []
        for path in paths:
            entry = {"file": path.relative_to(dataset).as_posix()}
            try:
                entry.update(read_header(path))
            except (OSError, ValueError, UnicodeError) as error:
                entry["error"] = str(error)
            entries.append(entry)
        report[dataset.name] = {
            "edf_count": len(paths),
            "recording_directories": sorted({p.parent.name for p in paths}),
            "annotations": [
                p.relative_to(dataset).as_posix()
                for p in sorted(dataset.rglob("*.txt"))
                if "summary" in p.name.lower() or "seizures-list" in p.name.lower()
            ],
            "headers": entries,
        }
    return report
