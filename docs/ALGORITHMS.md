# EDF metadata and bounded reads

`headers.read_header(path)` reads fixed EDF header fields, labels, units,
per-channel sample rates and declared byte length using the standard library.
`io.read_segment(path, start, duration, channels)` loads only the requested interval
through MNE and returns `(channels, samples)` in volts plus sample-index metadata.
`python -m eeg_dataset_explorer file.edf --plot` inspects a file and plots four channels.
Patient header fields are not included in manifests.

Sample indices are rounded from seconds. Mixed-rate EDF signals can be resampled
by MNE to a common rate: inspect original per-channel rates before selecting them.
The header checker assumes ordinary 16-bit EDF, not 24-bit BDF. Structural validity
is not proof of checksum integrity or clinical annotation accuracy.

The reader does not infer seizure labels. Pair metadata with the dedicated
annotation repository. Plotting is optional and amplitudes are displayed in uV.

## References

- [Method or implementation reference](https://mne.tools/stable/generated/mne.io.read_raw_edf.html)
- [CHB-MIT 1.0.0](https://physionet.org/content/chbmit/1.0.0/)
- [Siena 1.0.0](https://physionet.org/content/siena-scalp-eeg/1.0.0/)
