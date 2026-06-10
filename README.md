# fECG-open

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![Tests](https://github.com/CEMRG-publications/fECG-open/actions/workflows/python-tests.yml/badge.svg)](https://github.com/CEMRG-publications/fECG-open/actions/workflows/python-tests.yml)

> Open-source MATLAB and Python library for extracting maternal and fetal ECG waveforms and R-peak annotations from four-channel abdominal ECG recordings.

## Overview

Abdominal ECG electrodes pick up a mixture of two cardiac signals: the maternal ECG (strong) and the fetal ECG (weak). Separating them is a core problem in non-invasive fetal cardiac monitoring.

fECG-open implements the SAVER pipeline (Synchrosqueezing-based Adaptive VEctorcardiographic ECG Representation), which uses time-frequency ridge-following via the de-shape synchrosqueezing transform, SVD-based optimal shrinkage, and dynamic-programming beat tracking to robustly separate the two sources from four abdominal channels.

The library is available in two implementations:

| | MATLAB | Python |
|---|---|---|
| Entry point | `matlab/pipeline.m` | `python/pipeline.py` |
| Tested on | R2024b | Python 3.12 |
| Parallel pipeline | - | `python/pipeline_fast.py` |
| Unit tests | - | 58 pytest tests |

The MATLAB library is the original implementation. The Python library is a 1:1 translation, validated to produce bit-identical outputs.

## Repository Structure

```
fECG-open/
├── matlab/     MATLAB pipeline and signal processing library
├── python/     Python translation of the MATLAB pipeline
└── scripts/    Utility scripts (e.g. Monica DK raw file conversion)
```

Full details on requirements, installation, usage, and API are provided in each subfolder:

- [MATLAB documentation](matlab/README.md)
- [Python documentation](python/README.md)

## Quick Start

A sample recording is provided in `python/tests/test_recording` (`test.ch1` – `test.ch4`) to test the library installation.
A 92-segment recording is provided in `data/inputs/` (`recording.ch1` – `recording.ch4`).

Output: one `.mat` file per recording (MATLAB v7.3 / HDF5) containing per-segment waveforms and beat annotations.

### MATLAB

```matlab
addpath(genpath('/path/to/fECG-open/matlab'))
```

Set `input_folder = 'data/inputs'` and `output_folder = 'data/outputs'` in `pipeline.m`,
then run:

```matlab
pipeline
```

### Python

```bash
pip install ".[plot,test]"
python pipeline.py -i data/inputs -o data/outputs
```

## Algorithmic Parameters

The following parameters are fixed in the current implementation. They can be
modified directly in `pipeline.m` (MATLAB) or `pipeline.py` / `pipeline_fast.py`
(Python) for custom configurations.

| Parameter | Value | Unit |
|-----------|:-----:|:----:|
| **Recording and resampling** | | |
| Number of channels | 4 | — |
| Target sampling rate | 1000 | Hz |
| **Signal quality index (bSQI)** | | |
| bSQI window length | 4 | s |
| bSQI evaluation interval | 1 | s |
| Beat-matching tolerance for F1-based bSQI | 0.150 | s |
| bSQI acceptance threshold | 0.8 | — |
| **jQRS detector** | | |
| jQRS energy threshold | 0.3 | — |
| jQRS refractory period | 0.25 | s |
| jQRS integration window size | 7 | samples |
| jQRS segment window size | 15 | s |
| **STFT / cepstral / SST** | | |
| STFT Flattop window length (maternal) | 1000 | samples at 100 Hz |
| STFT Flattop window length (fetal) | 600 | samples at 100 Hz |
| STFT Flattop hop size | 20 | samples |
| Internal signal rate fed into STFT | 100 | Hz |
| STFT frequency resolution | 0.02 | cycles/sample |
| SST coefficient magnitude threshold | 1e-6 | — |
| Upper normalised frequency bound | 0.10 | — |
| Lower normalised frequency bound | 0.005 | — |
| Cepstral power exponent | 0.3 | — |
| Number of SST sub-harmonics | 1 | — |
| **Dynamic programming** | | |
| DP ridge-extraction smoothness penalty | 50 (5 in refinement pass) | — |
| DP beat-tracking smoothness penalty | 50 (5 in refinement pass) | — |
| Half-width of α grid | 7 | — |
| **Baseline removal** | | |
| No-morphology median filter length | 51 | samples |
| Long-window median filter length | 301 | samples |
| Morphology-preserving median filter length | 101 | samples |
| **Non-local median** | | |
| Non-local median pool size | 10 | — |

## Output Format

Each recording produces a `.mat` file (MATLAB v7.3 / HDF5) containing one entry per
60-second segment for each of the following variables:

| Variable | Description |
|---|---|
| `final_file_names` | Source recording file name for each segment |
| `final_start_times` | Start time of the segment (seconds) |
| `final_og_ECG` | Original interpolated ECG (no preprocessing) |
| `final_SQI` | Best bSQI achieved |
| `final_chs_used` | Channel pair used to achieve the above bSQI |
| `final_Om` | Extracted maternal ECG signal |
| `final_Of` | Extracted fetal ECG signal |
| `final_aECG` | Decomposed combined ECG signal |
| `final_mbeats` | Indices of maternal R-peaks |
| `final_fbeats` | Indices of fetal R-peaks |
| `final_proc_time` | Cumulative processing time (seconds) |

Files are readable by `MATLAB load()`, `h5py.File()`, and `hdf5storage.loadmat()`.

## Testing

The Python library includes an automated test suite (58 tests) covering core signal processing modules and the full extraction pipeline:

```bash
pytest -v
```

## Citation

If you use this library in your research, please cite: [to be added once the paper has been accepted]

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

For third-party components included in this library, see the Acknowledgements sections in [matlab/README.md](matlab/README.md).
