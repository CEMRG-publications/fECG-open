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

A sample 10-segment recording is provided in `data/inputs/test_library` (`test.ch1` – `test.ch4`) to test the library installation. 
A 92-segment recording is also provided in `data/inputs/` (`recording.ch1` – `recording.ch4`).

Output: one `.mat` file per recording (MATLAB v7.3 / HDF5) containing per-segment waveforms and beat annotations.

### MATLAB

```matlab
addpath(genpath('/path/to/fECG-open/matlab'))
```

Set `input_folder = 'data/inputs/test_library'` and `output_folder = 'data/outputs/test_library'` in `pipeline.m`,
then run:

```matlab
pipeline
```

### Python

```bash
pip install ".[plot,test]"
python pipeline.py -i data/inputs/test_library -o data/outputs/test_library
```

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
