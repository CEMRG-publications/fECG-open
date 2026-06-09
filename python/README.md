# fECG-open — Python

> Python translation of the fECG-open MATLAB pipeline: separates maternal and fetal ECG waveforms and R-peak annotations from four-channel abdominal ECG recordings.

## Background

Abdominal ECG electrodes pick up a mixture of two cardiac signals: the maternal ECG (strong) and the fetal ECG (weak). Separating them is a core problem in fetal cardiac monitoring. This library is a 1:1 Python translation of the MATLAB pipeline. It uses the de-shape synchrosqueezing transform for heart-rate tracking, SVD-based optimal shrinkage for template separation, and dynamic-programming beat tracking.

## Requirements

### Python

- **Tested on:** Python 3.12.7
- **Minimum version:** Python 3.7 (f-strings, `typing.Union`, `math.gcd` with 2 args)
- **Dependencies:**

| Package | Version floor | Notes |
|---|---|---|
| `numpy` | `>=1.20` | Core numeric library |
| `scipy` | `>=1.7` | Signal processing, interpolation, integration |
| `numba` | `>=0.53` | JIT compilation for DP inner loops (`beat_simple`, `CurveExt_M`) |
| `h5py` | `>=3.0` | HDF5 I/O (resume logic, output verification) |
| `hdf5storage` | `>=0.1.18` | MATLAB v7.3-compatible MAT file I/O |
| `matplotlib` | *(optional)* | Used only in `pan_tompkin_revised.py` when `gr=1`; never called in the main pipeline |

`pipeline_fast.py` additionally uses `concurrent.futures` and `multiprocessing` (both standard library).

## Installation

```bash
git clone <repo-url>
cd fECG_open/python
pip install -r requirements.txt
```

Or install as a package from the repo root:

```bash
pip install .
```

## Quick Start

### Sequential pipeline

```bash
python pipeline.py -i /path/to/input_folder -o /path/to/output_folder
```

`input_folder` must contain four binary channel files per recording, named `<basename>.ch1` through `<basename>.ch4` (Monica DK format, little-endian int32). Channels 1–3 are expected at 300 Hz; channel 4 at 900 Hz. All are resampled to 1000 Hz internally.

Minimal synthetic example:

```python
import numpy as np, os

# Write a dummy 10-second recording (3000 samples at 300 Hz for ch1-3, 9000 at 900 Hz for ch4)
os.makedirs('/tmp/fecg_in', exist_ok=True)
for ch in range(1, 4):
    (np.random.randn(3000) * 1000).astype(np.int32).tofile(f'/tmp/fecg_in/TEST.ch{ch}')
(np.random.randn(9000) * 1000).astype(np.int32).tofile('/tmp/fecg_in/TEST.ch4')
```

```bash
python pipeline.py -i /tmp/fecg_in -o /tmp/fecg_out
```

Output: `/tmp/fecg_out/TEST.mat` — a MATLAB v7.3 / HDF5 file containing per-segment waveforms and beat annotations.

### Parallel pipeline

`pipeline_fast.py` distributes the 84 beat-extraction calls (6 channel-pair × 14 alpha-mix,
when using NUM_CHANNEL=4 and alphaN=7) and the 84 bSQI-scoring calls across CPU cores via
`ProcessPoolExecutor`, and pipelines the refinement pass to overlap with the next segment's
search phase. The signal processing logic is otherwise identical to the sequential version.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    python pipeline_fast.py -i /path/to/input_folder -o /path/to/output_folder
```

Setting `OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1` prevents BLAS from spawning its
own threads inside each worker process, which would oversubscribe the CPU and reduce throughput.

> **Validated:** `pipeline_fast.py` produces bit-identical outputs to the sequential
> pipeline.

## Library Structure

```
python/
├── pipeline.py                          Main pipeline (sequential, entry point)
├── pipeline_fast.py                     Parallel pipeline (validated — see pipeline_fast.md)
└── fecg_open/
    ├── core/                            Signal processing algorithms
    │   ├── beat_simple.py               DP beat tracker (Numba-JIT)
    │   ├── cepstrum_convert.py          TFR → cepstral harmonic-enhancement mask
    │   ├── CFPH.py                      De-shape SST (STFT + cepstrum + synchrosqueezing)
    │   ├── CurveExt_M.py                DP ridge extraction (Numba-JIT)
    │   ├── detect_bsqi.py               Beat-quality index via jqrs vs. gqrs agreement
    │   ├── dwindow.py                   Window derivative (for IFD computation)
    │   ├── ECG_detrend.py               Cascaded-median baseline wander removal
    │   ├── ECG_shrinkage0.py            SVD optimal shrinkage — mECG / fECG template
    │   ├── ECG_shrinkage_median.py      Nonlocal-median shrinkage — fECG waveform
    │   ├── extract_fhr.py               Fetal HR ridge from synchrosqueezed TFR
    │   ├── extract_mhr.py               Maternal HR ridge from synchrosqueezed TFR
    │   ├── fbeats_extract_real.py       Fetal R-peak detection pipeline
    │   ├── fusion_mbeats.py             Multi-orientation maternal beat fusion
    │   ├── mbeats_extract_real.py       Maternal R-peak detection pipeline
    │   ├── mbeats_modify.py             Snap beat locations to local signal peaks
    │   ├── optimal_shrinkage.py         Gavish-Donoho SVD shrinkage (operator norm)
    │   ├── RRconstraint.py              Refractory-period filter for beat sequences
    │   ├── STFT_IFD_fast.py             STFT + instantaneous frequency deviation
    │   ├── synchrosqueeze1win.py        Synchrosqueezing reassignment
    │   ├── tftb_window.py               Flat-Top analysis window
    │   └── peak_detector/
    │       ├── ecgsqi.py                Windowed F1 beat-agreement SQI
    │       ├── qrs_detect2.py           Pan-Tompkins QRS detector (offline)
    │       ├── run_qrsdet_by_seg_ali.py Segmented jqrs wrapper
    │       └── setDetectOptions.py      SQI/detector options dict builder
    └── utils/
        ├── align_beats_to_ecg.py        Fine-align beat indices to local extrema
        ├── findQRSpeaks.py              Sliding-window local-maximum detector
        ├── matlab_smooth_wrapper.py     LOESS smooth matching MATLAB smooth(...,'loess')
        ├── MyReadDataq_32.py            Read little-endian int32 binary channel file
        ├── pan_tompkin_revised.py       Pan-Tompkins detector (adaptive threshold)
        └── parsave_output_file.py       Save pipeline outputs to MATLAB v7.3 MAT-file
```

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

fECG-open includes an automated test suite covering the core signal
processing modules and the full extraction pipeline. Tests are written
using [pytest](https://pytest.org) and can be run from this directory
with:

    pytest tests/ -v

All 58 tests should pass. No additional configuration is required.

### What the suite covers

The unit tests target the four main processing modules individually -
`ECG_detrend`, `optimal_shrinkage`, `RRconstraint`, and `mbeats_modify`
- using controlled synthetic signals with known expected outputs. Key
behaviors verified include baseline removal aggressiveness across
morphology modes (verified against the MATLAB original), beat boundary
filtering, overlapping RR-interval constraint semantics, and shrinkage
matrix validity.

The integration test runs the full pipeline (`pipeline.py`) on a short
real abdominal ECG recording from a healthy volunteer (see
`tests/test_recording/`). It verifies that the pipeline completes
without error, produces a valid output file, and extracts a non-zero
number of fetal beats. This test exercises the real processing path -
shrinkage, beat extraction, and fetal ECG separation - on genuine
cardiac signal rather than synthetic noise.

### Test data

The test recording in `tests/test_recording/` is a 2-minute excerpt of
the full volunteer recording provided in `./data/inputs/`. It is
committed to the repository so that the integration test is
self-contained and reproducible without access to the full dataset.

## License

MIT.

## Citation

If you use this library in your research, please cite: [to be added once the manuscript has been accepted]

## Contributing

Contributions are welcome. Please open an issue or pull request on GitHub.
