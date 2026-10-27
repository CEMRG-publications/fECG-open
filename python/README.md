# fECG-open — Python

> Python translation of the fECG-open MATLAB pipeline: separates maternal and fetal ECG waveforms and R-peak annotations from four-channel abdominal ECG recordings.

## Background

Abdominal ECG electrodes pick up a mixture of two cardiac signals: the maternal ECG (strong) and the fetal ECG (weak). Separating them is a core problem in fetal cardiac monitoring. This library is a 1:1 Python translation of the MATLAB pipeline. It uses the de-shape synchrosqueezing transform for heart-rate tracking, SVD-based optimal shrinkage for template separation, and dynamic-programming beat tracking.

## Requirements

### Python

- **Tested on:** Python 3.12.7
- **Tested on:** Ubuntu 22.04.5 LTS (64-bit)
- **Minimum version:** Python 3.8 (f-strings, `typing.Union`, `math.gcd` with 2 args)
- **Dependencies:**

| Package | Version floor | Notes |
|---|---|---|
| `numpy` | `>=1.21` | Core numeric library |
| `scipy` | `>=1.7` | Signal processing, interpolation, integration |
| `numba` | `>=0.56` | JIT compilation for DP inner loops (`beat_simple`, `CurveExt_M`) |
| `h5py` | `>=3.3` | HDF5 I/O (resume logic, output verification) |
| `hdf5storage` | `>=0.1.18` | MATLAB v7.3-compatible MAT file I/O |
| `matplotlib` | *(optional)* | Used only in `pan_tompkin_revised.py` when `gr=1`; never called in the main pipeline |

`pipeline_fast.py` additionally uses `concurrent.futures` and `multiprocessing` (both standard library).

## Installation

```bash
git clone <repo-url>
cd fECG-open
pip install ".[plot,test]"
```

For exact reproduction of the reported results, install the pinned versions in `requirements-tested.txt`.

## Quick Start

### Sequential pipeline

Input: four binary channel files per recording (`<basename>.ch1` – `.ch4`, Monica DK format).

To verify your installation, run the pipeline on the short test recording (2 segments):

```bash
python python/pipeline.py -i python/tests/test_recording -o python/tests/test_recording
```

The full sample recording (92 segments) can be processed with:

```bash
python python/pipeline.py -i data/inputs -o data/outputs
```

Output: one `.mat` file per recording (MATLAB v7.3 / HDF5) containing per-segment waveforms and beat annotations. See the Output Format section for details.

### Parallel pipeline

`pipeline_fast.py` distributes beat-extraction and bSQI-scoring calls across CPU cores via `ProcessPoolExecutor`. The signal processing logic is otherwise identical to the sequential version and produces identical outputs within floating-point precision.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    python python/pipeline_fast.py -i data/inputs -o data/outputs
```

Setting `OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1` prevents BLAS from spawning its own threads inside each worker process, which would oversubscribe the CPU.

By default, the worker count is set automatically. To override it, specify the desired number of worker processes (e.g., 8):

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    python python/pipeline_fast.py -i data/inputs -o data/outputs --workers 8
```

### Interruption and restart

Both scripts can continue after an interruption. Run the same command again.

- `pipeline.py` saves the output file after every segment and continues after the last saved segment. The first segment after a restart can occasionally select a different fetal beat detector, as the beat count it carries over is the saved count (no padding), which is a few beats lower than in an uninterrupted run.
- `pipeline_fast.py` saves each finished segment in a folder `<output>/<name>.partial/`. The final `<name>.mat` is written only when all segments are done, so a `.mat` file with the final name always contains all successfully processed segments, and a recording with a final file is skipped. After an interruption, the script continues from the first segment without a stored file, and it removes the folder once the final file is written.
- If the sizes of the input files or the settings change between the interrupted run and the restart, `pipeline_fast.py` stops and names the folder to delete.

To inspect the results of an interrupted `pipeline_fast.py` run without computing anything:

```bash
python python/pipeline_fast.py -i data/inputs -o data/outputs --merge-partial
```

This writes `<name>_partial.mat` from the stored segments, with the extra variables `partial`, `n_segments` and `segment_indices`, and keeps the folder. It is meant for inspection only: the final output comes from a complete run.

## Library Structure

```
python/
├── pipeline.py                          Main pipeline (sequential, entry point)
├── pipeline_fast.py                     Parallel pipeline
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
    │       ├── ecgsqi.py                Windowed beat-agreement SQI (minimum of two matching proportions)
    │       ├── qrs_detect2.py           Pan-Tompkins QRS detector (offline)
    │       ├── run_qrsdet_by_seg_ali.py Segmented jqrs wrapper
    │       └── setDetectOptions.py      SQI/detector options dict builder
    └── utils/
        ├── align_beats_to_ecg.py        Fine-align beat indices to local extrema
        ├── findQRSpeaks.py              Sliding-window local-maximum detector
        ├── matlab_smooth_wrapper.py     LOESS smooth matching MATLAB smooth(...,'loess')
        ├── MyReadDataq_32.py            Read little-endian int32 binary channel file
        ├── pan_tompkin_revised.py       Pan-Tompkins detector (adaptive threshold)
        ├── parsave_output_file.py       Save pipeline outputs to MATLAB v7.3 MAT-file
        └── segment_store.py             One file per segment, used by pipeline_fast.py to restart
```

## Output Format

For a description of all output variables, see [Output Format](../README.md#output-format) in the top-level README.

## Testing

fECG-open includes an automated test suite covering the core signal processing modules and the full extraction pipeline. Tests are written using [pytest](https://pytest.org) and can be run from this directory with:

    pytest tests/ -v

All 58 tests should pass. No additional configuration is required.

### What the suite covers

The unit tests target the four main processing modules individually - `ECG_detrend`, `optimal_shrinkage`, `RRconstraint`, and `mbeats_modify` - using controlled synthetic signals with known expected outputs. Key behaviors verified include baseline removal aggressiveness across morphology modes (verified against the MATLAB original), beat boundary filtering, overlapping RR-interval constraint semantics, and shrinkage matrix validity.

The integration test runs the full pipeline (`pipeline.py`) on a short real abdominal ECG recording from a healthy volunteer (see `tests/test_recording/`). It verifies that the pipeline completes without error, produces a valid output file, and extracts a non-zero number of fetal beats. This test exercises the real processing path - shrinkage, beat extraction, and fetal ECG separation - on genuine cardiac signal rather than synthetic noise.

### Test data

The test recording in `tests/test_recording/` is a 2-minute excerpt of the full volunteer recording provided in `./data/inputs/`. It is committed to the repository so that the integration test is self-contained and reproducible without access to the full dataset.

## License

MIT.

## Citation

If you use this library in your research, please cite: [to be added once the manuscript has been accepted]

## Contributing

Contributions are welcome. Please open an issue or pull request on GitHub.
