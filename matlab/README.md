# fECG-open — MATLAB Library

> MATLAB pipeline for extracting maternal and fetal ECG waveforms and R-peak annotations from four-channel abdominal ECG recordings.

## Background

Abdominal ECG electrodes pick up a mixture of two cardiac signals: the maternal ECG (strong, from the mother's heart) and the fetal ECG (weak, from the fetal heart). Separating them is a core problem in fetal cardiac monitoring. This library implements the SAVER (Synchrosqueezing-based Adaptive VEctorcardiographic ECG Representation) pipeline, which uses time-frequency ridge-following (de-shape synchrosqueezing transform), SVD-based optimal shrinkage, and dynamic-programming beat tracking to robustly separate the two sources from four abdominal channels sampled at different rates.

## Requirements

### MATLAB

- **Tested on:** MATLAB R2024b (24.2.0.2863752) Update 5
- **Minimum version:** R2016a — required for `movmedian` (base MATLAB); all other language features are compatible with earlier releases
- **Required toolboxes:**
  - Signal Processing Toolbox (`butter`, `filtfilt`, `resample`)
  - Statistics and Machine Learning Toolbox (`quantile`)
  - Curve Fitting Toolbox (`smooth(..., 'loess')`)

> **Deprecation notes:**  
> `histc` (used in `ecgsqi.m`) has been marked "not recommended" since R2014b; it still runs but may emit a warning. `quadl` (used in `optimal_shrinkage.m`) was deprecated in R2012a; it remains available for backwards compatibility.

## Installation

1. Clone or download this repository.
2. In MATLAB, add the library to your path:
   ```matlab
   addpath(genpath('/path/to/fECG_open/matlab'))
   ```
3. Verify by running:
   ```matlab
   help pipeline
   ```

## Quick Start

```matlab
% Set input/output paths and run the pipeline on your recordings.
% Edit the paths at the top of pipeline.m, then run:

pipeline
```

The script reads all `.ch1` files from `input_folder`, processes each recording in
60-second segments, and writes one `.mat` file per recording to `output_folder`.

Minimal synthetic example (generates a dummy file and processes it):

```matlab
% Write a dummy 300 Hz / 900 Hz four-channel recording (10 seconds each)
for ch = 1:3
    data = int32(randn(1, 3000) * 1000);
    fid = fopen(sprintf('/tmp/TEST.ch%d', ch), 'w', 'l');
    fwrite(fid, data, 'int32'); fclose(fid);
end
data = int32(randn(1, 9000) * 1000);
fid = fopen('/tmp/TEST.ch4', 'w', 'l'); fwrite(fid, data, 'int32'); fclose(fid);

% Run pipeline (edit input_folder / output_folder in pipeline.m first)
% input_folder  = '/tmp';
% output_folder = '/tmp/out';
pipeline
```

## Library Structure

```
matlab/
├── pipeline.m                         Main pipeline script (entry point)
└── fecg_open/
    ├── core/                          Signal processing algorithms
    │   ├── beat_simple.m              DP beat tracker (log-Gaussian transition cost)
    │   ├── cepstrum_convert.m         TFR → cepstral harmonic-enhancement mask
    │   ├── CFPH.m                     De-shape SST (STFT + cepstrum + synchrosqueezing)
    │   ├── CurveExt_M.m               DP ridge extraction from a TF energy matrix
    │   ├── detect_bsqi.m              Beat-quality index via jqrs vs. gqrs agreement
    │   ├── dwindow.m                  Window derivative (for IFD computation)
    │   ├── ECG_detrend.m              Cascaded-median baseline wander removal
    │   ├── ECG_shrinkage0.m           SVD optimal shrinkage — mECG / fECG template
    │   ├── ECG_shrinkage_median.m     Nonlocal-median shrinkage — fECG waveform
    │   ├── Extract_fhr.m              Fetal HR ridge from synchrosqueezed TFR
    │   ├── Extract_mhr.m              Maternal HR ridge from synchrosqueezed TFR
    │   ├── fbeats_extract_real.m      Fetal R-peak detection pipeline
    │   ├── fusion_mbeats.m            Multi-orientation maternal beat fusion
    │   ├── mbeats_extract_real.m      Maternal R-peak detection pipeline
    │   ├── mbeats_modify.m            Snap beat locations to local signal peaks
    │   ├── optimal_shrinkage.m        Gavish-Donoho SVD shrinkage (operator norm)
    │   ├── RRconstraint.m             Refractory-period filter for beat sequences
    │   ├── STFT_IFD_fast.m            STFT + instantaneous frequency deviation
    │   ├── synchrosqueeze1win.m       Synchrosqueezing reassignment
    │   ├── tftb_window.m              Flat-Top analysis window
    │   └── peak_detector/
    │       ├── ecgsqi.m               Windowed F1 beat-agreement SQI
    │       ├── qrs_detect2.m          Pan-Tompkins QRS detector (offline)
    │       ├── run_qrsdet_by_seg_ali.m  Segmented jqrs wrapper
    │       └── setDetectOptions.m     SQI/detector options struct builder
    └── utils/
        ├── align_beats_to_ecg.m       Fine-align beat indices to local extrema
        ├── findQRSpeaks.m             Sliding-window local-maximum detector
        ├── MyReadDataq_32.m           Read little-endian int32 binary channel file
        ├── pan_tompkin_revised.m      Pan-Tompkins detector (adaptive threshold)
        └── parsave_output_file.m      Save pipeline outputs to MAT-file (v7.3)
```

## License

MIT.

## Citation / Reference

If you use this library in your research, please cite: [to be added once the manuscript has been accepted]

## Contributing

Contributions are welcome. Please open an issue or pull request on GitHub.

## Acknowledgements

The beat detection and signal quality index scoring in this library uses
a modified version of the peak-detector toolbox by Johnson et al.:

  Johnson, A. E. W., Behar, J., Andreotti, F., Clifford, G. D. and Oster, J.
  (2015). Multimodal heart beat detection using signal quality indices,
  Physiological Measurement 36: 1665-1677.

  Source: https://github.com/alistairewj/peak-detector

The synchrosqueezed transform window functions (dwindow, tftb_window) are adapted
from the SST_compare toolbox:

  Yang, H. Robustness Analysis of Synchrosqueezed Transforms, preprint, 2014.
  Source: https://github.com/HaizhaoYang/SST_compare
  Original author: F. Auger. Copyright (c) 1996 CNRS (France). BSD-3-Clause license.
