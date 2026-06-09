"""
parsave_output_file.py

Save fECG-open pipeline results to a MATLAB v7.3-compatible MAT-file (HDF5).
The output file is simultaneously:
  - Readable by MATLAB load()
  - Readable by Python h5py.File() (h5py transparently handles the 512-byte
    offset since version 2.1)

-------------------------------------
Three encoding choices are made to match the structure produced by the
MATLAB pipeline:

  final_SQI, final_start_times, final_proc_time
      MATLAB stores these as cell arrays of (1,1) double scalars, not as
      plain double row vectors.  _to_cell() is used for all three so that
      h5py sees them as object-reference arrays (dtype=object) rather than
      flat float64 arrays, matching the MATLAB HDF5 layout exactly.

  final_og_ECG
      MATLAB stores each segment as a cell array of channel column vectors
      ({4x1} cell in MATLAB notation) rather than a single (4, N) matrix.
      _to_og_ecg_cell() converts each segment to a (1, 4) numpy object
      array before the outer _to_cell() wraps segments, producing a two-
      level cell structure that matches MATLAB's HDF5 reference graph:
          outer (1, N_segs) object refs
              -> inner (1, 4) object refs  [one per channel]
                  -> (N, 1) float64 column vector
"""

import os
import numpy as np
import h5py
import hdf5storage


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_cell(lst):
    """
    Convert a Python list to a numpy object array shaped (N, 1).

    hdf5storage with matlab_compatible=True writes arrays in Fortran
    (column-major) order, which reverses dimensions in HDF5.  A (N, 1)
    numpy array is therefore stored as (1, N) in HDF5, which matches
    the layout MATLAB uses for {1 x N} row cell arrays.
    """
    n = len(lst)
    arr = np.empty((n, 1), dtype=object)
    for i, v in enumerate(lst):
        arr[i, 0] = v
    return arr


def _to_og_ecg_cell(segment):
    """
    Encode one segment's og_ECG as a {1 x 4} MATLAB cell of row vectors.

    `segment` arrives as a (1,) numpy object array whose single element is
    a Python list of 4 channel arrays [each shape (N_samples,)].  This wrapper
    is used in main_fECG.py to prevent numpy from auto-stacking the 4 arrays:

        og_ecg_obj = np.empty((1,), dtype=object)
        og_ecg_obj[0] = og_ecg_seg   # list of 4 channel arrays
        final_og_ECG.append(og_ecg_obj)

    The inner cell is a (1, 4) numpy object array - NOT via _to_cell(), which
    would produce (4, 1) - so that hdf5storage's Fortran reversal writes it as
    (4, 1) in HDF5, which MATLAB reads back as {1 x 4}.

    Each channel is stored as a (1, N) numpy row vector so that hdf5storage
    writes it as (N, 1) in HDF5, which MATLAB reads back as a (1 x N) row vector.

    Resulting HDF5 structure (as seen by h5py):
        outer (1, N_segs) object refs     MATLAB: {N_segs x 1}
            inner (4, 1) object refs      MATLAB: {1 x 4}
                channel (N, 1) float64    MATLAB: (1 x N) row vector
    """
    # Unwrap the (1,) object-array wrapper to get the list of 4 channel arrays
    channel_list = np.asarray(segment).flat[0]

    # (1, 4) numpy -> hdf5storage reversal -> (4, 1) HDF5 -> MATLAB {1 x 4}
    n = len(channel_list)
    inner = np.empty((1, n), dtype=object)
    for i, ch in enumerate(channel_list):
        # (1, N) numpy -> hdf5storage reversal -> (N, 1) HDF5 -> MATLAB (1 x N)
        inner[0, i] = np.asarray(ch).reshape(1, -1)
    return inner


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def parsave_output_file(
    output_file_path,
    final_file_names,
    final_start_times,
    final_og_ECG,
    final_SQI,
    final_chs_used,
    final_Om,
    final_Of,
    final_aECG,
    final_mbeats,
    final_fbeats,
    final_proc_time
):
    """
    Save SAVER pipeline results to a MATLAB v7.3-compatible MAT-file.

    Uses hdf5storage.savemat() to produce a genuine MATLAB v7.3 / HDF5 file
    that can be opened with MATLAB load(), h5py.File(), and hdf5storage.

    Parameters
    ----------
    output_file_path : str
        Destination .mat file path.
    final_file_names : list[str]
        Source file name for each segment.
    final_start_times : list[float]
        Segment start time in seconds.
    final_og_ECG : list
        Original (pre-detrend) ECG data per segment.  Each element is either
        a (4, N) float64 array or a sequence of 4 channel arrays.
    final_SQI : list[float]
        Best bSQI score per segment.
    final_chs_used : list[ndarray]
        Channel pair used per segment.
    final_Om : list[ndarray]
        Maternal ECG per segment.
    final_Of : list[ndarray]
        Fetal ECG per segment.
    final_aECG : list[ndarray]
        Abdominal ECG per segment.
    final_mbeats : list[ndarray]
        Maternal R-peak sample indices (1-based) per segment.
    final_fbeats : list[ndarray]
        Fetal R-peak sample indices (1-based) per segment.
    final_proc_time : list[float]
        Cumulative processing time per segment.
    """
    output_dir = os.path.dirname(output_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # hdf5storage.savemat() opens in HDF5 append mode and never purges the
    # internal #refs# group.  Successive overwrites of the same path
    # accumulate one full copy of all reference objects per call, inflating
    # the file by a factor equal to the number of saves.  Deleting the file
    # first forces a clean write every time.
    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    n = len(final_file_names)

    mdict = {
        # Cell arrays of strings -> object array of str
        'final_file_names':   _to_cell(final_file_names),
        'final_start_times':  _to_cell(final_start_times),
        'final_SQI':          _to_cell(final_SQI),
        'final_proc_time':    _to_cell(final_proc_time),
        'final_og_ECG':    _to_cell([_to_og_ecg_cell(seg) for seg in final_og_ECG]),

        # Cell arrays of numeric arrays -> object array
        'final_chs_used':  _to_cell(final_chs_used),
        'final_Om':        _to_cell(final_Om),
        'final_Of':        _to_cell(final_Of),
        'final_aECG':      _to_cell(final_aECG),
        'final_mbeats':    _to_cell(final_mbeats),
        'final_fbeats':    _to_cell(final_fbeats),
    }

    hdf5storage.savemat(
        output_file_path,
        mdict,
        store_python_metadata=False,   # keep file clean for MATLAB
        matlab_compatible=True,        # enforces MATLAB cell/char conventions
    )


# ---------------------------------------------------------------------------
# Smoke test (run as script to verify round-trip)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import tempfile

    test_output = tempfile.mktemp(suffix='.mat')

    # Build og_ECG inputs using the same pattern as main_fECG.py:
    #   og_ecg_obj = np.empty((1,), dtype=object); og_ecg_obj[0] = list_of_4_ch
    def _make_og_ecg(n_samples=1000):
        obj = np.empty((1,), dtype=object)
        obj[0] = [np.random.randn(n_samples) for _ in range(4)]
        return obj

    parsave_output_file(
        test_output,
        final_file_names   = ['subj01_seg1', 'subj01_seg2'],
        final_start_times  = [0.0, 60.0],
        final_og_ECG       = [_make_og_ecg(), _make_og_ecg()],
        final_SQI          = [0.82, 0.91],
        final_chs_used     = [np.array([[1, 2]]), np.array([[1, 3]])],
        final_Om           = [np.random.randn(1000), np.random.randn(1000)],
        final_Of           = [np.random.randn(1000), np.random.randn(1000)],
        final_aECG         = [np.random.randn(1000), np.random.randn(1000)],
        final_mbeats       = [np.array([100, 200, 300]), np.array([101, 201])],
        final_fbeats       = [np.array([150, 250, 350]), np.array([152, 252])],
        final_proc_time    = [12.3, 24.6],
    )

    print(f'Written: {test_output}')

    # Verify it is readable by h5py (= valid HDF5 with MATLAB user-block)
    with h5py.File(test_output, 'r') as f:
        keys = list(f.keys())
        print(f'Top-level keys: {keys}')

        # Check SQI, start_times, proc_time are now object (cell) dtype
        for k in ('final_SQI', 'final_start_times', 'final_proc_time'):
            ds = f[k]
            assert ds.dtype == object, \
                f'{k}: expected object dtype (cell), got {ds.dtype}'
            print(f'{k}: shape={ds.shape}, dtype={ds.dtype}  [cell OK]')

        # Check og_ECG nesting depth: should be 2 levels (seg -> channel -> data)
        og_refs = f['final_og_ECG'][()]
        seg0_ref = og_refs.flat[0]
        seg0 = f[seg0_ref]
        assert seg0.dtype == object, \
            f'final_og_ECG seg0: expected object (inner cell), got {seg0.dtype}'
        ch0_ref = seg0[()].flat[0]
        ch0 = f[ch0_ref]
        assert ch0.dtype != object, \
            f'final_og_ECG ch0: expected float data, got {ch0.dtype} ' \
            f'(extra nesting level still present)'
        print(f'final_og_ECG: 2-level nesting confirmed '
              f'(seg shape={seg0.shape}, ch shape={ch0.shape})')

    # Verify it is readable by hdf5storage round-trip
    data = hdf5storage.loadmat(test_output)
    print(f'final_file_names via hdf5storage: {data["final_file_names"]}')

    # Check file header - first 10 bytes should be "MATLAB 7.3"
    with open(test_output, 'rb') as f:
        header = f.read(10)
    assert header == b'MATLAB 7.3', f'Bad header: {header!r}'
    print('Header OK: MATLAB 7.3')

    os.remove(test_output)
    print('Smoke test passed.')
