"""
segment_store.py: one small file per finished segment, so pipeline_fast.py can resume.

Each segment is written into a folder next to the final output file. Every write goes
to a temporary name and is then renamed, so a segment file is either complete or absent.
A segment without a valid candidate gets a small marker file, so a restart does not
compute it again. All functions take explicit paths and data and keep no state.
"""

import json
import os

import numpy as np


class FingerprintMismatch(Exception):
    """The stored segments were made with other input files or other settings."""


def partial_folder(output_folder, file_name):
    return os.path.join(output_folder, file_name + '.partial')


def make_fingerprint(input_folder, file_name, num_channels, settings):
    """Input file names and sizes, plus all option values, as a JSON string."""
    inputs = {}
    for ch in range(1, num_channels + 1):
        name = f'{file_name}.ch{ch}'
        inputs[name] = os.path.getsize(os.path.join(input_folder, name))
    return json.dumps({'inputs': inputs, 'settings': settings}, sort_keys=True,
                      default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))


def _write_atomic(path, write_function):
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        write_function(f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_segment(folder, seg_i, fingerprint, record):
    """
    Save one finished segment. `record` holds: file_name (str), start_time, SQI, proc_time,
    last_fbeats_len (floats), og_ECG (list of 4 equal-length 1-D arrays), chs_used,
    Om, Of, aECG, mbeats, fbeats (arrays).
    """
    os.makedirs(folder, exist_ok=True)
    arrays = dict(record)
    arrays['og_ECG'] = np.vstack(record['og_ECG'])
    arrays['fingerprint'] = np.array(fingerprint)
    arrays['seg_i'] = np.array(seg_i)
    path = os.path.join(folder, f'segment_{seg_i:04d}.npz')
    _write_atomic(path, lambda f: np.savez(f, **arrays))


def write_skipped(folder, seg_i, fingerprint):
    """Mark a segment that had no valid candidate."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f'segment_{seg_i:04d}.skipped')
    text = json.dumps({'seg_i': seg_i, 'fingerprint': fingerprint}).encode('utf-8')
    _write_atomic(path, lambda f: f.write(text))


def read_segments(folder, fingerprint):
    """
    Return {seg_i: record}. A skipped segment has the value None.
    Raises FingerprintMismatch if any file was made with other inputs or settings.
    """
    stored = {}
    if not os.path.isdir(folder):
        return stored
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if name.endswith('.npz'):
            with np.load(path, allow_pickle=False) as f:
                record = {key: f[key] for key in f.files}
            found = str(record.pop('fingerprint'))
            seg_i = int(record.pop('seg_i'))
            record['file_name'] = str(record['file_name'])
            record['og_ECG'] = list(record['og_ECG'])
            for key in ('start_time', 'SQI', 'proc_time', 'last_fbeats_len'):
                record[key] = float(record[key])
        elif name.endswith('.skipped'):
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
            found, seg_i, record = data['fingerprint'], int(data['seg_i']), None
        else:
            continue          # temporary files left by an interrupted write
        if found != fingerprint:
            raise FingerprintMismatch(f'{path} was made with other inputs or settings')
        stored[seg_i] = record
    return stored


def as_output_lists(stored):
    """Turn stored segments (in segment order, skipped ones left out) into the 11 lists
    that parsave_output_file expects, with the same types as a normal run."""
    keys = ('final_file_names', 'final_start_times', 'final_og_ECG', 'final_SQI',
            'final_chs_used', 'final_Om', 'final_Of', 'final_aECG', 'final_mbeats',
            'final_fbeats', 'final_proc_time')
    out = {key: [] for key in keys}
    for seg_i in sorted(stored):
        r = stored[seg_i]
        if r is None:
            continue
        og = np.empty((1,), dtype=object)
        og[0] = r['og_ECG']
        out['final_file_names'].append(r['file_name'])
        out['final_start_times'].append(r['start_time'])
        out['final_og_ECG'].append(og)
        out['final_SQI'].append(r['SQI'])
        out['final_chs_used'].append(r['chs_used'])
        out['final_Om'].append(r['Om'])
        out['final_Of'].append(r['Of'])
        out['final_aECG'].append(r['aECG'])
        out['final_mbeats'].append(r['mbeats'])
        out['final_fbeats'].append(r['fbeats'])
        out['final_proc_time'].append(r['proc_time'])
    return out
