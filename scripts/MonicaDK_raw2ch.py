#!/usr/bin/env python3
"""
MonicaDK_raw2ch.py — Monica AN24 .raw → .ch extractor

Converts a Monica AN24 raw recording (.raw) into four channel files
(.ch1, .ch2, .ch3, .ch4) byte-for-byte identical to the Monica DK
"Export AN24 RAW Abfecg file → 32-bit integer" output.

Usage:
    python MonicaDK_raw2ch.py <file.raw> [--outdir DIR] [--test]

    --outdir DIR   Write .ch files here (default: same dir as .raw)
    --test         After extraction, compare output byte-for-byte against
                   reference .ch files in the same directory as the .raw,
                   reporting PASS/FAIL for each channel.
"""

import sys
import os
import struct
import argparse
import warnings
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Constants derived from reverse engineering (see format spec at end of file)
# ──────────────────────────────────────────────────────────────────────────────

# NOTE: The first 4 bytes of the .raw file (and its secondary header) are NOT
# a fixed format magic.  They hold a per-recording session identifier that the
# device also repeats as word 1 of every data-block header.  Value varies with
# each recording (e.g. 0x004A47B5, 0x008C259D, 0x00897CED across three files).
# True format invariants are at 0x008 (0x00064000) and 0x00C (0x00010001).
FILE_HEADER_BYTES = 1024         # bytes before the first data block
BLOCK_INT32S      = 512          # int32 words per data block (= 2048 bytes)
BLOCK_BYTES       = BLOCK_INT32S * 4

# Minimum ratio of file-size-derived sample count to header sample count that
# triggers the stale-header fallback.  Observed ratios across five test files:
#   correct headers: 1.0005 – 1.0064   stale-low header (v4): 58.95
# 2.0 leaves a factor of ~280 headroom below v4 and ~200 above the correct-
# header ceiling, making misclassification implausible in practice.
STALE_HDR_RATIO   = 2.0

# Superframe start offsets (in int32 words) within each block.
# Each block = [2-word block header] + 4×[126-word superframe] + 3×[2-word sub-header]
# Layout: hdr(2) sf(126) sh(2) sf(126) sh(2) sf(126) sh(2) sf(126)
SF_OFFSETS        = [2, 130, 258, 386]
FRAMES_PER_SF     = 21    # frames (sample groups) per superframe
INT32S_PER_FRAME  = 6     # int32 words per frame
SAMPLES_PER_BLOCK = len(SF_OFFSETS) * FRAMES_PER_SF  # = 84 CH1/CH2/CH3 samples

# Frame layout within each 6-word group:
#   word 0 → CH1[k]
#   word 1 → CH4[3k]
#   word 2 → CH2[k]
#   word 3 → CH4[3k+1]
#   word 4 → CH3[k]
#   word 5 → CH4[3k+2]
FRAME_CH1_OFF = 0
FRAME_CH4A_OFF = 1
FRAME_CH2_OFF = 2
FRAME_CH4B_OFF = 3
FRAME_CH3_OFF = 4
FRAME_CH4C_OFF = 5

# Pre-computed column indices (within a 512-word block) for each channel.
# _CH1/2/3_COLS: 84 int32 word indices → one CH sample per frame per block.
# _CH4_COLS: 252 int32 word indices → three CH4 samples per frame per block.
# Used by extract_channels() for numpy fancy indexing across all blocks at once.
def _build_channel_cols():
    """Build numpy index arrays mapping Monica AN24 block positions to per-channel samples."""
    ch1, ch2, ch3, ch4 = [], [], [], []
    for sf_off in SF_OFFSETS:
        for j in range(FRAMES_PER_SF):
            base = sf_off + j * INT32S_PER_FRAME
            ch1.append(base + FRAME_CH1_OFF)
            ch2.append(base + FRAME_CH2_OFF)
            ch3.append(base + FRAME_CH3_OFF)
            ch4 += [base + FRAME_CH4A_OFF,
                    base + FRAME_CH4B_OFF,
                    base + FRAME_CH4C_OFF]
    return (np.array(ch1, dtype=np.intp),
            np.array(ch2, dtype=np.intp),
            np.array(ch3, dtype=np.intp),
            np.array(ch4, dtype=np.intp))

_CH1_COLS, _CH2_COLS, _CH3_COLS, _CH4_COLS = _build_channel_cols()

# ADC offset: ch_value = raw_int32 + ADC_OFFSET
# The device stores samples as signed 24-bit values biased by –2^23,
# so adding 2^23 recovers the unsigned 24-bit ADC count.
ADC_OFFSET = 1 << 23  # 8 388 608

# Sampling rates (confirmed from strings in Monica DK 1.8 binary)
SAMPLE_RATE_CH1_3 = 300   # Hz for CH1, CH2, CH3
SAMPLE_RATE_CH4   = 900   # Hz for CH4 (= 3 × CH1 rate)

# Header field byte offsets in the second 512-byte header block (at file offset 0x200)
H2_OFFSET         = 0x200
H2_SESSION_ID_OFF = 0x200   # session ID repeated in secondary header
H2_TS_START_OFF   = 0x204   # recording start timestamp (device ticks)
H2_TS_END_OFF     = 0x208   # recording end timestamp (device ticks)
H2_SUPERFRAMES_OFF = 0x210  # total superframe count + 2 (see NOTE below)
# NOTE: Monica DK stores (actual_superframes + 2) at H2_SUPERFRAMES_OFF.
# The exported sample count is therefore: (field_value - 2) × FRAMES_PER_SF.
H2_SF_BIAS        = 2


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def parse_header(data: bytes, file_size: int = 0, filename: str = "") -> dict:
    """
    Parse the 1024-byte file header of a Monica AN24 .raw file.

    Parameters
    ----------
    data      : bytes  At least 1024 bytes from the start of the .raw file.
    file_size : int    Total file size in bytes (used for fallback sample count).
                       Pass 0 to skip the fallback check.
    filename  : str    File path shown in any warning messages.

    Returns
    -------
    dict with keys:
        session_id      – per-recording identifier stored at header[0x000]
                          (also repeated in each block header word 1; varies per file)
        ts_start        – recording start timestamp (device ticks, unit unknown)
        ts_end          – recording end timestamp
        n_superframes   – actual number of complete 21-sample superframes
        n_samples       – total samples per CH1/CH2/CH3 channel
        n_samples_ch4   – total samples for CH4 (= 3 × n_samples)
        sample_rate     – CH1/CH2/CH3 sample rate in Hz (always 300)
        sample_rate_ch4 – CH4 sample rate in Hz (always 900)
        cal             – list of 20 per-electrode calibration values
                          (first 10 from primary header, next 10 from secondary)
    """
    if len(data) < FILE_HEADER_BYTES:
        raise ValueError(
            f"File too short: need {FILE_HEADER_BYTES} bytes for header, "
            f"got {len(data)}"
        )

    # ── Primary header (at offset 0x000, 80 bytes of data, rest zeros) ────────
    # Word 0 is a per-recording session ID (not a fixed format magic).
    session_id = struct.unpack_from("<I", data, 0x000)[0]

    # 10 per-electrode calibration values (offsets 0x020–0x047)
    cal_primary = list(struct.unpack_from("<10I", data, 0x020))

    # ── Secondary header (at offset 0x200, 64 bytes of data, rest zeros) ──────
    ts_start = struct.unpack_from("<I", data, H2_TS_START_OFF)[0]
    ts_end   = struct.unpack_from("<I", data, H2_TS_END_OFF)[0]

    # Superframe count field holds (actual_count + H2_SF_BIAS)
    sf_field      = struct.unpack_from("<I", data, H2_SUPERFRAMES_OFF)[0]
    n_superframes = sf_field - H2_SF_BIAS
    n_from_header = n_superframes * FRAMES_PER_SF

    # Monica DK firmware initialises the superframe count field at recording
    # start and may not update it on abnormal termination (power loss, crash).
    # This leaves a truncated header count despite intact data blocks on disk.
    # Only treat the header as stale-low when the file-size-derived count is
    # substantially larger (> STALE_HDR_RATIO × header); small excesses are
    # normal and indicate trailing data from a subsequent concatenated session,
    # which the boundary detectors in extract_channels() will clip correctly.
    if file_size > 0:
        n_from_filesize = (
            (file_size - FILE_HEADER_BYTES) // BLOCK_BYTES
        ) * SAMPLES_PER_BLOCK
        if n_from_filesize > n_from_header * STALE_HDR_RATIO:
            warnings.warn(
                f"Header sample count ({n_from_header:,}) is less than the "
                f"file-size-derived count ({n_from_filesize:,}) in "
                f"'{filename}' (ratio {n_from_filesize/n_from_header:.1f}×). "
                f"The superframe field at 0x210 may be stale "
                f"(abnormal termination). Using file-size-derived value.",
                stacklevel=2,
            )
            n_samples = n_from_filesize
        else:
            n_samples = n_from_header
    else:
        n_samples = n_from_header

    # 10 per-electrode calibration values from secondary header (0x218–0x23F)
    cal_secondary = list(struct.unpack_from("<10I", data, 0x218))

    return {
        "session_id":       session_id,
        "ts_start":         ts_start,
        "ts_end":           ts_end,
        "n_superframes":    n_superframes,
        "n_samples":        n_samples,
        "n_samples_ch4":    n_samples * 3,
        "sample_rate":      SAMPLE_RATE_CH1_3,
        "sample_rate_ch4":  SAMPLE_RATE_CH4,
        "cal":              cal_primary + cal_secondary,
    }


def extract_channels(raw_path: str, metadata: dict) -> tuple:
    """
    Read the data section of a .raw file and return four channel arrays.

    Uses numpy fancy indexing over all blocks at once instead of a Python loop.
    Pre-computed column index arrays (_CH1/2/3/4_COLS) select the correct words
    from each 512-word block in a single gather operation per channel.

    Parameters
    ----------
    raw_path : str   Path to the .raw file.
    metadata : dict  As returned by parse_header().

    Returns
    -------
    (ch1, ch2, ch3, ch4) – four numpy int32 arrays (little-endian).
    """
    n = metadata["n_samples"]

    raw_file_bytes = os.path.getsize(raw_path)
    n_blocks = (raw_file_bytes - FILE_HEADER_BYTES) // BLOCK_BYTES

    # Shape as (n_blocks, 512) so each row is one block — enables column slicing.
    raw_mm = np.memmap(raw_path, dtype="<i4", mode="r",
                       offset=FILE_HEADER_BYTES,
                       shape=(n_blocks, BLOCK_INT32S))

    # ── Session boundary detection ────────────────────────────────────────────
    # Three detectors, each covering a distinct file family (verified safe on
    # all four test files — no detector fires before the correct endpoint on
    # any file, so they can be evaluated in parallel and the minimum taken):
    #
    #   D1  hdr[0] == 0x00010001  (word 0)
    #       Fires once at the exact session-end block. Observed in v3-family
    #       files. Single occurrence; no false positives seen.
    #
    #   D2  sh_b[0] == 0x00010000  AND  sh_b[1] != session_id  (words 256-257)
    #       sh_b carries either a slow counter or a session-ID marker. When the
    #       marker holds a foreign session ID the block straddles a recording
    #       boundary; sf0+sf1 are still valid, sf2+sf3 belong to the next
    #       session. Observed in v1- and v2-family (concatenated) files.
    #
    #   D3  sh_a[0] == 0x00010000  AND  sh_a[1] != session_id  (words 128-129)
    #       Same marker structure as D2 but in the sh_a sub-header position.
    #       Observed in v4-family files (stale-low header, data from multiple
    #       sessions concatenated without the D2 marker being set).
    #
    # All three clip n to: boundary_block * SAMPLES_PER_BLOCK + 2 * FRAMES_PER_SF
    # (retaining sf0+sf1 of the boundary block, discarding sf2+sf3).
    sid_i32 = np.int32(metadata["session_id"])

    d1 = np.flatnonzero(raw_mm[:, 0] == np.int32(0x00010001))
    d2 = np.flatnonzero(
        (raw_mm[:, 256] == np.int32(0x00010000)) &
        (raw_mm[:, 257] != sid_i32)
    )
    d3 = np.flatnonzero(
        (raw_mm[:, 128] == np.int32(0x00010000)) &
        (raw_mm[:, 129] != sid_i32)
    )

    # Per-detector offsets reflect which sub-header position fired and therefore
    # how many complete superframes precede it within the boundary block:
    #   D1  hdr[0] (word 0)      → block header is the terminator → 0 valid SFs → +0
    #   D3  sh_a   (words 128-129) → fires between sf0 and sf1    → 1 valid SF  → +FRAMES_PER_SF
    #   D2  sh_b   (words 256-257) → fires between sf1 and sf2    → 2 valid SFs → +2*FRAMES_PER_SF
    #       Exception: if sh_a within the same D2 block holds 0x00010001 (an
    #       embedded termination marker), sf1 is already next-session data and
    #       only sf0 is valid → +FRAMES_PER_SF.
    boundary_candidates = []
    if d1.size:
        b = int(d1[0])
        boundary_candidates.append(b * SAMPLES_PER_BLOCK)
    if d2.size:
        b = int(d2[0])
        if raw_mm[b, 128] == np.int32(0x00010001):
            offset = FRAMES_PER_SF        # sh_a marks termination: sf0 only
        else:
            offset = 2 * FRAMES_PER_SF    # normal D2: sf0 + sf1 valid
        boundary_candidates.append(b * SAMPLES_PER_BLOCK + offset)
    if d3.size:
        b = int(d3[0])
        boundary_candidates.append(b * SAMPLES_PER_BLOCK + FRAMES_PER_SF)

    if boundary_candidates:
        n_actual = min(n, *boundary_candidates)
    else:
        n_actual = n

    # ── Vectorised extraction ─────────────────────────────────────────────────
    # Process only the blocks we need.  ceil(n_actual / 84), clamped to n_blocks.
    n_proc = min((-(-n_actual // SAMPLES_PER_BLOCK)), n_blocks)  # ceil division
    raw = raw_mm[:n_proc]  # view: (n_proc, 512)

    # Gather one channel per fancy-index call.  Each call reads exactly the
    # words it needs (84 or 252 per block) and returns a contiguous C array.
    # += applies the ADC offset in-place, avoiding a second temporary array.
    ch1 = raw[:, _CH1_COLS].ravel()   # (n_proc × 84,)  dtype <i4
    ch1 += np.int32(ADC_OFFSET)
    ch2 = raw[:, _CH2_COLS].ravel()
    ch2 += np.int32(ADC_OFFSET)
    ch3 = raw[:, _CH3_COLS].ravel()
    ch3 += np.int32(ADC_OFFSET)
    ch4 = raw[:, _CH4_COLS].ravel()   # (n_proc × 252,) dtype <i4
    ch4 += np.int32(ADC_OFFSET)

    del raw_mm

    # Trim to actual sample count (handles partial last block and session boundary).
    return ch1[:n_actual], ch2[:n_actual], ch3[:n_actual], ch4[:n_actual * 3]


def write_channel(samples: np.ndarray, metadata: dict, path: str) -> None:
    """
    Write a channel array to a .ch file.

    The .ch file format is headerless: a flat sequence of 32-bit signed
    little-endian integers, one per sample.

    Parameters
    ----------
    samples  : numpy array of dtype '<i4'
    metadata : dict (unused for writing, kept for API symmetry)
    path     : output file path
    """
    # Ensure correct dtype and byte order before writing
    if samples.dtype != np.dtype("<i4"):
        samples = samples.astype("<i4")
    samples.tofile(path)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    """Parse CLI arguments and run the Monica AN24 .raw-to-.ch conversion."""
    parser = argparse.ArgumentParser(
        description="Extract Monica AN24 .raw → .ch1/.ch2/.ch3/.ch4"
    )
    parser.add_argument("raw", help="Path to the input .raw file")
    parser.add_argument("--outdir", default=None,
                        help="Output directory (default: same as .raw file)")
    parser.add_argument("--test", action="store_true",
                        help="Compare output byte-for-byte against reference "
                             ".ch files in the same directory as the .raw")
    args = parser.parse_args()

    raw_path = args.raw

    # ── Input validation ───────────────────────────────────────────────────
    if not os.path.isfile(raw_path):
        print(f"ERROR: file not found: {raw_path}", file=sys.stderr)
        sys.exit(1)

    raw_size = os.path.getsize(raw_path)
    if raw_size < FILE_HEADER_BYTES + BLOCK_BYTES:
        print(f"ERROR: file too small to contain any data blocks ({raw_size} bytes)",
              file=sys.stderr)
        sys.exit(1)

    data_bytes = raw_size - FILE_HEADER_BYTES
    if data_bytes % BLOCK_BYTES != 0:
        print(f"WARNING: data section ({data_bytes} bytes) is not a multiple of "
              f"block size ({BLOCK_BYTES} bytes); trailing bytes will be ignored.")

    # ── Parse header ───────────────────────────────────────────────────────
    with open(raw_path, "rb") as fh:
        header_bytes = fh.read(FILE_HEADER_BYTES)

    try:
        meta = parse_header(header_bytes, file_size=raw_size, filename=raw_path)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # ── Output paths ───────────────────────────────────────────────────────
    stem    = os.path.splitext(os.path.basename(raw_path))[0]
    outdir  = args.outdir if args.outdir else os.path.dirname(os.path.abspath(raw_path))
    os.makedirs(outdir, exist_ok=True)

    out_paths = {
        1: os.path.join(outdir, stem + ".ch1"),
        2: os.path.join(outdir, stem + ".ch2"),
        3: os.path.join(outdir, stem + ".ch3"),
        4: os.path.join(outdir, stem + ".ch4"),
    }

    # ── Summary ────────────────────────────────────────────────────────────
    duration_s = meta["n_samples"] / meta["sample_rate"]
    print(f"Input  : {raw_path}  ({raw_size:,} bytes)")
    print(f"Session: 0x{meta['session_id']:08X}")
    print(f"Samples: {meta['n_samples']:,} per CH1/CH2/CH3  |  "
          f"{meta['n_samples_ch4']:,} for CH4")
    print(f"Rates  : CH1-CH3 = {meta['sample_rate']} Hz  |  "
          f"CH4 = {meta['sample_rate_ch4']} Hz")
    print(f"Duration: {duration_s:.1f} s  ({duration_s/60:.1f} min)")
    print()

    # ── Extract ────────────────────────────────────────────────────────────
    print("Extracting channels...", flush=True)
    ch1, ch2, ch3, ch4 = extract_channels(raw_path, meta)

    # ── Write ──────────────────────────────────────────────────────────────
    for ch_num, arr in [(1, ch1), (2, ch2), (3, ch3), (4, ch4)]:
        write_channel(arr, meta, out_paths[ch_num])
        print(f"  Written CH{ch_num}: {out_paths[ch_num]}  "
              f"({os.path.getsize(out_paths[ch_num]):,} bytes)")

    print()

    # ── Self-test ──────────────────────────────────────────────────────────
    if args.test:
        raw_dir = os.path.dirname(os.path.abspath(raw_path))
        all_pass = True
        print("=== Self-test: byte-for-byte comparison ===")
        for ch_num in [1, 2, 3, 4]:
            ref_path = os.path.join(raw_dir, stem + f".ch{ch_num}")
            if not os.path.isfile(ref_path):
                print(f"  CH{ch_num}: SKIP  (reference not found: {ref_path})")
                continue

            with open(out_paths[ch_num], "rb") as fh:
                got = fh.read()
            with open(ref_path, "rb") as fh:
                ref = fh.read()

            if got == ref:
                print(f"  CH{ch_num}: PASS  ({len(ref):,} bytes match exactly)")
            else:
                all_pass = False
                n_diff = sum(a != b for a, b in zip(got, ref))
                size_diff = len(got) - len(ref)
                print(f"  CH{ch_num}: FAIL  "
                      f"{n_diff:,} bytes differ, size delta={size_diff:+d}")
                # Show first discrepancy
                for i, (a, b) in enumerate(zip(got, ref)):
                    if a != b:
                        print(f"    First diff at byte {i}: "
                              f"got=0x{a:02x}, ref=0x{b:02x}")
                        break

        print()
        if all_pass:
            print("Result: ALL CHANNELS PASS")
        else:
            print("Result: ONE OR MORE CHANNELS FAILED")
            sys.exit(2)


if __name__ == "__main__":
    main()


# ══════════════════════════════════════════════════════════════════════════════
# FILE FORMAT SPECIFICATION (reverse-engineered)
# ══════════════════════════════════════════════════════════════════════════════
#
# ── Monica AN24 .raw file ─────────────────────────────────────────────────────
#
# Overall layout:
#   [1024-byte file header] [N × 2048-byte data blocks]
#
# File header (1024 bytes):
#   Primary block at 0x000 (80 bytes of data, padded to 512 bytes with zeros):
#     0x000 [4B LE uint32] session_id — per-recording identifier (varies per file;
#                          also appears at secondary[0x200] and each block word 1)
#     0x004 [4B LE uint32] device/session ID
#     0x008 [4B LE uint32] field08 (meaning unclear; value ≈ 409 600)
#     0x00C [4B LE uint32] field0C (version? value = 0x00010001)
#     0x010 [4B LE uint32] ts_start — recording start (device ticks)
#     0x014 [4B LE uint32] initial TypeB sub-header counter value
#     0x018 [4B LE uint32] field18 (value ≈ 1806)
#     0x01C [4B LE uint32] field1C (baseline magnitude, ≈ 7 133 892)
#     0x020 [40B]          10 × LE uint32 per-electrode calibration values
#     0x048–0x1FF          zeros
#
#   Secondary block at 0x200 (64 bytes of data, padded to 512 bytes with zeros):
#     0x200 [4B LE uint32] session_id (repeated; in a complete recording equals
#                          primary[0x000]; may differ in interrupted recordings)
#     0x204 [4B LE uint32] ts_start (same as 0x010)
#     0x208 [4B LE uint32] ts_end — recording end (device ticks)
#     0x20C [4B LE uint32] value = 2 (purpose unclear)
#     0x210 [4B LE uint32] (n_superframes + 2)  ← key field for sample count
#                           n_samples = (field - 2) × 21
#     0x214 [4B LE uint32] field214 (purpose unclear)
#     0x218 [40B]          10 × LE uint32 per-electrode calibration values
#     0x240–0x3FF          zeros
#
# Data section (N blocks of 2048 bytes each, N = (file_size - 1024) / 2048):
#
#   Each block (512 × LE int32):
#     words  0–1   block header: [type_word, session_id]
#                  type_word = 0x00010000 (constant across all files);
#                  session_id = per-recording value from header[0x000];
#                  the data layout is identical regardless of type_word.
#     words  2–127  superframe 0  (126 int32s = 21 frames × 6 int32s)
#     words 128–129 sub-header A  [0xFDxx0004, timestamp/status]
#     words 130–255 superframe 1  (126 int32s)
#     words 256–257 sub-header B  word 0 = type marker; word 1 = payload
#                  Two observed forms:
#                    [0x00010003, counter] — slow incrementing counter
#                    [0x00010000, session_id] — session ID marker
#                  When word 0 == 0x00010000 and word 1 != session_id,
#                  sf2 and sf3 belong to the next recording; stop extraction.
#     words 258–383 superframe 2  (126 int32s)
#     words 384–385 sub-header A  [0xFDxx0004, timestamp/status]
#     words 386–511 superframe 3  (126 int32s)
#
#   Each superframe: 21 frames.
#   Each frame (6 × LE int32), for global frame index k:
#     word 0  raw CH1[k]
#     word 1  raw CH4[3k]
#     word 2  raw CH2[k]
#     word 3  raw CH4[3k+1]
#     word 4  raw CH3[k]
#     word 5  raw CH4[3k+2]
#
#   ADC encoding:
#     exported_value = raw_int32 + 2^23
#     The device stores 24-bit ADC counts biased by –2^23 (so that the
#     mid-scale reads as zero in the raw stream). Adding 2^23 restores the
#     unsigned ADC count, which is what Monica DK writes to the .ch files.
#
# ── Monica AN24 .ch file ──────────────────────────────────────────────────────
#
#   Format: headerless flat array of 32-bit signed little-endian integers.
#   No metadata, no padding.
#
#   CH1, CH2, CH3: n_samples int32s each  (at 300 Hz)
#   CH4:           3 × n_samples int32s   (at 900 Hz)
#
#   Value range: 24-bit unsigned ADC counts (0 – 16 777 215), stored in the
#   low 24 bits of each int32; the high byte is always 0x00.
#
#   Sample k of CHx = (raw_int32 at frame-offset for CHx, block b, superframe s,
#                       frame j) + 8 388 608
#   where k = b×84 + s×21 + j
#
# ══════════════════════════════════════════════════════════════════════════════
