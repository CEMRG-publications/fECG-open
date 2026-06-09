"""
MyReadDataq_32.py

Read a little-endian int32 binary data file (MATLAB: fopen with 'l' endianness flag,
fread as 'int32'). This matches MATLAB's MyReadDataq_32 function, which reads raw
binary ECG channel files (.ch1–.ch4) from the SAVER data acquisition system.

Returns empty array if the file cannot be opened (matching MATLAB's -1 check on fopen).
"""

import numpy as np


def MyReadDataq_32(filestr):
    """
    Read a little-endian int32 binary data file.

    Opens the binary file at path filestr, reads its entire contents as
    signed 32-bit integers (little-endian), and returns them as a 1-D array.
    This matches MATLAB's fopen(filestr,'r','l') + fread(fi,'int32').

    This format is used by the SAVER data acquisition system to store
    multi-channel abdominal ECG recordings (.ch1–.ch4 files). The caller is
    responsible for any further reshaping or DC-removal.

    Parameters
    ----------
    filestr : str
        Full path to the binary file.

    Returns
    -------
    a : ndarray, dtype int32
        1-D array of raw samples. Returns an empty array if the file
        cannot be opened (matching MATLAB's fi == -1 guard).
    """
    try:
        # Read entire file as little-endian int32
        with open(filestr, 'rb') as f:
            # Read all bytes and interpret as int32, little-endian
            data = np.fromfile(f, dtype=np.int32, sep='')
        return data

    except FileNotFoundError:
        print(f"{filestr} file not found!")
        return np.array([], dtype=np.int32)
    except Exception as e:
        print(f"Error reading {filestr}: {e}")
        return np.array([], dtype=np.int32)


if __name__ == "__main__":
    # Test example
    import os

    # Create a test file
    test_file = "test_binary.bin"
    test_data = np.array([1, 2, 3, 100, -50, 12345], dtype=np.int32)

    # Write test data
    with open(test_file, 'wb') as f:
        test_data.astype(np.int32).tofile(f)

    # Read it back
    read_data = MyReadDataq_32(test_file)

    print(f"Written: {test_data}")
    print(f"Read:    {read_data}")
    print(f"Match: {np.array_equal(test_data, read_data)}")

    # Cleanup
    os.remove(test_file)
