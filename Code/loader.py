import re
import numpy as np
import pandas as pd
from scipy import signal


def load_labview_pulse(filepath):
    # Locate end of LabVIEW header and extract sampling rate
    header_end = 0
    fs = 1000.0

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    for idx, line in enumerate(lines):
        line_str = line.strip().lower()
        if "sampling rate" in line_str or "scan rate" in line_str:
            match = re.search(r"[-+]?\d*\.\d+|\d+", line_str)
            if match:
                fs = float(match.group(0))

        if "***end_of_header***" in line_str:
            header_end = idx + 1
            break

    # Parse numeric data table
    df = pd.read_csv(
        filepath,
        skiprows=header_end,
        sep=r"\s+",
        engine="python",
        header=None,
    )
    df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all").reset_index(drop=True)

    if df.empty:
        raise ValueError(f"No numeric data found in {filepath}")

    # Column 1 is current. Column 0 is fallback
    col_idx = 1 if df.shape[1] >= 2 else 0
    raw_sig = df.iloc[:, col_idx].values
    raw_sig = np.nan_to_num(raw_sig)

    if len(raw_sig) < 100:
        raise ValueError(f"Signal too short in {filepath}")

    # Remove DC offset
    detrended = raw_sig - np.mean(raw_sig)

    # 4th-order Butterworth SOS bandpass from 0.5 Hz to 450 Hz
    nyq = 0.5 * fs
    low = max(0.5 / nyq, 0.001)
    high = min(450.0 / nyq, 0.999)
    sos = signal.butter(4, [low, high], btype="band", output="sos")
    cleaned_sig = signal.sosfiltfilt(sos, detrended)

    return np.nan_to_num(cleaned_sig), fs