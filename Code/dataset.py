import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

from labels import resolve_genre_and_song
from loader import load_labview_pulse
from features import extract_window_features


def calibrate_noise_floor(data_path):
    # Reference baseline for noise floor calibration
    baseline_file = data_path / "baseline_1.data"
    if baseline_file.exists():
        try:
            sig, _ = load_labview_pulse(str(baseline_file))
            noise_rms = float(np.sqrt(np.mean(sig ** 2)))
            print(f"[Calibration] Reference Baseline Calibrated: RMS Noise Floor = {noise_rms:.4e} A")
            return max(noise_rms, 1e-12)
        except Exception as e:
            print(f"[Warning] Failed reading baseline_1.data: {e}")

    # Fallback nominal floor if baseline measurement is missing
    print("[Notice] baseline_1.data not found. Using default 50 pA floor.")
    return 5.0e-11


def assemble_windowed_dataset(data_dir, window_sec=5.0, overlap=0.5):
    data_path = Path(data_dir)
    noise_floor_rms = calibrate_noise_floor(data_path)

    rows = []
    labels = []
    groups = []

    for fname in sorted(os.listdir(data_path)):
        if not fname.endswith(".data"):
            continue

        genre, song_id = resolve_genre_and_song(fname)

        # Baseline file is strictly for calibration. Exclude from classification
        if genre in ("Baseline_Calibration", "Unknown"):
            continue

        filepath = data_path / fname
        try:
            sig, fs = load_labview_pulse(str(filepath))
            win_len = int(window_sec * fs)
            hop = int(win_len * (1.0 - overlap))

            # Handle files shorter than window length
            if len(sig) < win_len:
                feats = extract_window_features(sig, fs=fs, noise_floor_rms=noise_floor_rms)
                rows.append(feats)
                labels.append(genre)
                groups.append(song_id)
                continue

            # Sliding window segmentation
            for start in range(0, len(sig) - win_len + 1, hop):
                chunk = sig[start : start + win_len]
                feats = extract_window_features(chunk, fs=fs, noise_floor_rms=noise_floor_rms)
                rows.append(feats)
                labels.append(genre)
                groups.append(song_id)

        except Exception as err:
            print(f"Error processing {fname}: {err}")

    X = pd.DataFrame(rows)
    y = np.array(labels)
    song_groups = np.array(groups)

    return X, y, song_groups