import numpy as np
from scipy import signal
from scipy.stats import skew, kurtosis


def extract_window_features(sig_window, fs=1000.0, noise_floor_rms=6.59e-11):
    feats = {}
    sig = np.nan_to_num(sig_window)

    # Basic amplitude metrics and peak-to-average ratio
    abs_sig = np.abs(sig)
    rms = float(np.sqrt(np.mean(sig**2)))
    peak = float(np.max(abs_sig))

    feats["rms"] = rms
    feats["max_amplitude"] = peak
    feats["crest_factor"] = float(np.clip(peak / (rms + 1e-12), 1.0, 50.0))

    # SNR. Signal-to-noise ratio
    snr = 20.0 * np.log10((rms + 1e-12) / (noise_floor_rms + 1e-12))
    feats["snr_db"] = float(np.clip(snr, -20.0, 80.0))

    # Waveform symmetry and heavy tails
    feats["skewness"] = float(np.clip(skew(sig), -10.0, 10.0))
    feats["kurtosis"] = float(np.clip(kurtosis(sig), -10.0, 30.0))

    # Rate of sign transitions
    zc = np.where(np.diff(sig > 0))[0]
    feats["zcr"] = float(len(zc) / len(sig)) if len(sig) else 0.0

    # Pulse rate & rhythm tracking (3-Sigma noise floor)
    prominence = max(3.0 * noise_floor_rms, np.std(sig) * 0.4)
    peaks, _ = signal.find_peaks(sig, distance=int(0.25 * fs), prominence=prominence)
    if len(peaks) > 2:
        ibi = np.diff(peaks) / fs
        avg_ibi = np.mean(ibi)
        feats["bpm"] = float(np.clip(60.0 / avg_ibi, 30.0, 240.0)) if avg_ibi > 0 else 0.0
        feats["tempo_variation_cv"] = float(np.clip(np.std(ibi) / (avg_ibi + 1e-9), 0.0, 5.0))
    else:
        feats["bpm"] = 0.0
        feats["tempo_variation_cv"] = 0.0

    # Power spectral distribution
    win_len = min(len(sig), int(fs))
    freqs, psd = signal.welch(sig, fs=fs, nperseg=win_len)
    psd = np.nan_to_num(psd)

    total_power = np.sum(psd)
    feats["log_total_power"] = float(np.log10(total_power + 1e-14))

    # band power integration; 1e-14 offsets ambient transducer floor
    denom = total_power + 1e-14
    sub_bass = np.sum(psd[(freqs >= 5) & (freqs < 35)]) / denom
    bass = np.sum(psd[(freqs >= 35) & (freqs < 100)]) / denom
    mid = np.sum(psd[(freqs >= 100) & (freqs < 250)]) / denom
    high = np.sum(psd[(freqs >= 250) & (freqs <= 450)]) / denom

    feats["band_sub_bass"] = float(np.clip(sub_bass, 0.0, 1.0))
    feats["band_bass"] = float(np.clip(bass, 0.0, 1.0))
    feats["band_mid"] = float(np.clip(mid, 0.0, 1.0))
    feats["band_high"] = float(np.clip(high, 0.0, 1.0))

    feats["log_sub_to_mid"] = float(np.log10((sub_bass + 1e-5) / (mid + 1e-5)))
    feats["log_bass_to_high"] = float(np.log10((bass + 1e-5) / (high + 1e-5)))
    feats["spectral_centroid"] = float(np.clip(np.sum(freqs * psd) / denom, 0.0, fs / 2))

    # Spectral flux
    slices = 8
    step = len(sig) // slices
    if step > 64:
        frames = []
        for i in range(slices):
            chunk = sig[i * step : (i + 1) * step]
            _, chunk_psd = signal.welch(chunk, fs=fs, nperseg=min(len(chunk), 128))
            frames.append(chunk_psd / (np.sum(chunk_psd) + 1e-14))
        diffs = [np.linalg.norm(frames[i] - frames[i - 1]) for i in range(1, slices)]
        feats["spectral_flux"] = float(np.clip(np.mean(diffs), 0.0, 10.0))
    else:
        feats["spectral_flux"] = 0.0

    # Pulse energy entropy
    p = (sig ** 2) / (np.sum(sig ** 2) + 1e-14)
    p = p[p > 0]
    feats["energy_entropy"] = float(np.clip(-np.sum(p * np.log(p + 1e-14)), 0.0, 15.0))

    # Clean any residual numerical overflows
    for k, v in feats.items():
        if np.isnan(v) or np.isinf(v):
            feats[k] = 0.0

    return feats