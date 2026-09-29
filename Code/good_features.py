import numpy as np
from scipy import signal
from scipy.stats import skew, kurtosis

def extract_window_features(sig_window, fs=1000.0, noise_floor_rms=1e-11):
    """
    Extracts dynamics, periodicity, and spectral features bounded strictly
    against overflow, referencing the physical baseline noise floor.
    """
    feats = {}
    sig = np.nan_to_num(sig_window, nan=0.0, posinf=0.0, neginf=0.0)

    # 1. Amplitude Dynamics & Physical Signal-to-Noise Ratio (SNR)
    abs_sig = np.abs(sig)
    rms = np.sqrt(np.mean(sig ** 2))
    feats['rms'] = float(rms)
    feats['max_amplitude'] = float(np.max(abs_sig))
    feats['crest_factor'] = float(np.clip(feats['max_amplitude'] / (rms + 1e-12), 1.0, 50.0))
    
    # SNR relative to calibrated transducer quiet state
    snr_val = 20.0 * np.log10((rms + 1e-12) / (noise_floor_rms + 1e-12))
    feats['snr_db'] = float(np.clip(snr_val, -20.0, 80.0))

    # 2. Distribution Shape
    feats['skewness'] = float(np.clip(skew(sig), -10.0, 10.0))
    feats['kurtosis'] = float(np.clip(kurtosis(sig), -10.0, 30.0))

    # Zero-Crossing Rate
    zero_crossings = np.nonzero(np.diff(sig > 0))[0]
    feats['zcr'] = float(len(zero_crossings) / max(len(sig), 1))

    # 3. Peak Periodicity Anchored by 3-Sigma Noise Floor
    detection_threshold = max(3.0 * noise_floor_rms, np.std(sig) * 0.4)
    peaks, _ = signal.find_peaks(sig, distance=int(0.25 * fs), prominence=detection_threshold)
    if len(peaks) > 2:
        ibi = np.diff(peaks) / fs
        mean_ibi = np.mean(ibi)
        feats['bpm'] = float(np.clip(60.0 / mean_ibi, 30.0, 240.0)) if mean_ibi > 0 else 0.0
        feats['tempo_variation_cv'] = float(np.clip(np.std(ibi) / (mean_ibi + 1e-9), 0.0, 5.0))
    else:
        feats['bpm'] = 0.0
        feats['tempo_variation_cv'] = 0.0

    # 4. Spectral Distribution (Welch PSD)
    freqs, psd = signal.welch(sig, fs=fs, nperseg=min(len(sig), int(fs)))
    psd = np.nan_to_num(psd, nan=0.0, posinf=0.0, neginf=0.0)
    total_power = np.sum(psd)
    feats['log_total_power'] = float(np.log10(total_power + 1e-14))
    
    denom = total_power + 1e-14
    p_sub_bass = np.sum(psd[(freqs >= 5) & (freqs < 35)]) / denom
    p_bass     = np.sum(psd[(freqs >= 35) & (freqs < 100)]) / denom
    p_mid      = np.sum(psd[(freqs >= 100) & (freqs < 250)]) / denom
    p_high     = np.sum(psd[(freqs >= 250) & (freqs <= 450)]) / denom

    feats['band_sub_bass'] = float(np.clip(p_sub_bass, 0.0, 1.0))
    feats['band_bass']     = float(np.clip(p_bass, 0.0, 1.0))
    feats['band_mid']      = float(np.clip(p_mid, 0.0, 1.0))
    feats['band_high']     = float(np.clip(p_high, 0.0, 1.0))

    # Log Ratios (Safe against scaling explosion)
    feats['log_sub_to_mid'] = float(np.log10((p_sub_bass + 1e-5) / (p_mid + 1e-5)))
    feats['log_bass_to_high'] = float(np.log10((p_bass + 1e-5) / (p_high + 1e-5)))

    centroid = np.sum(freqs * psd) / denom
    feats['spectral_centroid'] = float(np.clip(centroid, 0.0, fs / 2))

    # 5. Spectral Flux
    n_slices = 8
    slice_len = len(sig) // n_slices
    if slice_len > 64:
        sub_psds = []
        for i in range(n_slices):
            sub_sig = sig[i * slice_len : (i + 1) * slice_len]
            _, s_psd = signal.welch(sub_sig, fs=fs, nperseg=min(len(sub_sig), 128))
            s_sum = np.sum(s_psd) + 1e-14
            sub_psds.append(s_psd / s_sum)
        diffs = [np.linalg.norm(sub_psds[i] - sub_psds[i - 1]) for i in range(1, n_slices)]
        feats['spectral_flux'] = float(np.clip(np.mean(diffs), 0.0, 10.0))
    else:
        feats['spectral_flux'] = 0.0

    # 6. Pulse Energy Entropy
    energy_dist = (sig ** 2) / (np.sum(sig ** 2) + 1e-14)
    entropy = -np.sum(energy_dist * np.log(energy_dist + 1e-14))
    feats['energy_entropy'] = float(np.clip(entropy, 0.0, 15.0))

    return {k: float(0.0 if np.isnan(v) or np.isinf(v) else v) for k, v in feats.items()}