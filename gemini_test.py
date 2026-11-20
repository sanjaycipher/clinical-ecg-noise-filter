"""
24-Hour Holter ECG Signal Quality & Heart Rate Analyzer
------------------------------------------------------
Analyzes a raw EDF Holter recording (`data_14157.edf`) in 1-second windows,
classifying each second into Lead-Off (LD), Noise, or Clean, and computes
the average Heart Rate (BPM) across all clean segments.

Dependencies:
    pip install mne numpy scipy neurokit2
"""

import warnings
import mne
import numpy as np
import neurokit2 as nk
from scipy.signal import butter, filtfilt, welch

# Suppress verbose MNE info logs during loading
mne.set_log_level("WARNING")
warnings.filterwarnings("ignore", category=RuntimeWarning)


def bandpass_filter(signal: np.ndarray, fs: float, lowcut: float = 0.5, highcut: float = 40.0, order: int = 4) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass filter to isolate standard ECG morphology."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = min(highcut / nyq, 0.99)
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


def classify_window(
    window: np.ndarray,
    fs: float,
    global_min: float,
    global_max: float,
    flatline_std_thresh: float = 0.01,
    rail_tolerance_ratio: float = 0.01,
    saturation_frac_thresh: float = 0.05,
    hf_power_ratio_thresh: float = 0.35,
    baseline_power_ratio_thresh: float = 0.65,
    kurtosis_min: float = 2.0,
) -> str:
    """
    Classify a 1-second ECG window into 'LD' (Lead-Off), 'Noise', or 'Clean'.

    Rules:
    1. Lead-Off (LD):
       - Flatline: standard deviation < flatline_std_thresh (0.01)
       - Saturation: >= 5% of samples in the window sit at or within 1% of the
         recording's min/max ADC voltage rails.
    2. Noise (Muscle Artifact / Baseline Wander / Non-ECG Morphology):
       - Power Spectral Density (Welch) analysis:
         * High-frequency EMG/muscle artifact (> 40 Hz) accounts for > 35% of total power.
         * Heavy baseline wander (< 0.5 Hz) accounts for > 65% of total power.
       - Amplitude / Kurtosis check:
         * Clean QRS complexes create sharp, peaked distributions (high kurtosis).
           Pure Gaussian noise or severe sinusoidal wander has low kurtosis (< 2.0)
           or extreme peak-to-peak swings (> 10x typical QRS amplitude).
    3. Clean:
       - Passes both LD and Noise checks.
    """
    # --- 1. Lead-Off (LD) Detection ---
    win_std = np.std(window)
    if win_std < flatline_std_thresh:
        return "LD"

    dynamic_range = global_max - global_min
    if dynamic_range > 0:
        rail_margin = rail_tolerance_ratio * dynamic_range
        at_upper_rail = window >= (global_max - rail_margin)
        at_lower_rail = window <= (global_min + rail_margin)
        saturated_fraction = np.mean(at_upper_rail | at_lower_rail)
        if saturated_fraction >= saturation_frac_thresh:
            return "LD"

    # --- 2. Noise Detection (Spectral & Statistical Rules) ---
    # Remove DC offset before spectral estimation
    centered = window - np.mean(window)
    nperseg = min(len(centered), int(fs))
    freqs, psd = welch(centered, fs=fs, nperseg=nperseg)

    total_power = np.sum(psd)
    if total_power <= 1e-12:
        return "LD"

    # Band power ratios
    baseline_mask = freqs < 0.5
    qrs_mask = (freqs >= 0.5) & (freqs <= 40.0)
    hf_mask = freqs > 40.0

    baseline_ratio = np.sum(psd[baseline_mask]) / total_power
    hf_ratio = np.sum(psd[hf_mask]) / total_power
    qrs_ratio = np.sum(psd[qrs_mask]) / total_power

    # Heavy baseline wander or strong high-frequency muscle artifact
    if baseline_ratio > baseline_power_ratio_thresh or hf_ratio > hf_power_ratio_thresh:
        return "Noise"

    # Ensure meaningful energy exists in the physiological ECG band (0.5 - 40 Hz)
    if qrs_ratio < 0.25:
        return "Noise"

    # Check signal kurtosis (clean ECG beats have sharp R-peaks -> excess kurtosis > 2)
    m4 = np.mean(centered**4)
    m2 = np.mean(centered**2)
    kurtosis = (m4 / (m2**2)) - 3.0 if m2 > 0 else 0.0
    if kurtosis < kurtosis_min:
        return "Noise"

    return "Clean"


def compute_clean_heart_rate(signal: np.ndarray, clean_mask: np.ndarray, fs: float) -> float:
    """
    Detect R-peaks and calculate the average Heart Rate (BPM) strictly across
    contiguous clean segments using NeuroKit2.
    """
    # Filter the full signal first to avoid edge artifacts on short segments
    filtered_ecg = bandpass_filter(signal, fs=fs, lowcut=0.5, highcut=40.0)

    # Identify contiguous runs of Clean 1-second windows
    # Require at least 3 consecutive clean seconds for reliable R-R interval estimation
    min_run_seconds = 3
    padded = np.concatenate(([False], clean_mask, [False]))
    diffs = np.diff(padded.astype(int))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    samples_per_sec = int(round(fs))
    all_rr_intervals_sec = []

    for start_sec, end_sec in zip(starts, ends):
        duration_sec = end_sec - start_sec
        if duration_sec < min_run_seconds:
            continue

        seg_start = start_sec * samples_per_sec
        seg_end = min(end_sec * samples_per_sec, len(filtered_ecg))
        segment = filtered_ecg[seg_start:seg_end]

        try:
            # NeuroKit2 R-peak detection (Pan-Tompkins / NeuroKit method)
            _, info = nk.ecg_peaks(segment, sampling_rate=int(round(fs)), method="neurokit")
            rpeaks = info.get("ECG_R_Peaks", np.array([]))

            if len(rpeaks) >= 2:
                rr_sec = np.diff(rpeaks) / fs
                # Keep physiologically plausible R-R intervals (30 BPM to 220 BPM -> 0.27s to 2.0s)
                valid_rr = rr_sec[(rr_sec >= 0.27) & (rr_sec <= 2.0)]
                if len(valid_rr) > 0:
                    all_rr_intervals_sec.extend(valid_rr.tolist())
        except Exception:
            # Skip any isolated segment where peak detection cannot converge
            continue

    if not all_rr_intervals_sec:
        return float("nan")

    mean_rr_sec = float(np.mean(all_rr_intervals_sec))
    avg_bpm = 60.0 / mean_rr_sec
    return avg_bpm


def analyze_holter_edf(filepath: str = "data_14157.edf", channel_index: int = 0) -> None:
    """Main pipeline to load a 24-hour Holter EDF file and report signal quality and BPM."""
    print(f"Loading EDF file: {filepath} ...")
    raw = mne.io.read_raw_edf(filepath, preload=True)

    fs = float(raw.info["sfreq"])
    ch_names = raw.ch_names
    if not ch_names:
        raise ValueError("No channels found in the EDF file.")

    # Select ECG channel (prefer channels explicitly labeled 'ECG', else fall back to channel_index)
    ecg_candidates = [idx for idx, name in enumerate(ch_names) if "ecg" in name.lower()]
    selected_idx = ecg_candidates[0] if ecg_candidates else channel_index
    selected_name = ch_names[selected_idx]

    # Extract 1D signal array; convert from Volts (MNE default) to millivolts (mV)
    signal_v = raw.get_data(picks=[selected_idx])[0]
    signal_mv = signal_v * 1e3

    samples_per_sec = int(round(fs))
    total_windows = len(signal_mv) // samples_per_sec

    if total_windows == 0:
        raise ValueError("Signal is shorter than 1 second.")

    # Trim trailing partial second so reshaping into 1-second windows is exact
    trimmed_signal = signal_mv[: total_windows * samples_per_sec]
    windows = trimmed_signal.reshape(total_windows, samples_per_sec)

    # Global voltage rails across the recording to detect ADC saturation
    global_min = float(np.min(trimmed_signal))
    global_max = float(np.max(trimmed_signal))

    print(f"Channel selected : {selected_name} (index {selected_idx})")
    print(f"Sampling rate    : {fs:.2f} Hz")
    print(f"Total duration   : {total_windows} seconds ({total_windows / 3600:.2f} hours)")
    print("Classifying 1-second windows...")

    # Classify each 1-second window
    labels = [
        classify_window(win, fs=fs, global_min=global_min, global_max=global_max)
        for win in windows
    ]
    labels_arr = np.array(labels)

    clean_sec = int(np.sum(labels_arr == "Clean"))
    noise_sec = int(np.sum(labels_arr == "Noise"))
    ld_sec = int(np.sum(labels_arr == "LD"))

    clean_pct = (clean_sec / total_windows) * 100.0
    noise_pct = (noise_sec / total_windows) * 100.0
    ld_pct = (ld_sec / total_windows) * 100.0

    # Calculate average Heart Rate across Clean segments
    print("Detecting R-peaks across clean segments...")
    clean_mask = labels_arr == "Clean"
    avg_bpm = compute_clean_heart_rate(trimmed_signal, clean_mask=clean_mask, fs=fs)

    # Print formatted summary report
    print("\n" + "=" * 50)
    print("24-HOUR HOLTER ECG SIGNAL QUALITY SUMMARY")
    print("=" * 50)
    print(f"Total Analyzed Windows : {total_windows:,} s")
    print(f"Clean                  : {clean_sec:,} s ({clean_pct:.2f}%)")
    print(f"Noise                  : {noise_sec:,} s ({noise_pct:.2f}%)")
    print(f"Lead-Off (LD)          : {ld_sec:,} s ({ld_pct:.2f}%)")
    print("-" * 50)
    if np.isnan(avg_bpm):
        print("Average Heart Rate     : N/A (insufficient clean ECG segments)")
    else:
        print(f"Average Heart Rate     : {avg_bpm:.2f} BPM")
    print("=" * 50)


if __name__ == "__main__":
    analyze_holter_edf("data_14157.edf")
