import os
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt
from beatly_pipeline import load_edf


def filter_10s_segment(signal: np.ndarray, fs: float, start_sec: int, pad_sec: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts a 10-second ECG window with padding, applies a 0.5 Hz high-pass Butterworth
    filter to remove baseline wander, and returns the time axis (0-10s) and filtered signal.
    """
    total_seconds = len(signal) / fs
    pad_start = max(0.0, start_sec - pad_sec)
    pad_end = min(total_seconds, start_sec + 10.0 + pad_sec)

    s_idx = int(round(pad_start * fs))
    e_idx = int(round(pad_end * fs))
    pad_signal = signal[s_idx:e_idx]

    # 0.5 Hz 2nd-order high-pass Butterworth filter
    b_hp, a_hp = butter(2, 0.5, btype="highpass", fs=fs)

    try:
        filtered = filtfilt(b_hp, a_hp, pad_signal)
    except Exception:
        filtered = pad_signal - np.nanmedian(pad_signal)

    # Slice exact 10 seconds
    rel_start = int(round((start_sec - pad_start) * fs))
    exact_samples = int(round(10.0 * fs))
    seg = filtered[rel_start : rel_start + exact_samples]
    seg = seg - np.nanmedian(seg)

    t_axis = np.linspace(0, 10, len(seg), endpoint=False)
    return t_axis, seg


def main():
    print("=" * 70)
    print(" Generating Visual Accuracy Proof: Clean vs. Noise vs. Lead-Off")
    print("=" * 70)

    # 1. Target Event Specifications
    test_cases = [
        {
            "category": "Clean",
            "file": "data.edf",
            "start_sec": 0,
            "title": "Clean: Diagnostic Quality (NeuroKit2: Excellent)",
            "color": "#2e7d32",
            "bg_color": "green",
            "bg_alpha": 0.08,
        },
        {
            "category": "Noise",
            "file": "data_14172.edf",
            "start_sec": 4632,
            "title": "Noise: Muscle Artifact (NeuroKit2: Excellent, Pipeline: Noise)",
            "color": "#e65100",
            "bg_color": "orange",
            "bg_alpha": 0.12,
        },
        {
            "category": "Lead-Off",
            "file": "data_14149.edf",
            "start_sec": 83504,
            "title": "Lead-Off: Sensor Disconnected (NeuroKit2: Barely acceptable)",
            "color": "#c62828",
            "bg_color": "red",
            "bg_alpha": 0.12,
        },
    ]

    # Create figure with 3 stacked subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    for idx, (tc, ax) in enumerate(zip(test_cases, axes)):
        print(f"\n[{idx+1}/3] Loading '{tc['file']}' for {tc['category']} event at {tc['start_sec']}s...")
        signal, fs = load_edf(tc["file"])

        # Filter and extract exact 10-second segment
        t_axis, segment = filter_10s_segment(signal, fs, tc["start_sec"])

        # Plot waveform
        ax.plot(t_axis, segment, color="#222222", linewidth=0.9, zorder=3)

        # Background tinting for clear visual distinction
        ax.axvspan(0, 10, color=tc["bg_color"], alpha=tc["bg_alpha"], zorder=1)

        # Standard clinical limits and styling
        ax.set_xlim(0, 10)
        ax.set_ylim(-1.5, 1.5)

        # ECG Grid Paper (major and minor)
        ax.grid(True, which="major", linestyle="-", alpha=0.35, color="#e0a0a0")
        ax.minorticks_on()
        ax.grid(True, which="minor", linestyle=":", alpha=0.2, color="#e8c0c0")

        # Labels & Titles
        ax.set_ylabel("Amplitude (mV)", fontsize=10, fontweight="bold")
        ax.set_title(tc["title"], fontsize=12, fontweight="bold", color=tc["color"], pad=8)

    # Bottom x-axis label
    axes[-1].set_xlabel("Time (seconds)", fontsize=11, fontweight="bold")

    plt.suptitle("Clinical ECG Triage Layer - Ground-Truth Visual Proof", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_png = "Accuracy_Visual_Proof.png"
    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\n" + "=" * 70)
    print(f"[SUCCESS] Figure saved as '{output_png}'")
    print("=" * 70)


if __name__ == "__main__":
    from typing import Tuple
    main()
