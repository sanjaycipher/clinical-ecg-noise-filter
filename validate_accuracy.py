import sys
import mne
import numpy as np
import neurokit2 as nk
import pandas as pd
from beatly_pipeline import load_edf, classify_signal_windows

# 1. Select input file (supports CLI argument or defaults to data.edf)
filename = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else 'data.edf'
print(f"Loading {filename}...")
signal, fs = load_edf(filename)

# 2. Run YOUR pipeline's classification
print("Running your Triage Layer...")
windows = classify_signal_windows(signal, fs)

# 3. Find a window where YOUR pipeline says "LD" or "Noise"
target_window = None
# Check initial windows first
for w in windows[:100]:
    if w['label'] in ['LD', 'Noise']:
        target_window = w
        break

# If not in the first 100 seconds, scan the full recording
if target_window is None:
    for w in windows:
        if w['label'] in ['LD', 'Noise']:
            target_window = w
            break

# If data.edf was 100% clean, seamlessly test against data_14149.edf (known LD event)
if target_window is None:
    print(f"\n[INFO] No LD/Noise found in {filename} (this recording was 100% clean/bradycardic).")
    fallback_file = 'data_14149.edf'
    print(f"Switching to '{fallback_file}' to evaluate an actual Lead-Off (LD) event...")
    filename = fallback_file
    signal, fs = load_edf(filename)
    windows = classify_signal_windows(signal, fs)
    target_window = next((w for w in windows if w['label'] in ['LD', 'Noise']), None)

if target_window is None:
    print("No LD/Noise found across evaluated recordings. Try a different file.")
    sys.exit(0)

# 4. Extract the raw signal for that window
start_sec = target_window['start_sec']
end_sec = target_window['end_sec']
start_idx = int(start_sec * fs)
end_idx = int(end_sec * fs)
raw_segment = signal[start_idx:end_idx]

print(f"\n--- ACCURACY TEST ---")
print(f"Target Recording: {filename}")
print(f"Your Pipeline Label: {target_window['label']}")
print(f"Window Time: {start_sec}s to {end_sec}s")

# 5. Ask NeuroKit2 to evaluate the SAME raw segment
# We use the 'zhao2018' method as the gold standard
try:
    cleaned_segment = nk.ecg_clean(raw_segment, sampling_rate=fs)
    nk_quality = nk.ecg_quality(cleaned_segment, sampling_rate=fs, method="zhao2018", approach="fuzzy")
    print(f"NeuroKit2 Gold Standard Quality: {nk_quality}")
except Exception as e:
    print(f"NeuroKit2 could not evaluate this segment (likely too short/noisy): {e}")

# 6. Test a "Clean" window for comparison
clean_window = next((w for w in windows[:100] if w['label'] == 'Clean'), None)
if clean_window:
    c_start = int(clean_window['start_sec'] * fs)
    c_end = int(clean_window['end_sec'] * fs)
    c_segment = signal[c_start:c_end]
    c_cleaned = nk.ecg_clean(c_segment, sampling_rate=fs)
    c_quality = nk.ecg_quality(c_cleaned, sampling_rate=fs, method="zhao2018", approach="fuzzy")
    print(f"\nFor comparison, a 'Clean' window at {clean_window['start_sec']}s has NK Quality: {c_quality}")

# 7. Clinical context evaluation (10-second standard epoch)
print(f"\n--- CLINICAL CONTEXT EVALUATION (10-Second Epoch) ---")
try:
    ctx_start = max(0, int(start_idx))
    ctx_end = min(len(signal), ctx_start + int(10 * fs))
    ctx_seg = signal[ctx_start:ctx_end]
    ctx_cleaned = nk.ecg_clean(ctx_seg, sampling_rate=fs)
    ctx_q = nk.ecg_quality(ctx_cleaned, sampling_rate=fs, method="zhao2018", approach="fuzzy")
    print(f"10-Second Event Window ({target_window['label']}) NK Quality : {ctx_q}")
except Exception as e:
    print(f"Context evaluation note: {e}")

if clean_window:
    try:
        cl_start = int(clean_window['start_sec'] * fs)
        cl_end = min(len(signal), cl_start + int(10 * fs))
        cl_seg = signal[cl_start:cl_end]
        cl_cleaned = nk.ecg_clean(cl_seg, sampling_rate=fs)
        cl_q = nk.ecg_quality(cl_cleaned, sampling_rate=fs, method="zhao2018", approach="fuzzy")
        print(f"10-Second Clean Window (Clean Rhythm) NK Quality : {cl_q}")
    except Exception as e:
        print(f"Clean context evaluation note: {e}")
