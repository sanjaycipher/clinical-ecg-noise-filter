import os
import mne
import wfdb

# Path to the extracted folder
DATA_DIR = "mit-bih-long-term-ecg-database-1.0.0"
RECORD_NAME = "14134"  # You can change this to any of the 7 records (e.g., 14134, 14149, etc.)

record_path = os.path.join(DATA_DIR, RECORD_NAME)

print(f"Loading WFDB record from {record_path}...")
record = wfdb.rdrecord(record_path)

# Extract the first channel (usually ECG lead)
# PhysioNet is in mV, MNE needs Volts, so multiply by 1e-3
signal = record.p_signal[:, 0] * 1e-3
fs = record.fs

print(f"Sampling Rate: {fs} Hz, Duration: {len(signal)/fs/3600:.2f} hours")

# Create MNE Raw object
info = mne.create_info(ch_names=['ECG'], sfreq=fs, ch_types='ecg')
raw = mne.io.RawArray(signal.reshape(1, -1), info)

print("Exporting to data.edf...")
raw.export('data.edf', fmt='edf', overwrite=True)
print("Success! data.edf is ready in P:\\ECG Event Detection Pipeline")
