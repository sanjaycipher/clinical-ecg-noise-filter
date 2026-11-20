import os
import mne
import wfdb

# The actual 7 records in PhysioNet 'ltdb' are:
# ['14046', '14134', '14149', '14157', '14172', '14184', '15814']
# 14134, 14149, and 14184 are already downloaded and converted.
# We download the remaining 4 records:
remaining_records = ['14046', '14157', '14172', '15814']
data_dir = 'mit-bih-long-term-ecg-database-1.0.0'

os.makedirs(data_dir, exist_ok=True)

for rec_name in remaining_records:
    try:
        print(f"\n--- Processing Record {rec_name} ---")
        print(f"Downloading record {rec_name} from ltdb...")
        wfdb.dl_database('ltdb', dl_dir=data_dir, records=[rec_name])
        
        rec_path = os.path.join(data_dir, rec_name)
        print(f"Reading WFDB record from {rec_path}...")
        record = wfdb.rdrecord(rec_path)
        signal = record.p_signal[:, 0] * 1e-3
        info = mne.create_info(ch_names=['ECG'], sfreq=record.fs, ch_types='ecg')
        raw = mne.io.RawArray(signal.reshape(1, -1), info, verbose='ERROR')
        
        edf_filename = f"data_{rec_name}.edf"
        print(f"Exporting to {edf_filename}...")
        raw.export(edf_filename, fmt='edf', overwrite=True)
        print(f"Success! Created {edf_filename}")
    except Exception as e:
        print(f"Failed on {rec_name}: {e}")

print("\nAll ltdb database records downloaded and converted to EDF!")
