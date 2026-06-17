import os
import glob
import numpy as np
import mne
from CORSICA.artifact_reduction import ci_artifact_reduction

# Ordner mit allen EEGLAB-Dateien
data_dir = "/Users/leonierichter/Documents/2100_Work/2026_Uni_Chair_Senory_in_Neuroengineering/2026.03.27_Code_und_Daten/simple_prepro/102"

# Alle .set Dateien finden
eeg_files = sorted(glob.glob(os.path.join(data_dir, "*.set")))

# Parameter
snr_threshold_my = 20
fs_eeg = 1000

# Output-Ordner
output_dir = "/Users/leonierichter/Documents/2100_Work/2026_Uni_Chair_Senory_in_Neuroengineering"

for eeg_path in eeg_files:

    print(f"\nVerarbeite: {os.path.basename(eeg_path)}")

    # EEG laden
    raw = mne.io.read_raw_eeglab(eeg_path, preload=True)

    # Audio-Kanal (Kanal 32 -> Index 31)
    attended_audio = raw.get_data()[31, :]
    print(attended_audio)

    # Dateiname ohne Endung als Trial-ID
    trial_id = os.path.splitext(os.path.basename(eeg_path))[0]

    # CORSICA
    cleaned_eeg = ci_artifact_reduction(
        raw,
        "104",                 # Subject-ID
        trial_id,              # Trial-ID
        output_dir,
        fs_eeg,
        attended_audio,
        snr_threshold=snr_threshold_my,
        plot=True,
        metadata=True
    )

    print(f"Fertig: {trial_id}")