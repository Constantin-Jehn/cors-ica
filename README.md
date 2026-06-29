**CI Artifact Reduction**

```python
ci_artifact_reduction(
    raw,
    subject_id,
    trial_id,
    output_dir,
    fs_eeg,
    attended_audio,
    distraction_audio=None,
    snr_threshold=27,
    start_search_window=-0.005,
    end_search_window=0.012,
    n_components=None,
    plot=False,
    metadata=False
)
```

Filtering CI Artifacts based on Correlation of Independent Components and audio 

### Method Overview

1. EEG data is decomposed into independent components using ICA.
2. Each component is cross-correlated with the audio stimulus.
3. The peak correlation within a search window is identified.
4. The SNR value is computed from the correlation peak.
5. Components with an SNR value above the defined threshold are classified as CI artifact-related components and are removed.
6. The cleaned EEG signal is reconstructed from the remaining components.

--- 

### Parameters

**raw** : mne.io.Raw  
&nbsp;&nbsp;&nbsp;&nbsp;EEG recording loaded with MNE-Python

**subject_id** : str | int  
&nbsp;&nbsp;&nbsp;&nbsp;Identifier of the subject used for output file naming

**trial_id** : str | int  
&nbsp;&nbsp;&nbsp;&nbsp;Identifier of the trial used for output file naming

**output_dir** : str  
&nbsp;&nbsp;&nbsp;&nbsp;Directory where plots and metadata files are saved

**fs_eeg** : int  
&nbsp;&nbsp;&nbsp;&nbsp;Sampling frequency of the EEG recording in Hz

**attended_audio** : np.ndarray  
&nbsp;&nbsp;&nbsp;&nbsp;1D array containing the attended audio signal

**distraction_audio** : np.ndarray, optional  
&nbsp;&nbsp;&nbsp;&nbsp;1D array containing the distracting audio signal, default = None

**snr_threshold** : float  
&nbsp;&nbsp;&nbsp;&nbsp;Independent components with SNR values above this threshold are removed, default = 27

**start_search_window** : float, optional  
&nbsp;&nbsp;&nbsp;&nbsp;Start point of the search window relative to zero lag (in seconds), default = -0.005

**end_search_window** : float, optional  
&nbsp;&nbsp;&nbsp;&nbsp;End point of the search window relative to zero lag (in seconds), default = 0.012

**n_components** : int | float | None, optional 
&nbsp;&nbsp;&nbsp;&nbsp;Number of components passed to the ICA algorithm. If None, 0.999999 will be used, default = None.

**plot** : bool, optional  
&nbsp;&nbsp;&nbsp;&nbsp;If True, saves cross-correlation plots of all independent components, default = False

**metadata** : bool, optional  
&nbsp;&nbsp;&nbsp;&nbsp;If True, saves metadata and summary statistics as CSV files, default = False

---    

### Returns

**cleaned_eeg** : np.ndarray  
&nbsp;&nbsp;&nbsp;&nbsp;EEG data after CI artifact removal

**raw_cleaned** : mne.io.Raw  
&nbsp;&nbsp;&nbsp;&nbsp;Cleaned EEG data as an MNE Raw object

**metadata_out** : dict
&nbsp;&nbsp;&nbsp;&nbsp;Dictionary containing summary information about the artifact reduction, including all SNR values, excluded ICs, used ICs, peak times, and threshold settings

---

### Raises

**ValueError**  
&nbsp;&nbsp;&nbsp;&nbsp;If `output_dir` is not specified.

**ValueError**  
&nbsp;&nbsp;&nbsp;&nbsp;If EEG and audio dimensions do not match.

---

### Warns

**UserWarning**  
&nbsp;&nbsp;&nbsp;&nbsp;If EEG sampling frequency is below 500 Hz.

**UserWarning**
&nbsp;&nbsp;&nbsp;&nbsp;If a Trial ID already exists in the metadata file and will be overwritten.

**UserWarning**
&nbsp;&nbsp;&nbsp;&nbsp;If Trial ID contains non-numeric values and sorting is skipped.

---

## Installation
This package requires:

- NumPy
- Pandas
- SciPy
- Matplotlib
- MNE-Python

All dependencies are installed automatically when installing the package.

Install the latest release using pip:
```bash
pip install \
  -i https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple \
  cors-ica
```

<!-- pip install cors-ica -->

---

### Notes
- EEG and audio must already be temporally aligned and have identical sampling and length. No resampling is performed internally.
- ICA is performed using MNE’s Infomax algorithm.
- In dual-speaker scenarios, attended and distraction signals are summed to form the reference stimulus.

---

## Dataset

The dataset used for development and evaluation of this method is available on Zenodo:

[CI Artifact Reduction Dataset](https://zenodo.org/records/17952844)

---

### Example

```python

import numpy as np
import mne
from CORSICA.artifact_reduction import ci_artifact_reduction

# Load EEG data (EEGLAB .set file)
raw = mne.io.read_raw_eeglab("subject_01_trial_01.set", preload=True)

# Sampling frequency
fs_eeg = 1000

# Ensure matching length with EEG
n_samples = raw.get_data().shape[1]

# Example audio signals (must be time-aligned with EEG)
attended_audio = raw.get_data()[31, :]
distraction_audio = raw.get_data()[32, :]

           

# Run CI artifact reduction

cleaned_eeg, raw_cleaned, metadata_out  = ci_artifact_reduction(
    raw=raw,
    subject_id="301",
    trial_id="123",
    output_dir="./results",
    fs_eeg=fs_eeg,
    attended_audio=attended_audio,
    distraction_audio=distraction_audio,
    snr_threshold=9.5,
    start_search_window=-0.005,
    end_search_window=0.012,
    raw_cleaned, metadata_out
    plot=True,
    metadata=True
)

print("Cleaned EEG shape:", cleaned_eeg.shape)
print("Excluded ICs:", metadata_out["Indices of excluded ICs"]) 
print("Used ICs:", metadata_out["Indices of used ICs"])
```


