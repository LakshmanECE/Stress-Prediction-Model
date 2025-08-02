import os
import pickle
import numpy as np
import pandas as pd
from scipy.stats import mode
from scipy.signal import butter, filtfilt

def butter_lowpass_filter(data, cutoff, fs, order=4):
    """Apply low-pass Butterworth filter to data."""
    b, a = butter(order, cutoff/(0.5*fs), btype='low')
    return filtfilt(b, a, data)


def winsorize_signal(data, lower_pct=2, upper_pct=98):
    """Clip signal extremes between percentiles."""
    lower = np.percentile(data, lower_pct)
    upper = np.percentile(data, upper_pct)
    data_clipped = np.copy(data)
    data_clipped[data < lower] = lower
    data_clipped[data > upper] = upper
    return data_clipped


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Apply bandpass Butterworth filter to data."""
    b, a = butter(order, [lowcut/(0.5*fs), highcut/(0.5*fs)], btype='band')
    return filtfilt(b, a, data)

def process_subject(pkl_path, output_dir, ignore_labels=None):
    """
    Process a single subject's data file:
    - Load pickle
    - Extract wrist EDA, BVP, labels
    - Downsample and clean labels
    - Filter EDA and BVP signals
    - Save processed CSVs
    """
    if ignore_labels is None:
        ignore_labels = [0, 5, 6, 7]

    # Load pickle
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    # Extract wrist signals
    bvp = data['signal']['wrist']['BVP'].flatten()
    eda = data['signal']['wrist']['EDA'].flatten()
    labels = data['label'].flatten()

    # Downsample labels from 700Hz → 4Hz (mode)
    LABEL_ORIG_HZ = 700
    TARGET_HZ = 4
    factor = LABEL_ORIG_HZ // TARGET_HZ
    n_new = len(labels) // factor
    downsampled_labels = np.zeros(n_new, dtype=int)
    for i in range(n_new):
        seg = labels[i*factor:(i+1)*factor]
        downsampled_labels[i] = mode(seg, keepdims=False)[0]

    # Clean labels (set ignored to NaN)
    labels_clean = downsampled_labels.astype(float)
    for val in ignore_labels:
        labels_clean[labels_clean == val] = np.nan

    # Filter EDA (low-pass 0.5 Hz @ 4 Hz)
    eda_filtered = butter_lowpass_filter(eda, cutoff=0.5, fs=4, order=4)

    # Filter BVP (winsorize 2–98%, bandpass 0.5–8 Hz @ 64 Hz)
    bvp_wins = winsorize_signal(bvp, 2, 98)
    bvp_filtered = butter_bandpass_filter(bvp_wins, lowcut=0.5, highcut=8, fs=64, order=4)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save to CSV (EDA, BVP, Labels separately)
    subject_id = os.path.splitext(os.path.basename(pkl_path))[0]
    pd.DataFrame(eda_filtered).to_csv(os.path.join(output_dir, f"{subject_id}_EDA_new.csv"), index=False)
    pd.DataFrame(bvp_filtered).to_csv(os.path.join(output_dir, f"{subject_id}_BVP_new.csv"), index=False)
    pd.DataFrame(labels_clean).to_csv(os.path.join(output_dir, f"{subject_id}_labels_new.csv"), index=False)

    print(f"Saved: {subject_id}_EDA_new.csv, {subject_id}_BVP_new.csv, {subject_id}_labels_new.csv")
    print(f"{subject_id}: EDA {len(eda_filtered)}, BVP {len(bvp_filtered)}, Labels {len(labels_clean)}")


def process_all_subjects(input_dir, output_dir, subjects, ignore_labels=None):
    """Loop through all subjects and process each one."""
    for subj in subjects:
        pkl_path = os.path.join(input_dir, f"{subj}.pkl")
        print(f"\nProcessing {subj} ...")
        process_subject(pkl_path, output_dir, ignore_labels)
