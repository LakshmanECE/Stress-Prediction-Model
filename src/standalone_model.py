# src/standalone_model.py
"""
Standalone model training pipeline:
- Loads preprocessed EDA/BVP data
- Normalizes signals
- Creates overlapping windows
- Builds CNN-BiLSTM model
- Trains with class weights and evaluates with key metrics
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

def load_signals(subjects, data_dir):
    """Load EDA, BVP, and labels for given subjects."""
    eda_list, bvp_list, label_list = [], [], []
    for subj in subjects:
        eda = pd.read_csv(f"{data_dir}/{subj}_EDA_new.csv").iloc[:,0].values
        bvp = pd.read_csv(f"{data_dir}/{subj}_BVP_new.csv").iloc[:,0].values
        labels = pd.read_csv(f"{data_dir}/{subj}_labels_new.csv").iloc[:,0].values
        eda_list.append(eda)
        bvp_list.append(bvp)
        label_list.append(labels)
    return np.concatenate(eda_list), np.concatenate(bvp_list), np.concatenate(label_list)


def normalize_signals(train_signal, test_signal):
    """Standardize train/test signals separately using training stats."""
    scaler = StandardScaler().fit(train_signal.reshape(-1, 1))
    train_norm = scaler.transform(train_signal.reshape(-1, 1)).flatten()
    test_norm = scaler.transform(test_signal.reshape(-1, 1)).flatten()
    return train_norm, test_norm

def create_windows(eda_norm, bvp_norm, labels_clean,
                   fs_eda=4, fs_bvp=64, window_size_s=60, stride_s=15.0):
    """Create overlapping windows for EDA and BVP with label aggregation (mode)."""
    window_size_eda = int(window_size_s * fs_eda)
    window_size_bvp = int(window_size_s * fs_bvp)
    stride_eda = int(stride_s * fs_eda)

    eda_windows, bvp_windows, window_labels = [], [], []

    for start_eda in range(0, len(eda_norm) - window_size_eda + 1, stride_eda):
        end_eda = start_eda + window_size_eda
        window_label = labels_clean[start_eda:end_eda]
        if np.any(np.isnan(window_label)):
            continue
        win_label = mode(window_label, keepdims=False)[0]
        eda_win = eda_norm[start_eda:end_eda]

        start_bvp = start_eda * (fs_bvp // fs_eda)
        end_bvp = start_bvp + window_size_bvp
        if end_bvp > len(bvp_norm):
            continue
        bvp_win = bvp_norm[start_bvp:end_bvp]

        eda_windows.append(eda_win)
        bvp_windows.append(bvp_win)
        window_labels.append(win_label)

    return np.stack(eda_windows), np.stack(bvp_windows), np.array(window_labels)

def build_standalone_model(input_shape):
    """CNN + BiLSTM standalone model."""
    model = models.Sequential([
        layers.Conv1D(32, 13, activation='relu', padding='same', input_shape=input_shape),
        layers.MaxPooling1D(4),
        layers.Conv1D(64, 13, activation='relu', padding='same'),
        layers.MaxPooling1D(4),
        layers.Bidirectional(layers.LSTM(32, return_sequences=True)),
        layers.Bidirectional(layers.LSTM(16)),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def evaluate_model(model, X_test, y_test, threshold=0.6):
    """Evaluate model and print metrics."""
    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs >= threshold).astype(int).flatten()

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_probs)
    }

    print("\nEvaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k.capitalize()}: {v:.4f}")

    return y_pred_probs, metrics


def plot_precision_recall(y_test, y_pred_probs):
    """Plot Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_test, y_pred_probs)
    ap = average_precision_score(y_test, y_pred_probs)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR Curve (AP={ap:.3f})', color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - Standalone Model')
    plt.legend()
    plt.grid(True)
    plt.show()
