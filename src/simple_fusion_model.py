import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import mode
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt


def load_signals(subjects, data_dir):
    eda_list, bvp_list, label_list = [], [], []
    for subj in subjects:
        eda = pd.read_csv(f"{data_dir}/{subj}_EDA_new.csv").iloc[:,0].values
        bvp = pd.read_csv(f"{data_dir}/{subj}_BVP_new.csv").iloc[:,0].values
        labels = pd.read_csv(f"{data_dir}/{subj}_labels_new.csv").iloc[:,0].values
        eda_list.append(eda)
        bvp_list.append(bvp)
        label_list.append(labels)
    return np.concatenate(eda_list), np.concatenate(bvp_list), np.concatenate(label_list)

# Windowing Function
def create_windows(eda_norm, bvp_norm, labels_clean, fs_eda=4, fs_bvp=64, window_size_s=60, stride_s=15.0):
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

# Model Definition
def build_simple_fusion_model():
    # BVP branch
    bvp_input = Input(shape=(3840, 1), name='bvp_input')
    x_bvp = layers.Conv1D(32, 13, activation='relu', padding='same')(bvp_input)
    x_bvp = layers.MaxPooling1D(4)(x_bvp)
    x_bvp = layers.Conv1D(64, 13, activation='relu', padding='same')(x_bvp)
    x_bvp = layers.MaxPooling1D(4)(x_bvp)
    x_bvp = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(x_bvp)
    x_bvp = layers.Bidirectional(layers.LSTM(16))(x_bvp)
    bvp_features = layers.Dense(32, activation='relu')(x_bvp)

    # EDA branch
    eda_input = Input(shape=(240, 1), name='eda_input')
    x_eda = layers.Conv1D(16, 5, activation='relu', padding='same')(eda_input)
    x_eda = layers.MaxPooling1D(2)(x_eda)
    x_eda = layers.Conv1D(32, 5, activation='relu', padding='same')(x_eda)
    x_eda = layers.MaxPooling1D(2)(x_eda)
    x_eda = layers.Bidirectional(layers.LSTM(16, return_sequences=True))(x_eda)
    x_eda = layers.Bidirectional(layers.LSTM(8))(x_eda)
    eda_features = layers.Dense(16, activation='relu')(x_eda)

    # Fusion
    fusion = layers.concatenate([bvp_features, eda_features])
    fusion = layers.Dense(32, activation='relu')(fusion)
    fusion = layers.Dropout(0.5)(fusion)
    output = layers.Dense(1, activation='sigmoid')(fusion)

    model = Model(inputs=[bvp_input, eda_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Evaluation Function
def evaluate_model(model, X_bvp_test, X_eda_test, y_test):
    y_pred_probs = model.predict({'bvp_input': X_bvp_test, 'eda_input': X_eda_test})
    y_pred = (y_pred_probs >= 0.5).astype(int).flatten()

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

    for k, v in metrics.items():
        print(f"{k.capitalize():<12}: {v:.4f}")

    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_probs)
    ap = average_precision_score(y_test, y_pred_probs)

    plt.figure(figsize=(8,6))
    plt.plot(recall_vals, precision_vals, label=f'PR curve (AP={ap:.3f})', color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Simple Fusion)')
    plt.legend()
    plt.grid(True)
    plt.show()
