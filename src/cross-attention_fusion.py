import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import mode
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, Model, Input

# 1. Load and preprocess data
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

train_subjects = ['S2','S3','S4','S5','S6','S7','S8','S9']
test_subjects  = ['S10','S11','S13','S14','S15','S16','S17']
data_dir = "/content/drive/MyDrive/Filtered_Signals"

# Load data
train_eda, train_bvp, train_labels = load_signals(train_subjects, data_dir)
test_eda, test_bvp, test_labels = load_signals(test_subjects, data_dir)

# Normalize data using train statistics
eda_scaler = StandardScaler().fit(train_eda.reshape(-1, 1))
bvp_scaler = StandardScaler().fit(train_bvp.reshape(-1, 1))

train_eda_norm = eda_scaler.transform(train_eda.reshape(-1, 1)).flatten()
test_eda_norm  = eda_scaler.transform(test_eda.reshape(-1, 1)).flatten()
train_bvp_norm = bvp_scaler.transform(train_bvp.reshape(-1, 1)).flatten()
test_bvp_norm  = bvp_scaler.transform(test_bvp.reshape(-1, 1)).flatten()

# 2. Windowing function
def create_windows(eda_norm, bvp_norm, labels_clean,
                   fs_eda=4, fs_bvp=64, window_size_s=60, stride_s=15.0):
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

    return (np.stack(eda_windows), np.stack(bvp_windows), np.array(window_labels))

# Generate train/test windows
train_eda_win, train_bvp_win, train_win_labels = create_windows(train_eda_norm, train_bvp_norm, train_labels)
test_eda_win, test_bvp_win, test_win_labels   = create_windows(test_eda_norm, test_bvp_norm, test_labels)

# Convert labels to binary (stress vs non-stress)
train_bin_labels = np.where(train_win_labels == 2, 1, 0)
test_bin_labels  = np.where(test_win_labels == 2, 1, 0)

# Reshape for model
X_bvp_train = train_bvp_win[..., np.newaxis]
X_bvp_test  = test_bvp_win[..., np.newaxis]
X_eda_train = train_eda_win[..., np.newaxis]
X_eda_test  = test_eda_win[..., np.newaxis]
y_train = train_bin_labels
y_test  = test_bin_labels

# Train/validation split
X_bvp_tr, X_bvp_val, X_eda_tr, X_eda_val, y_tr, y_val = train_test_split(
    X_bvp_train, X_eda_train, y_train,
    test_size=0.15,
    random_state=42,
    stratify=y_train
)

# 3. Build hybrid cross-attention fusion model
def build_hybrid_cross_attention_fusion(bvp_shape=(3840, 1), eda_shape=(240, 1), emb_dim=32):
    # BVP Branch
    bvp_input = Input(shape=bvp_shape, name='bvp_input')
    x_bvp = layers.Conv1D(32, 13, activation='relu', padding='same')(bvp_input)
    x_bvp = layers.MaxPooling1D(4)(x_bvp)
    x_bvp = layers.Conv1D(64, 13, activation='relu', padding='same')(x_bvp)
    x_bvp = layers.MaxPooling1D(4)(x_bvp)
    x_bvp = layers.Bidirectional(layers.LSTM(emb_dim, return_sequences=True))(x_bvp)
    bvp_seq = layers.TimeDistributed(layers.Dense(emb_dim))(x_bvp)

    # EDA Branch
    eda_input = Input(shape=eda_shape, name='eda_input')
    x_eda = layers.Conv1D(16, 5, activation='relu', padding='same')(eda_input)
    x_eda = layers.MaxPooling1D(2)(x_eda)
    x_eda = layers.Conv1D(32, 5, activation='relu', padding='same')(x_eda)
    x_eda = layers.MaxPooling1D(2)(x_eda)
    x_eda = layers.Bidirectional(layers.LSTM(emb_dim, return_sequences=True))(x_eda)
    eda_seq = layers.TimeDistributed(layers.Dense(emb_dim))(x_eda)

    # Cross-Attention (BVP queries, EDA keys/values)
    attention = layers.Attention()([bvp_seq, eda_seq])

    # Pooling and fusion
    attn_pooled = layers.GlobalAveragePooling1D()(attention)
    bvp_skip    = layers.GlobalAveragePooling1D()(bvp_seq)
    fused       = layers.concatenate([attn_pooled, bvp_skip])

    # Classification head
    x = layers.Dense(32, activation='relu')(fused)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(1, activation='sigmoid')(x)

    return Model(inputs=[bvp_input, eda_input], outputs=output)

model = build_hybrid_cross_attention_fusion()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training with early stopping
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
history = model.fit(
    {'bvp_input': X_bvp_tr, 'eda_input': X_eda_tr}, y_tr,
    validation_data=({'bvp_input': X_bvp_val, 'eda_input': X_eda_val}, y_val),
    epochs=20,
    batch_size=128,
    callbacks=[callback]
)

# Evaluation and metrics
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, precision_recall_curve,
    average_precision_score
)

# Predictions
y_pred_probs = model.predict({'bvp_input': X_bvp_test, 'eda_input': X_eda_test})
y_pred = (y_pred_probs >= 0.5).astype(int).flatten()

# Metrics calculation
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
accuracy     = accuracy_score(y_test, y_pred)
precision    = precision_score(y_test, y_pred)
recall       = recall_score(y_test, y_pred)
f1           = f1_score(y_test, y_pred)
specificity  = tn / (tn + fp) if (tn + fp) > 0 else 0
roc_auc      = roc_auc_score(y_test, y_pred_probs)

print(f"Accuracy     : {accuracy:.4f}")
print(f"Precision    : {precision:.4f}")
print(f"Recall (Sens): {recall:.4f}")
print(f"Specificity  : {specificity:.4f}")
print(f"F1 Score     : {f1:.4f}")
print(f"ROC-AUC      : {roc_auc:.4f}")

# Precision-Recall curve
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_probs)
ap = average_precision_score(y_test, y_pred_probs)

plt.figure(figsize=(8,6))
plt.plot(recall_curve, precision_curve, label=f'AP={ap:.3f}', color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Cross-Attention Fusion)')
plt.legend()
plt.grid(True)
plt.show()
