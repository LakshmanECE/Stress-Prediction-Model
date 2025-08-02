# Stress Detection with Multi-Modal Fusion (EDA + BVP)

This project implements stress detection using physiological signals from the WESAD dataset, leveraging Electrodermal Activity (EDA) and Blood Volume Pulse (BVP). It explores three deep learning architectures:

1. **Standalone Model** – Single-modality model using BVP signals.
2. **Simple Fusion Model** – Parallel feature extraction from EDA and BVP, followed by late fusion.
3. **Cross-Attention Fusion Model** – Hybrid architecture with cross-attention between modalities for richer feature interactions.

## Features

* End-to-end preprocessing pipeline for WESAD signals (filtering, windowing, normalization).
* Standalone and fusion deep learning models with Conv1D + BiLSTM backbones.
* Precision, recall, F1, specificity, and ROC-AUC evaluation metrics.
* Precision-Recall curves for visual model comparison.
* Modular structure: `src/` for code, `results/` for evaluation metrics.

## Project Structure

```
├── src/
│   ├── standalone_model.py       # Single modality (BVP) pipeline
│   ├── simple_fusion_model.py    # EDA+BVP fusion pipeline
│   ├── cross_attention_model.py  # Cross-attention fusion pipeline
├── results/
│   ├── results.md                # Evaluation summary and curves
├── requirements.txt              # Dependencies
├── README.md                     # Project documentation
```

## Setup

### 1. Clone Repository

```bash
git clone <repo-url>
cd <repo-name>
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare Dataset

* Download the **WESAD dataset** from [link](https://www.ubicomplab.cs.washington.edu/wesad/).
* Preprocess signals using filtering and windowing (scripts provided in `src/`).
* Place processed CSV files in the `Filtered_Signals/` directory (path configurable in scripts).

## Usage

### Train and Evaluate Models

Run individual scripts:

```bash
python src/standalone_model.py
python src/simple_fusion_model.py
python src/cross_attention_model.py
```

Each script outputs classification metrics and precision-recall curves.

## Results Summary

**Standalone Model:** Accuracy 79.7%, ROC-AUC 87.9%

**Simple Fusion Model:** Accuracy 85.3%, ROC-AUC 89.2%

**Cross-Attention Fusion Model:** Accuracy 83.1%, ROC-AUC 90.6%

See detailed analysis in [results.md`](results/results.md).

## Insights

* Fusion models outperform standalone models by combining complementary EDA and BVP features.
* Simple fusion yields balanced precision and recall; cross-attention excels in recall (safety-critical scenarios).

## Future Work

* Add real-time inference pipeline for wearable devices.
* Optimize architectures for edge deployment (TensorFlow Lite / microcontrollers).
* Incorporate additional signals (temperature, accelerometer) for improved robustness.

## License

MIT License
