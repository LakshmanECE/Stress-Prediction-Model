# Model Fusion Results

## Overview

This document summarizes evaluation metrics and key insights for three model variants tested on the WESAD-derived stress detection pipeline.

---

## Metrics Summary

| Model                         | Accuracy | Precision | Recall (Sens) | Specificity | F1 Score | ROC-AUC |
| ----------------------------- | -------- | --------- | ------------- | ----------- | -------- | ------- |
| **Standalone (BVP only)**     | 79.70%   | 61.08%    | 34.58%        | 93.34%      | 44.16%   | 87.86%  |
| **Simple Fusion (EDA + BVP)** | 85.37%   | 63.73%    | 85.76%        | 85.25%      | 73.12%   | 89.21%  |
| **Cross-Attention Fusion**    | 83.08%   | 58.70%    | 91.53%        | 80.53%      | 71.52%   | 90.55%  |

---

## Observations

### Standalone Model

* High specificity but low recall: reliably detects non-stress but misses many stress events.
* ROC-AUC is decent (87.86%) but precision-recall tradeoff is weak.

### Simple Fusion Model

* Balanced sensitivity and specificity.
* Major improvement in recall and F1 score compared to standalone.
* Good tradeoff between precision and recall, suitable for balanced use cases.

### Cross-Attention Fusion Model

* Highest recall (91.53%) – captures most stress events.
* Precision lower than simple fusion, leading to more false positives.
* Best ROC-AUC overall (90.55%) – strong discrimination ability.

---

## Precision-Recall Curves

* **Standalone**: <img width="866" height="680" alt="image" src="https://github.com/user-attachments/assets/b8ff3940-2a8c-4ba9-b926-6a2cb7621932" />

* **Simple Fusion**: <img width="860" height="677" alt="image" src="https://github.com/user-attachments/assets/42d5e7a6-2e9d-4edb-aa8d-32857630daf9" />

* **Cross-Attention Fusion**: <img width="866" height="680" alt="image" src="https://github.com/user-attachments/assets/c88334bf-5931-4274-8f22-ec9049879cb7" />


---

## Key Insights

* Fusion models (EDA + BVP) outperform standalone BVP-only model.
* Simple fusion offers **best balanced performance**.
* Cross-attention is **best when prioritizing recall** (safety-critical detection where missing stress is worse than false alarms).
* ROC-AUC ranking: **Cross-Attention > Simple Fusion > Standalone**.

---

*Note: Add links or references to plots (Precision-Recall curves) once generated as PNG/SVG files.*
