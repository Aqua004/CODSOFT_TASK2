# Credit Card Fraud Detection (CodSoft Task 2)

Detect fraudulent transactions using the Kaggle dataset: https://www.kaggle.com/datasets/kartik2112/fraud-detection

## Dataset
Download the dataset and place `fraudTrain.csv` in `data/`. The label column is `is_fraud` (0 = legitimate, 1 = fraud). The test file `fraudTest.csv` may be used for later evaluation; this project uses a stratified holdout split from `fraudTrain.csv` for reproducible baseline metrics. Do not commit raw transaction data.

## Methodology
- Feature engineering: transaction hour and day of week from transaction timestamp.
- Preprocessing: median imputation and scaling for numeric fields; most-frequent imputation and one-hot encoding for categorical fields.
- Model: class-weighted Logistic Regression.
- Evaluation: stratified 80/20 train/test split with `random_state=42`.
- Metrics: fraud-class precision, recall, F1, average precision (PR-AUC), ROC-AUC, and confusion matrix.

## Run
```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\activate
pip install -r requirements.txt
python src/train.py --data data/fraudTrain.csv
```

## Confusion Matrix
After training, the script saves a confusion-matrix image at:

```text
results/confusion_matrix.png
```

It also saves the numeric matrix in `results/metrics.json` under `confusion_matrix`.

|  | Predicted: Not Fraud | Predicted: Fraud |
|---|---:|---:|
| **Actual: Not Fraud** | True Negative (TN) | False Positive (FP) |
| **Actual: Fraud** | False Negative (FN) | True Positive (TP) |

- **True Negative:** Legitimate transaction correctly classified as legitimate.
- **False Positive:** Legitimate transaction incorrectly flagged as fraud.
- **False Negative:** Fraudulent transaction missed by the model.
- **True Positive:** Fraudulent transaction correctly detected.

For fraud detection, the most important cells are **False Negative** and **True Positive**: a high True Positive count and low False Negative count mean the model detects more actual fraud cases.

## Verified Results
The project was run locally on the downloaded dataset.

- Test samples: 259,335
- Accuracy: 0.88
- Fraud precision: 0.04
- Fraud recall: 0.76
- Fraud F1-score: 0.07
- Average precision / PR-AUC: 0.1733

The dataset is highly imbalanced: the test split contained 1,501 fraud transactions and 257,834 legitimate transactions. Accuracy is therefore not a sufficient quality measure. The baseline detects 76% of fraud cases but has low precision, meaning it creates many false-positive fraud alerts. Improving threshold selection, feature engineering, and model choice would be appropriate next steps.

## Output
Training saves `models/fraud_model.joblib`, `results/metrics.json`, and `results/confusion_matrix.png` locally. These files are excluded from Git.
