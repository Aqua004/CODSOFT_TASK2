# Credit Card Fraud Detection (CodSoft Task 2)

Binary transaction classifier using a leakage-safe preprocessing pipeline and class-weighted logistic regression.

## Dataset
Download the dataset linked in the CodSoft Task 2 PDF and place `fraudTrain.csv` in `data/`. This project targets the common dataset with an `is_fraud` label; if your download uses another schema, adjust the parser before training. Never publish raw financial or personal data.

## Run
```bash
python -m venv .venv
# Activate the environment
pip install -r requirements.txt
python src/train.py --data data/fraudTrain.csv
```
Outputs: `results/metrics.json` and `models/fraud_model.joblib` (gitignored). Inspect fraud-class precision and recall, average precision (PR-AUC), ROC-AUC, and confusion matrix rather than accuracy alone. The holdout split is stratified; all imputers, encoders, and scaling fit on training data only. For production-like testing, use a future time-period holdout and entity-aware checks.

## Demo / submission
Record a training run and explain class imbalance and errors. Add observed numbers only after running on your dataset.

## Results
Not yet run or independently verified.
