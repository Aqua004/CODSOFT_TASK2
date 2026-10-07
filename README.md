# Credit Card Fraud Detection (CodSoft Task 2)

Detect fraudulent transactions using the Kaggle dataset: https://www.kaggle.com/datasets/kartik2112/fraud-detection

## Dataset
Download the dataset and place `fraudTrain.csv` in `data/`. The label column is `is_fraud` (0 = legitimate, 1 = fraud). The test file `fraudTest.csv` is for final evaluation; this project uses a holdout split from `fraudTrain.csv` for reproducible metrics. Do not commit raw transaction data.

## Run
```bash
python -m venv .venv
# Activate the virtual environment
pip install -r requirements.txt
python src/train.py --data data/fraudTrain.csv
```
Outputs: `models/fraud_model.joblib` and `results/metrics.json` (both gitignored). Reports fraud-class precision/recall/F1, ROC-AUC, PR-AUC, and confusion matrix. All preprocessing is fit after the stratified split.

## Submission
Run on your downloaded dataset, record a demo video, and add measured results here only after verifying them.

## Results
Not yet run or independently verified.
