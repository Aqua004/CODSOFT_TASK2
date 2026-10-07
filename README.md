## Results

The model was trained using a preprocessing pipeline and class-weighted Logistic Regression.

- Accuracy: 0.88
- Fraud precision: 0.04
- Fraud recall: 0.76
- Fraud F1-score: 0.07
- Average precision / PR-AUC: 0.1733
- Test samples: 259,335

The dataset is highly imbalanced. Therefore, accuracy alone is not sufficient. The model detects many fraudulent transactions, as shown by its recall of 0.76, but its low precision of 0.04 indicates a high number of false positives.