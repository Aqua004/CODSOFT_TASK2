import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, average_precision_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/fraudTrain.csv')
    args = parser.parse_args()
    data = pd.read_csv(args.data)
    if 'is_fraud' not in data:
        parser.error('Expected is_fraud label; adapt columns for another dataset')
    y = pd.to_numeric(data['is_fraud'], errors='raise').astype(int)
    if set(y.unique()) != {0, 1}:
        parser.error('is_fraud must contain both 0 and 1')
    data = data.drop(columns=['is_fraud'])
    if 'trans_date_trans_time' in data:
        ts = pd.to_datetime(data.pop('trans_date_trans_time'), errors='coerce')
        data['hour'] = ts.dt.hour
        data['day_of_week'] = ts.dt.dayofweek
    excluded = ['Unnamed: 0', 'trans_num', 'cc_num', 'first', 'last', 'street', 'dob', 'merchant', 'job', 'city', 'state', 'zip']
    data = data.drop(columns=excluded, errors='ignore')
    numeric = data.select_dtypes(include='number').columns.tolist()
    categorical = data.select_dtypes(include=['object', 'category']).columns.tolist()
    x = data[numeric + categorical]
    if x.shape[1] == 0:
        parser.error('No usable features found')
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)
    preprocess = ColumnTransformer([
        ('num', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), numeric),
        ('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('encode', OneHotEncoder(handle_unknown='ignore'))]), categorical),
    ])
    model = Pipeline([('preprocess', preprocess), ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000))])
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    scores = model.predict_proba(x_test)[:, 1]

    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=['Not Fraud', 'Fraud'],
        cmap='Blues',
        colorbar=True
    )
    plt.title('Credit Card Fraud Detection - Confusion Matrix')
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png', dpi=200)
    plt.close()

    RocCurveDisplay.from_predictions(
        y_test,
        scores,
        name='Logistic Regression',
        color='darkorange'
    )
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random classifier')
    plt.title('Credit Card Fraud Detection - ROC-AUC Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/roc_curve.png', dpi=200)
    plt.close()

    metrics = {
        'fraud_rate_test': float(y_test.mean()),
        'average_precision': average_precision_score(y_test, scores),
        'roc_auc': roc_auc_score(y_test, scores),
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(y_test, predictions, output_dict=True, zero_division=0),
    }
    Path('models').mkdir(exist_ok=True)
    Path('results').mkdir(exist_ok=True)
    joblib.dump(model, 'models/fraud_model.joblib')
    Path('results/metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(classification_report(y_test, predictions, zero_division=0))
    print(f'Average precision: {metrics["average_precision"]:.4f}')
    print(f'ROC-AUC: {metrics["roc_auc"]:.4f}')
    print('Confusion matrix saved to results/confusion_matrix.png')
    print('ROC-AUC curve saved to results/roc_curve.png')


if __name__ == '__main__':
    main()
