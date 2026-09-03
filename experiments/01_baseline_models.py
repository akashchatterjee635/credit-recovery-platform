'''
experiments/01_baseline_models.py
Compares Logistic Regression vs LightGBM on the Home Credit dataset.
Outputs a metrics table (ROC-AUC, PR-AUC, Recall, Precision, F1, Brier).

Usage:
    cd credit_recovery_platform
    .\venv\Scripts\python experiments/01_baseline_models.py
'''
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              recall_score, precision_score, f1_score,
                              brier_score_loss)
import lightgbm as lgb

from backend.models.feature_engineering import build_enriched_dataset, CreditDataTransformer

DATA_DIR = 'data'
NUMERIC = ['AMT_CREDIT', 'AMT_INCOME_TOTAL', 'AMT_ANNUITY',
           'DAYS_BIRTH', 'DAYS_EMPLOYED']
CAT = ['NAME_EDUCATION_TYPE']


def build_pipeline(classifier):
    pre = ColumnTransformer([
        ('num', Pipeline([('imp', SimpleImputer(strategy='median')),
                          ('sc', StandardScaler())]), NUMERIC),
        ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                          ('ohe', OneHotEncoder(handle_unknown='ignore'))]), CAT),
    ])
    return Pipeline([('prep', pre), ('clf', classifier)])


def evaluate(name, model, X_test, y_test):
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return {
        'Model': name,
        'ROC-AUC':   round(roc_auc_score(y_test, prob), 4),
        'PR-AUC':    round(average_precision_score(y_test, prob), 4),
        'Recall':    round(recall_score(y_test, pred), 4),
        'Precision': round(precision_score(y_test, pred, zero_division=0), 4),
        'F1':        round(f1_score(y_test, pred, zero_division=0), 4),
        'Brier':     round(brier_score_loss(y_test, prob), 4),
    }


if __name__ == '__main__':
    print('Loading data...')
    df = build_enriched_dataset(DATA_DIR)
    df = df.dropna(subset=['TARGET'])

    use_cols = NUMERIC + CAT
    X = df[[c for c in use_cols if c in df.columns]]
    y = df['TARGET'].astype(int)

    tf = CreditDataTransformer()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    X_train = tf.fit_transform(X_train)
    X_test  = tf.transform(X_test)

    results = []

    lr = build_pipeline(LogisticRegression(max_iter=1000, class_weight='balanced',
                                            random_state=42))
    print('Training Logistic Regression...')
    lr.fit(X_train, y_train)
    results.append(evaluate('Logistic Regression', lr, X_test, y_test))

    lgbm = build_pipeline(lgb.LGBMClassifier(n_estimators=300, class_weight='balanced',
                                               random_state=42, verbosity=-1))
    print('Training LightGBM...')
    lgbm.fit(X_train, y_train)
    results.append(evaluate('LightGBM', lgbm, X_test, y_test))

    print('\n' + '='*70)
    print('BASELINE MODEL COMPARISON')
    print('='*70)
    res_df = pd.DataFrame(results).set_index('Model')
    print(res_df.to_string())
    print('='*70)
