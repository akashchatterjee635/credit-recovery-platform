'''
backend/models/feature_engineering.py
Sklearn-compatible transformer that:
  1. Fixes the DAYS_EMPLOYED 365243 bug
  2. Clips income outliers (computed on TRAIN SET only -> no leakage)
  3. Aggregates bureau, previous_application, installments auxiliary tables
     when they are available alongside application_train.csv

The transformer is saved as part of the persisted sklearn Pipeline so that
training and inference use identical transformations.
'''
import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CreditDataTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, income_clip_iqr_factor: float = 3.0):
        self.income_clip_iqr_factor = income_clip_iqr_factor
        self._income_upper: float = None   # fitted on train only

    # ── fit: learn clipping bounds from training data ─────────────────────────
    def fit(self, X: pd.DataFrame, y=None):
        if 'AMT_INCOME_TOTAL' in X.columns:
            q1 = X['AMT_INCOME_TOTAL'].quantile(0.25)
            q3 = X['AMT_INCOME_TOTAL'].quantile(0.75)
            iqr = q3 - q1
            self._income_upper = q3 + self.income_clip_iqr_factor * iqr
        return self

    # ── transform: apply all cleaning steps ──────────────────────────────────
    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X = X.copy()

        # 1. Fix 1000-year employment bug
        if 'DAYS_EMPLOYED' in X.columns:
            X['DAYS_EMPLOYED'] = X['DAYS_EMPLOYED'].replace(365243, np.nan)

        # 2. Clip income outliers (using bound learned in fit)
        if 'AMT_INCOME_TOTAL' in X.columns and self._income_upper is not None:
            X['AMT_INCOME_TOTAL'] = X['AMT_INCOME_TOTAL'].clip(upper=self._income_upper)

        # 3. Derived features (safe to compute on any split)
        if 'AMT_ANNUITY' in X.columns and 'AMT_INCOME_TOTAL' in X.columns:
            X['DERIVED_DTI'] = np.where(
                X['AMT_INCOME_TOTAL'] > 0,
                X['AMT_ANNUITY'] / X['AMT_INCOME_TOTAL'],
                np.nan
            )

        if 'AMT_ANNUITY' in X.columns and 'AMT_CREDIT' in X.columns:
            X['DERIVED_ANNUITY_CREDIT_RATIO'] = np.where(
                X['AMT_CREDIT'] > 0,
                X['AMT_ANNUITY'] / X['AMT_CREDIT'],
                np.nan
            )

        if 'AMT_CREDIT' in X.columns and 'AMT_INCOME_TOTAL' in X.columns:
            X['DERIVED_CREDIT_INCOME_RATIO'] = np.where(
                X['AMT_INCOME_TOTAL'] > 0,
                X['AMT_CREDIT'] / X['AMT_INCOME_TOTAL'],
                np.nan
            )

        return X


def load_bureau_features(data_dir: str) -> pd.DataFrame:
    '''
    Aggregate bureau.csv into per-applicant summary features.
    Returns empty DataFrame if file not available.
    '''
    bureau_path = os.path.join(data_dir, 'bureau.csv')
    if not os.path.exists(bureau_path):
        print(f'  [INFO] bureau.csv not found at {bureau_path} – skipping bureau features.')
        return pd.DataFrame()

    bureau = pd.read_csv(bureau_path)
    agg = bureau.groupby('SK_ID_CURR').agg(
        BUREAU_ACTIVE_COUNT=('CREDIT_ACTIVE', lambda x: (x == 'Active').sum()),
        BUREAU_CLOSED_COUNT=('CREDIT_ACTIVE', lambda x: (x == 'Closed').sum()),
        BUREAU_AVG_DAYS_CREDIT=('DAYS_CREDIT', 'mean'),
        BUREAU_MAX_OVERDUE=('AMT_CREDIT_MAX_OVERDUE', 'max'),
        BUREAU_TOTAL_DEBT=('AMT_CREDIT_SUM_DEBT', 'sum'),
        BUREAU_TOTAL_LIMIT=('AMT_CREDIT_SUM_LIMIT', 'sum'),
        BUREAU_AVG_DAYS_OVERDUE=('CREDIT_DAY_OVERDUE', 'mean'),
        BUREAU_N_PREV_CREDITS=('SK_ID_BUREAU', 'count'),
    ).reset_index()
    print(f'  [INFO] Bureau features: {agg.shape[1]-1} cols, {len(agg)} applicants')
    return agg


def load_prev_app_features(data_dir: str) -> pd.DataFrame:
    path = os.path.join(data_dir, 'previous_application.csv')
    if not os.path.exists(path):
        print(f'  [INFO] previous_application.csv not found – skipping.')
        return pd.DataFrame()

    prev = pd.read_csv(path)
    agg = prev.groupby('SK_ID_CURR').agg(
        PREV_APP_COUNT=('SK_ID_PREV', 'count'),
        PREV_APP_REFUSED_COUNT=('NAME_CONTRACT_STATUS', lambda x: (x == 'Refused').sum()),
        PREV_APP_APPROVED_COUNT=('NAME_CONTRACT_STATUS', lambda x: (x == 'Approved').sum()),
        PREV_AMT_CREDIT_MEAN=('AMT_CREDIT', 'mean'),
        PREV_AMT_ANNUITY_MEAN=('AMT_ANNUITY', 'mean'),
    ).reset_index()
    agg['PREV_REFUSED_RATIO'] = agg['PREV_APP_REFUSED_COUNT'] / agg['PREV_APP_COUNT'].clip(lower=1)
    print(f'  [INFO] Prev-app features: {agg.shape[1]-1} cols, {len(agg)} applicants')
    return agg


def load_installment_features(data_dir: str) -> pd.DataFrame:
    path = os.path.join(data_dir, 'installments_payments.csv')
    if not os.path.exists(path):
        print(f'  [INFO] installments_payments.csv not found – skipping.')
        return pd.DataFrame()

    inst = pd.read_csv(path)
    inst['PAYMENT_DIFF'] = inst['AMT_PAYMENT'] - inst['AMT_INSTALMENT']
    inst['DAYS_LATE'] = inst['DAYS_ENTRY_PAYMENT'] - inst['DAYS_INSTALMENT']
    inst['IS_LATE'] = (inst['DAYS_LATE'] > 0).astype(int)

    agg = inst.groupby('SK_ID_CURR').agg(
        INST_COUNT=('NUM_INSTALMENT_VERSION', 'count'),
        INST_LATE_COUNT=('IS_LATE', 'sum'),
        INST_AVG_PAYMENT_DIFF=('PAYMENT_DIFF', 'mean'),
        INST_AVG_DAYS_LATE=('DAYS_LATE', 'mean'),
    ).reset_index()
    agg['INST_LATE_RATIO'] = agg['INST_LATE_COUNT'] / agg['INST_COUNT'].clip(lower=1)
    print(f'  [INFO] Installment features: {agg.shape[1]-1} cols, {len(agg)} applicants')
    return agg


def build_enriched_dataset(data_dir: str) -> pd.DataFrame:
    '''Merge application_train with all available auxiliary tables.'''
    app = pd.read_csv(os.path.join(data_dir, 'application_train.csv'))
    print(f'  [INFO] Base application data: {app.shape}')

    for loader in [load_bureau_features, load_prev_app_features, load_installment_features]:
        aux = loader(data_dir)
        if not aux.empty:
            app = app.merge(aux, on='SK_ID_CURR', how='left')

    print(f'  [INFO] Enriched dataset: {app.shape}')
    return app
