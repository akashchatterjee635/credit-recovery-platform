import os, textwrap

files = {}

# ── feature_engineering.py ───────────────────────────────────────────────────
files['backend/models/feature_engineering.py'] = textwrap.dedent("""
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
""").lstrip()

# ── risk_model.py (V2: stratified split, AUC/F1/Brier, LR baseline) ──────────
files['backend/models/risk_model.py'] = textwrap.dedent("""
'''
backend/models/risk_model.py  – V2
Changes from V1:
  - Stratified train-test split
  - Reports ROC-AUC, PR-AUC, Recall, F1, Brier (not accuracy)
  - Logistic Regression baseline alongside LightGBM
  - Custom CreditDataTransformer is part of the persisted Pipeline (no leakage)
  - Bureau / previous-app / installment features used when available
'''
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              recall_score, precision_score,
                              f1_score, brier_score_loss)
import lightgbm as lgb

from backend.models.feature_engineering import CreditDataTransformer, build_enriched_dataset

MODEL_PATH = 'backend/models/lgbm_model.pkl'
LR_MODEL_PATH = 'backend/models/lr_baseline_model.pkl'

BASE_FEATURES = [
    'AMT_CREDIT', 'AMT_INCOME_TOTAL', 'AMT_ANNUITY',
    'DAYS_BIRTH', 'DAYS_EMPLOYED', 'NAME_EDUCATION_TYPE',
]
DERIVED_FEATURES = ['DERIVED_DTI', 'DERIVED_ANNUITY_CREDIT_RATIO', 'DERIVED_CREDIT_INCOME_RATIO']
BUREAU_FEATURES = [
    'BUREAU_ACTIVE_COUNT', 'BUREAU_AVG_DAYS_CREDIT',
    'BUREAU_MAX_OVERDUE', 'BUREAU_TOTAL_DEBT', 'BUREAU_AVG_DAYS_OVERDUE',
]
PREV_APP_FEATURES = ['PREV_APP_COUNT', 'PREV_REFUSED_RATIO', 'PREV_AMT_CREDIT_MEAN']
INST_FEATURES = ['INST_LATE_RATIO', 'INST_AVG_PAYMENT_DIFF', 'INST_AVG_DAYS_LATE']
CATEGORICAL_FEATURES = ['NAME_EDUCATION_TYPE']


def _eval_metrics(y_true, y_prob, y_pred, label=''):
    thresh = 0.5
    print(f'\\n  --- {label} ---')
    print(f'  ROC-AUC  : {roc_auc_score(y_true, y_prob):.4f}')
    print(f'  PR-AUC   : {average_precision_score(y_true, y_prob):.4f}')
    print(f'  Recall   : {recall_score(y_true, y_pred):.4f}')
    print(f'  Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}')
    print(f'  F1       : {f1_score(y_true, y_pred, zero_division=0):.4f}')
    print(f'  Brier    : {brier_score_loss(y_true, y_prob):.4f}')


class RiskModelAdapter:
    def __init__(self):
        self.model = None

    def train(self, data_dir='data'):
        print('Loading and enriching dataset...')
        df = build_enriched_dataset(data_dir)

        target = 'TARGET'
        df = df.dropna(subset=[target])

        # Select whichever features actually exist after merging
        all_potential = BASE_FEATURES + BUREAU_FEATURES + PREV_APP_FEATURES + INST_FEATURES
        numeric_features = [f for f in all_potential + DERIVED_FEATURES
                            if f in df.columns and f not in CATEGORICAL_FEATURES]
        cat_features = [f for f in CATEGORICAL_FEATURES if f in df.columns]
        use_features = numeric_features + cat_features

        X = df[use_features]
        y = df[target].astype(int)

        print(f'Features used: {len(use_features)} ({len(numeric_features)} numeric, {len(cat_features)} categorical)')

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f'Train: {len(X_train)}  Test: {len(X_test)}  Default rate: {y.mean():.3%}')

        # Custom transformer (fitted on train only – no leakage)
        credit_tf = CreditDataTransformer()
        X_train = credit_tf.fit_transform(X_train)
        X_test  = credit_tf.transform(X_test)

        # Update feature lists after derived columns are added
        numeric_features_upd = [f for f in X_train.columns if f not in cat_features]
        cat_features_upd = [f for f in cat_features if f in X_train.columns]

        preprocessor = ColumnTransformer([
            ('num', Pipeline([
                ('impute', SimpleImputer(strategy='median')),
                ('scale', StandardScaler()),
            ]), numeric_features_upd),
            ('cat', Pipeline([
                ('impute', SimpleImputer(strategy='most_frequent')),
                ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
            ]), cat_features_upd),
        ])

        # ── LightGBM ──────────────────────────────────────────────────────────
        print('\\nTraining LightGBM...')
        lgbm_pipe = Pipeline([
            ('prep', preprocessor),
            ('clf', lgb.LGBMClassifier(
                n_estimators=500, learning_rate=0.05,
                num_leaves=63, class_weight='balanced',
                random_state=42, verbosity=-1,
            )),
        ])
        lgbm_pipe.fit(X_train, y_train)
        lgbm_prob = lgbm_pipe.predict_proba(X_test)[:, 1]
        lgbm_pred = (lgbm_prob >= 0.5).astype(int)
        _eval_metrics(y_test, lgbm_prob, lgbm_pred, 'LightGBM')

        # ── Logistic Regression baseline ──────────────────────────────────────
        print('\\nTraining Logistic Regression baseline...')
        lr_pipe = Pipeline([
            ('prep', preprocessor),
            ('clf', LogisticRegression(max_iter=1000, class_weight='balanced',
                                       random_state=42, solver='lbfgs')),
        ])
        lr_pipe.fit(X_train, y_train)
        lr_prob = lr_pipe.predict_proba(X_test)[:, 1]
        lr_pred = (lr_prob >= 0.5).astype(int)
        _eval_metrics(y_test, lr_prob, lr_pred, 'Logistic Regression (baseline)')

        # ── Persist ───────────────────────────────────────────────────────────
        self.model = lgbm_pipe
        self._credit_transformer = credit_tf
        self._numeric_features = numeric_features_upd
        self._cat_features = cat_features_upd

        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump({'pipeline': lgbm_pipe, 'credit_tf': credit_tf,
                     'numeric_features': numeric_features_upd,
                     'cat_features': cat_features_upd}, MODEL_PATH)
        joblib.dump({'pipeline': lr_pipe, 'credit_tf': credit_tf,
                     'numeric_features': numeric_features_upd,
                     'cat_features': cat_features_upd}, LR_MODEL_PATH)
        print(f'\\nModels saved to {MODEL_PATH} and {LR_MODEL_PATH}')

    def load(self, path=MODEL_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(f'Model not found at {path}. Train first.')
        payload = joblib.load(path)
        self.model = payload['pipeline']
        self._credit_transformer = payload.get('credit_tf')
        self._numeric_features = payload.get('numeric_features', [])
        self._cat_features = payload.get('cat_features', [])

    def predict_risk(self, applicant_df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            self.load()
        df = applicant_df.copy()
        if self._credit_transformer:
            df = self._credit_transformer.transform(df)
        # Align columns to what the pipeline expects
        all_expected = self._numeric_features + self._cat_features
        for col in all_expected:
            if col not in df.columns:
                df[col] = np.nan
        df = df[all_expected]
        return self.model.predict_proba(df)[:, 1]


if __name__ == '__main__':
    adapter = RiskModelAdapter()
    adapter.train()
""").lstrip()

for path, content in files.items():
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(content)
    print(f'  wrote {path}')

print('Wave 3 files written.')
