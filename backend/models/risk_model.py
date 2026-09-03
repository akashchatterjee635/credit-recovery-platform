'''
backend/models/risk_model.py

RiskModelAdapter with:
  1. 3-way stratified split: train 60%, calibration 20%, test 20%
  2. LightGBM trained WITHOUT class_weight='balanced' (for calibration correctness)
  3. Isotonic calibration using sklearn CalibratedClassifierCV(method='isotonic', cv='prefit') fitted on calibration split
  4. ECE computation function (Expected Calibration Error, 10 equal-width bins)
  5. Reports: ROC-AUC, PR-AUC, Recall, Precision, F1, Brier, ECE for both uncalibrated and calibrated models
  6. LR baseline (with class_weight='balanced') for comparison
  7. Saves calibration data to experiments/outputs/calibration_data.json for the calibration experiment
  8. Persists the CALIBRATED model as production model, raw pipeline stored separately
  9. predict_risk() returns calibrated probabilities
  10. get_raw_lgbm() method to extract the underlying LightGBM model for SHAP
'''
import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    recall_score,
    precision_score,
    f1_score,
    brier_score_loss,
)
import lightgbm as lgb

from backend.models.feature_engineering import CreditDataTransformer, build_enriched_dataset

MODEL_PATH = 'backend/models/lgbm_model.pkl'
RAW_MODEL_PATH = 'backend/models/lgbm_raw_pipeline.pkl'
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


def compute_ece(y_true, y_prob, n_bins: int = 10) -> float:
    '''
    Compute Expected Calibration Error (ECE) using equal-width bins.
    ECE = sum(|B_m| / N * |acc(B_m) - conf(B_m)|)
    '''
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_samples = len(y_true)

    for i in range(n_bins):
        bin_lower = bin_edges[i]
        bin_upper = bin_edges[i + 1]
        if i == n_bins - 1:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)

        bin_size = int(np.sum(in_bin))
        if bin_size > 0:
            avg_prob = float(np.mean(y_prob[in_bin]))
            avg_true = float(np.mean(y_true[in_bin]))
            ece += (bin_size / n_samples) * abs(avg_true - avg_prob)

    return float(ece)


def _get_calibration_curve_data(y_true, y_prob, n_bins: int = 10) -> dict:
    '''Compute calibration curve points for reliability diagram visualization.'''
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    return {
        'prob_true': [float(x) for x in prob_true],
        'prob_pred': [float(x) for x in prob_pred],
    }


def _eval_metrics(y_true, y_prob, y_pred, label: str = '') -> dict:
    '''Compute and report comprehensive evaluation metrics.'''
    roc = float(roc_auc_score(y_true, y_prob))
    pr = float(average_precision_score(y_true, y_prob))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    brier = float(brier_score_loss(y_true, y_prob))
    ece = float(compute_ece(y_true, y_prob))

    print(f'\n  --- {label} ---')
    print(f'  ROC-AUC  : {roc:.4f}')
    print(f'  PR-AUC   : {pr:.4f}')
    print(f'  Recall   : {rec:.4f}')
    print(f'  Precision: {prec:.4f}')
    print(f'  F1       : {f1:.4f}')
    print(f'  Brier    : {brier:.4f}')
    print(f'  ECE      : {ece:.4f}')

    return {
        'roc_auc': round(roc, 4),
        'pr_auc': round(pr, 4),
        'recall': round(rec, 4),
        'precision': round(prec, 4),
        'f1': round(f1, 4),
        'brier': round(brier, 4),
        'ece': round(ece, 4),
    }


class RiskModelAdapter:
    def __init__(self):
        self.model = None              # Production calibrated classifier
        self.raw_pipeline = None       # Uncalibrated pipeline (prep + LightGBM)
        self._credit_transformer = None
        self._numeric_features = []
        self._cat_features = []

    def train(self, data_dir: str = 'data'):
        print('Loading and enriching dataset...')
        df = build_enriched_dataset(data_dir)

        target = 'TARGET'
        df = df.dropna(subset=[target])

        # Select whichever features actually exist after merging
        all_potential = BASE_FEATURES + BUREAU_FEATURES + PREV_APP_FEATURES + INST_FEATURES
        numeric_features = [
            f for f in all_potential + DERIVED_FEATURES
            if f in df.columns and f not in CATEGORICAL_FEATURES
        ]
        cat_features = [f for f in CATEGORICAL_FEATURES if f in df.columns]
        use_features = numeric_features + cat_features

        X = df[use_features]
        y = df[target].astype(int)

        print(f'Features used: {len(use_features)} ({len(numeric_features)} numeric, {len(cat_features)} categorical)')

        # 1. 3-way stratified split: train 60%, calibration 20%, test 20%
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.40, random_state=42, stratify=y
        )
        X_cal, X_test, y_cal, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
        )

        print(f'Split sizes -> Train: {len(X_train)} (60%), Calibration: {len(X_cal)} (20%), Test: {len(X_test)} (20%)')
        print(f'Default rate -> Train: {y_train.mean():.3%}, Cal: {y_cal.mean():.3%}, Test: {y_test.mean():.3%}')

        # Custom feature transformer (fitted on train split only - no leakage)
        credit_tf = CreditDataTransformer()
        X_train = credit_tf.fit_transform(X_train)
        X_cal   = credit_tf.transform(X_cal)
        X_test  = credit_tf.transform(X_test)

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

        # 2. LightGBM trained WITHOUT class_weight='balanced' (for calibration correctness)
        print('\nTraining LightGBM (without class_weight for calibration correctness)...')
        lgbm_pipe = Pipeline([
            ('prep', preprocessor),
            ('clf', lgb.LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=63,
                random_state=42,
                verbosity=-1,
            )),
        ])
        lgbm_pipe.fit(X_train, y_train)

        # 3. Isotonic calibration using sklearn CalibratedClassifierCV(method='isotonic', cv='prefit') fitted on calibration split
        print('\nFitting Isotonic Calibration on calibration split...')
        try:
            cal_model = CalibratedClassifierCV(
                estimator=lgbm_pipe,
                method='isotonic',
                cv='prefit',
            )
            cal_model.fit(X_cal, y_cal)
        except Exception:
            # Compatibility with scikit-learn >= 1.6 where cv='prefit' uses FrozenEstimator
            from sklearn.frozen import FrozenEstimator
            cal_model = CalibratedClassifierCV(
                estimator=FrozenEstimator(lgbm_pipe),
                method='isotonic',
            )
            cal_model.fit(X_cal, y_cal)

        # 6. Logistic Regression baseline (with class_weight='balanced') for comparison
        print('\nTraining Logistic Regression baseline (with class_weight="balanced")...')
        lr_pipe = Pipeline([
            ('prep', preprocessor),
            ('clf', LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=42,
                solver='lbfgs',
            )),
        ])
        lr_pipe.fit(X_train, y_train)

        # 5. Reports: ROC-AUC, PR-AUC, Recall, Precision, F1, Brier, ECE for both uncalibrated and calibrated models
        print('\n' + '=' * 60)
        print('EVALUATION ON TEST SPLIT (20%)')
        print('=' * 60)

        # Uncalibrated LightGBM
        lgbm_prob = lgbm_pipe.predict_proba(X_test)[:, 1]
        lgbm_pred = (lgbm_prob >= 0.5).astype(int)
        uncal_metrics = _eval_metrics(y_test, lgbm_prob, lgbm_pred, 'LightGBM (Uncalibrated)')

        # Calibrated LightGBM
        cal_prob = cal_model.predict_proba(X_test)[:, 1]
        cal_pred = (cal_prob >= 0.5).astype(int)
        cal_metrics = _eval_metrics(y_test, cal_prob, cal_pred, 'LightGBM (Calibrated - Isotonic)')

        # LR baseline
        lr_prob = lr_pipe.predict_proba(X_test)[:, 1]
        lr_pred = (lr_prob >= 0.5).astype(int)
        lr_metrics = _eval_metrics(y_test, lr_prob, lr_pred, 'Logistic Regression (Baseline)')
        print('=' * 60)

        # 7. Saves calibration data to experiments/outputs/calibration_data.json for the calibration experiment
        cal_data = {
            'uncalibrated': {
                'metrics': uncal_metrics,
                'curve': _get_calibration_curve_data(y_test, lgbm_prob),
            },
            'calibrated': {
                'metrics': cal_metrics,
                'curve': _get_calibration_curve_data(y_test, cal_prob),
            },
            'lr_baseline': {
                'metrics': lr_metrics,
                'curve': _get_calibration_curve_data(y_test, lr_prob),
            },
            'summary': {
                'n_train': int(len(X_train)),
                'n_calibration': int(len(X_cal)),
                'n_test': int(len(X_test)),
                'default_rate': float(y_test.mean()),
                'features_used': use_features,
            },
        }

        cal_out_dir = os.path.join('experiments', 'outputs')
        os.makedirs(cal_out_dir, exist_ok=True)
        cal_out_path = os.path.join(cal_out_dir, 'calibration_data.json')
        with open(cal_out_path, 'w', encoding='utf-8') as f:
            json.dump(cal_data, f, indent=2)
        print(f'\nSaved calibration data to {cal_out_path}')

        # 8. Persists the CALIBRATED model as production model, raw pipeline stored separately
        self.model = cal_model
        self.raw_pipeline = lgbm_pipe
        self._credit_transformer = credit_tf
        self._numeric_features = numeric_features_upd
        self._cat_features = cat_features_upd

        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

        # Production model file contains the calibrated model as 'pipeline' / 'calibrated_model'
        # and retains the raw pipeline as 'raw_pipeline'
        joblib.dump({
            'pipeline': cal_model,
            'calibrated_model': cal_model,
            'raw_pipeline': lgbm_pipe,
            'credit_tf': credit_tf,
            'numeric_features': numeric_features_upd,
            'cat_features': cat_features_upd,
        }, MODEL_PATH)

        # Raw pipeline stored separately
        joblib.dump({
            'pipeline': lgbm_pipe,
            'credit_tf': credit_tf,
            'numeric_features': numeric_features_upd,
            'cat_features': cat_features_upd,
        }, RAW_MODEL_PATH)

        # LR baseline model
        joblib.dump({
            'pipeline': lr_pipe,
            'credit_tf': credit_tf,
            'numeric_features': numeric_features_upd,
            'cat_features': cat_features_upd,
        }, LR_MODEL_PATH)

        print(f'Calibrated production model saved to {MODEL_PATH}')
        print(f'Raw uncalibrated pipeline saved to {RAW_MODEL_PATH}')
        print(f'Logistic Regression baseline saved to {LR_MODEL_PATH}')

    def load(self, path: str = MODEL_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(f'Model not found at {path}. Train first.')
        payload = joblib.load(path)
        self.model = payload.get('calibrated_model', payload.get('pipeline'))
        self.raw_pipeline = payload.get('raw_pipeline')
        if self.raw_pipeline is None and os.path.exists(RAW_MODEL_PATH):
            try:
                raw_payload = joblib.load(RAW_MODEL_PATH)
                self.raw_pipeline = raw_payload.get('pipeline', raw_payload)
            except Exception:
                pass
        self._credit_transformer = payload.get('credit_tf')
        self._numeric_features = payload.get('numeric_features', [])
        self._cat_features = payload.get('cat_features', [])

    def predict_risk(self, applicant_df: pd.DataFrame) -> np.ndarray:
        '''
        Returns calibrated default probabilities for input applicants.
        '''
        if self.model is None:
            self.load()
        df = applicant_df.copy()
        if self._credit_transformer:
            df = self._credit_transformer.transform(df)
        all_expected = self._numeric_features + self._cat_features
        for col in all_expected:
            if col not in df.columns:
                df[col] = np.nan
        df = df[all_expected]
        return self.model.predict_proba(df)[:, 1]

    def get_raw_lgbm(self):
        '''
        Extract the underlying LightGBM model for SHAP (e.g. TreeExplainer).
        '''
        if self.model is None and self.raw_pipeline is None:
            self.load()

        # Check raw pipeline first
        if self.raw_pipeline is not None:
            if hasattr(self.raw_pipeline, 'named_steps') and 'clf' in self.raw_pipeline.named_steps:
                return self.raw_pipeline.named_steps['clf']
            return self.raw_pipeline

        # Extract from calibrated model
        if self.model is not None:
            if hasattr(self.model, 'named_steps') and 'clf' in self.model.named_steps:
                return self.model.named_steps['clf']
            if hasattr(self.model, 'estimator'):
                est = self.model.estimator
                if hasattr(est, 'estimator'):
                    est = est.estimator
                if hasattr(est, 'named_steps') and 'clf' in est.named_steps:
                    return est.named_steps['clf']
                return est
            if hasattr(self.model, 'calibrated_classifiers_') and len(self.model.calibrated_classifiers_) > 0:
                cc = self.model.calibrated_classifiers_[0]
                if hasattr(cc, 'estimator'):
                    est = cc.estimator
                    if hasattr(est, 'estimator'):
                        est = est.estimator
                    if hasattr(est, 'named_steps') and 'clf' in est.named_steps:
                        return est.named_steps['clf']
                    return est

        return None


if __name__ == '__main__':
    adapter = RiskModelAdapter()
    adapter.train()
