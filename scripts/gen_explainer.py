import os, textwrap

files = {}

files['backend/engine/explainer.py'] = textwrap.dedent('''
import numpy as np
import pandas as pd

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class RiskExplainer:
    def __init__(self, risk_adapter, feature_contract):
        self.adapter = risk_adapter
        self.feature_contract = feature_contract
        self._explainer = None

    def _init_explainer(self):
        if not SHAP_AVAILABLE:
            return
        raw_lgbm = self.adapter.get_raw_lgbm()
        if raw_lgbm is not None:
            try:
                self._explainer = shap.TreeExplainer(raw_lgbm)
            except Exception:
                self._explainer = None

    def explain(self, applicant_df, top_k=5):
        if not SHAP_AVAILABLE:
            return {'available': False, 'message': 'shap not installed'}

        if self._explainer is None:
            self._init_explainer()
        if self._explainer is None:
            return {'available': False, 'message': 'Could not init SHAP explainer'}

        try:
            df = applicant_df.copy()
            if self.adapter._credit_transformer:
                df = self.adapter._credit_transformer.transform(df)
            all_cols = self.adapter._numeric_features + self.adapter._cat_features
            for col in all_cols:
                if col not in df.columns:
                    df[col] = np.nan
            df = df[all_cols]

            pipeline = self.adapter._raw_pipeline
            prep = pipeline.named_steps['prep']
            X_transformed = prep.transform(df)

            feature_names = []
            for name, trans, cols in prep.transformers_:
                if name == 'num':
                    feature_names.extend(cols)
                elif name == 'cat':
                    ohe = trans.named_steps.get('ohe')
                    if ohe is not None and hasattr(ohe, 'get_feature_names_out'):
                        feature_names.extend(ohe.get_feature_names_out(cols).tolist())
                    else:
                        feature_names.extend(cols)

            if hasattr(X_transformed, 'toarray'):
                X_transformed = X_transformed.toarray()
            X_transformed = pd.DataFrame(X_transformed, columns=feature_names[:X_transformed.shape[1]])

            shap_values = self._explainer.shap_values(X_transformed)
            if isinstance(shap_values, list):
                sv = shap_values[1][0]
            else:
                sv = shap_values[0]

            drivers = []
            indices = np.argsort(np.abs(sv))[::-1][:top_k]
            for idx in indices:
                fname = feature_names[idx] if idx < len(feature_names) else f'feature_{idx}'
                base_name = fname.split('_x0_')[0] if '_x0_' in fname else fname
                fc_entry = self.feature_contract.get(base_name)
                drivers.append({
                    'feature': fname,
                    'contribution': round(float(sv[idx]), 4),
                    'direction': 'increases_risk' if sv[idx] > 0 else 'decreases_risk',
                    'actionability': fc_entry.feature_class if fc_entry else 'UNKNOWN',
                    'action': fc_entry.corresponding_action if fc_entry else '',
                })

            return {'available': True, 'top_risk_drivers': drivers}
        except Exception as e:
            return {'available': False, 'message': f'SHAP error: {e}'}
''').lstrip()

for path, content in files.items():
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(content)
    print(f'  wrote {path}')
print('Done.')
