import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib
import os

try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
except ImportError:
    variance_inflation_factor = None

MODEL_PATH = "backend/models/lgbm_model.pkl"

def clean_outliers(df):
    print("Applying advanced preprocessing: Cleaning outliers...")
    # 1. Fix the 1000-year employment bug (365243 days)
    if 'DAYS_EMPLOYED' in df.columns:
        df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)
    
    # 2. Clip extreme income outliers (using 3x IQR for extreme outliers)
    if 'AMT_INCOME_TOTAL' in df.columns:
        Q1 = df['AMT_INCOME_TOTAL'].quantile(0.25)
        Q3 = df['AMT_INCOME_TOTAL'].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 3 * IQR
        df['AMT_INCOME_TOTAL'] = df['AMT_INCOME_TOTAL'].clip(upper=upper_bound)
        
    return df

def calculate_vif(df, features):
    if variance_inflation_factor is None:
        print("statsmodels not installed, skipping VIF calculation.")
        return
        
    print("Calculating Variance Inflation Factor (VIF) to check multicollinearity...")
    # Drop NAs for VIF calculation
    numeric_df = df[features].select_dtypes(include=[np.number]).dropna()
    vif_data = pd.DataFrame()
    vif_data["Feature"] = numeric_df.columns
    vif_data["VIF"] = [variance_inflation_factor(numeric_df.values, i) for i in range(len(numeric_df.columns))]
    print(vif_data.to_string(index=False))
    print("Note: High VIF (>5) indicates multicollinearity. We will handle this using structural constraints in the solver rather than dropping features.")

class RiskModelAdapter:
    def __init__(self):
        self.model = None
        self.features = []
        
    def train(self, data_path="data/application_train.csv"):
        print("Loading data...")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file {data_path} not found. Please run download script.")
            
        df = pd.read_csv(data_path)
        
        # Apply outlier cleaning
        df = clean_outliers(df)
        
        target = 'TARGET'
        self.features = ['AMT_CREDIT', 'AMT_INCOME_TOTAL', 'AMT_ANNUITY', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'NAME_EDUCATION_TYPE']
        
        df = df.dropna(subset=[target])
        X = df[self.features]
        y = df[target]
        
        numeric_features = ['AMT_CREDIT', 'AMT_INCOME_TOTAL', 'AMT_ANNUITY', 'DAYS_BIRTH', 'DAYS_EMPLOYED']
        categorical_features = ['NAME_EDUCATION_TYPE']
        
        # Check multicollinearity
        calculate_vif(df, numeric_features)
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])
            
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])
            
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)])
                
        clf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', lgb.LGBMClassifier(random_state=42))])
                              
        print("Training model...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf.fit(X_train, y_train)
        print(f"Model trained. Accuracy: {clf.score(X_test, y_test):.3f}")
        
        self.model = clf
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(self.model, MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")
        
    def load(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("Model not found. Train it first.")
        self.model = joblib.load(MODEL_PATH)
        
    def predict_risk(self, applicant_data: pd.DataFrame) -> float:
        if self.model is None:
            self.load()
        return self.model.predict_proba(applicant_data)[:, 1]

if __name__ == '__main__':
    adapter = RiskModelAdapter()
    try:
        adapter.train()
    except Exception as e:
        print(f"Error training: {e}")
