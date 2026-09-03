import os
import re

rm_path = "backend/models/risk_model.py"
with open(rm_path, "r") as f:
    rm_content = f.read()

save_logic = """
        # Save reference datasets for consistency
        os.makedirs('data', exist_ok=True)
        # We save the un-transformed features for the reference sets so they can be fed directly to the API
        X_trainval_orig = df.loc[X_trainval.index]
        X_train_orig = X_trainval_orig.loc[X_train.index]
        X_cal_orig = X_trainval_orig.loc[X_cal.index]
        X_test_orig = df.loc[X_test.index]
        
        X_train_orig.to_csv('data/train_reference.csv', index=False)
        X_cal_orig.to_csv('data/cal_reference.csv', index=False)
        X_test_orig.to_csv('data/test_reference.csv', index=False)
        
        print('\\nReference datasets saved to data/')
"""
if "Reference datasets saved" not in rm_content:
    rm_content = re.sub(
        r"(joblib\.dump\(\{'pipeline': lr_pipe.*?LR_MODEL_PATH\)\n.*?print\(f'Logistic Regression baseline saved.*?'\)\n)",
        r"\1" + save_logic,
        rm_content, flags=re.DOTALL
    )

with open(rm_path, "w") as f:
    f.write(rm_content)
print("Fixed risk_model.py")
