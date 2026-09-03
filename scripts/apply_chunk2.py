import os

# 1. Update risk_model.py to save splits
rm_path = "backend/models/risk_model.py"
with open(rm_path, "r") as f:
    rm_content = f.read()

# Add saving logic just before "print(f'\nCalibrated model saved"
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
    rm_content = rm_content.replace(
        "print(f'\\nCalibrated model saved to {MODEL_PATH}')",
        save_logic + "\n        print(f'\\nCalibrated model saved to {MODEL_PATH}')"
    )

with open(rm_path, "w") as f:
    f.write(rm_content)
print("Updated risk_model.py")
