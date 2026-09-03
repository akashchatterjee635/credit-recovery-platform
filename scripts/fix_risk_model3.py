import os
import re

rm_path = "backend/models/risk_model.py"
with open(rm_path, "r") as f:
    content = f.read()

correct_save_logic = """
        os.makedirs('data', exist_ok=True)
        X_train_orig = df.loc[X_train.index]
        X_cal_orig = df.loc[X_cal.index]
        X_test_orig = df.loc[X_test.index]
        
        X_train_orig.to_csv('data/train_reference.csv', index=False)
        X_cal_orig.to_csv('data/cal_reference.csv', index=False)
        X_test_orig.to_csv('data/test_reference.csv', index=False)
        print('Reference datasets saved to data/')
"""

content = re.sub(r"os\.makedirs\('data', exist_ok=True\).*?print\('Reference datasets saved to data/'\)", correct_save_logic.strip(), content, flags=re.DOTALL)

with open(rm_path, "w") as f:
    f.write(content)
print("Fixed risk_model.py logic")
