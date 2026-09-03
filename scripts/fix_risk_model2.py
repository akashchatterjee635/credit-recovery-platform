import os

rm_path = "backend/models/risk_model.py"
with open(rm_path, "r") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    new_lines.append(line)
    if "Logistic Regression baseline saved to" in line:
        new_lines.append("""
        os.makedirs('data', exist_ok=True)
        X_trainval_orig = df.loc[X_trainval.index]
        X_train_orig = X_trainval_orig.loc[X_train.index]
        X_cal_orig = X_trainval_orig.loc[X_cal.index]
        X_test_orig = df.loc[X_test.index]
        
        X_train_orig.to_csv('data/train_reference.csv', index=False)
        X_cal_orig.to_csv('data/cal_reference.csv', index=False)
        X_test_orig.to_csv('data/test_reference.csv', index=False)
        print('Reference datasets saved to data/')
""")

with open(rm_path, "w") as f:
    f.writelines(new_lines)
print("Forced append to risk_model.py")
