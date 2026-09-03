import os

bench_path = "experiments/07_solver_benchmark.py"
with open(bench_path, "r") as f:
    bench_content = f.read()

# Fix bootstrap_ci call
bench_content = bench_content.replace(
    "mean, lower, upper = bootstrap_ci(vals, is_pct=is_pct)",
    "mean, lower, upper = bootstrap_ci(vals)"
)

# Use test_reference
new_load_logic = """
def get_held_out_applicants(adapter, n=1000):
    df = pd.read_csv('data/test_reference.csv')
    df = df.dropna(subset=['TARGET'])
    risks = adapter.predict_risk(df)
    threshold = DEFAULT_REGISTRY.recourse_threshold()
    above_thresh = df[risks > threshold].copy()
    if len(above_thresh) > n:
        sample = above_thresh.sample(n=n, random_state=42)
    else:
        sample = above_thresh
    return sample
"""
import re
bench_content = re.sub(
    r"def get_held_out_applicants\(adapter, n=1000\):.*?return sample",
    new_load_logic.strip(), bench_content, flags=re.DOTALL
)

bench_content = bench_content.replace(
    "trainval, _, _, _ = train_test_split(\n        df_all, df_all['TARGET'].astype(int), test_size=0.2, random_state=42, stratify=df_all['TARGET'].astype(int)\n    )\n    X_train, _, _, _ = train_test_split(\n        trainval, trainval['TARGET'].astype(int), test_size=0.25, random_state=42, stratify=trainval['TARGET'].astype(int)\n    )",
    "X_train = pd.read_csv('data/train_reference.csv')"
)

with open(bench_path, "w") as f:
    f.write(bench_content)
print("Updated 07_solver_benchmark.py")
