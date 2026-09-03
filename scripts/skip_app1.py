import re

path = "experiments/09_mpc_benchmark.py"
with open(path, "r") as f:
    content = f.read()

content = content.replace("sample_df = test_df[above_mask].head(N_APPLICANTS)", "sample_df = test_df[above_mask].iloc[5:15]")
content = re.sub(r"N_APPLICANTS = \d+", "N_APPLICANTS = 10", content)

with open(path, "w") as f:
    f.write(content)
