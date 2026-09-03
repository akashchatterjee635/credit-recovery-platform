import re

path = "experiments/09_mpc_benchmark.py"
with open(path, "r") as f:
    content = f.read()

content = re.sub(r"N_APPLICANTS = 50\b", "N_APPLICANTS = 10", content)
content = content.replace("for i in range(len(sample_df)):", "for i in range(len(sample_df)):\n        print(f'Processing applicant {i+1}/{len(sample_df)}...', flush=True)")

with open(path, "w") as f:
    f.write(content)
print("Configured benchmark for N=10.")
