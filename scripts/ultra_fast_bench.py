import re
path = "experiments/09_mpc_benchmark.py"
with open(path, "r") as f:
    content = f.read()

content = re.sub(r"N_APPLICANTS = 5\b", "N_APPLICANTS = 1", content)
content = content.replace("Processing applicant {i+1}", "Processing applicant {i+1}")

with open(path, "w") as f:
    f.write(content)
