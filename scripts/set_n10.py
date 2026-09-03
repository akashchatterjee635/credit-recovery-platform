import re
path = "experiments/09_mpc_benchmark.py"
with open(path, "r") as f:
    content = f.read()

content = re.sub(r"N_APPLICANTS = 50\b", "N_APPLICANTS = 10", content)

with open(path, "w") as f:
    f.write(content)
