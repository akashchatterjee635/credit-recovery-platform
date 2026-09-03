import re
path = "experiments/09_mpc_benchmark.py"
with open(path, "r") as f:
    content = f.read()
    
# Change N_APPLICANTS from 50 to 5
content = re.sub(r"N_APPLICANTS\s*=\s*\d+", "N_APPLICANTS = 5", content)

with open(path, "w") as f:
    f.write(content)
print("Updated N_APPLICANTS to 5")
