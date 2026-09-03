import os

reqs_path = "requirements.txt"
with open(reqs_path, "r", encoding="utf-8") as f:
    lines = f.read().splitlines()

core_reqs = []
dev_reqs = []

dev_packages = ["pytest", "matplotlib"]

for line in lines:
    line = line.strip()
    if not line: continue
    
    is_dev = False
    for dp in dev_packages:
        if line.startswith(dp):
            is_dev = True
            break
            
    if is_dev:
        dev_reqs.append(line)
    else:
        core_reqs.append(line)

with open(reqs_path, "w", encoding="utf-8") as f:
    f.write("\n".join(core_reqs) + "\n")

with open("requirements-dev.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(dev_reqs) + "\n")

print("Separated requirements")
