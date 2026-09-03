import os

sim_path = "backend/engine/simulator.py"
with open(sim_path, "r") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if line.startswith("realized: Dict[str, float] = {}"):
        lines[i] = "        " + line

with open(sim_path, "w") as f:
    f.writelines(lines)
print("Indentation fixed.")
