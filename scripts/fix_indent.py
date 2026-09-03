import os

sim_path = "backend/engine/simulator.py"
with open(sim_path, "r") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if line.startswith("        from backend.engine.feature_contract import FEATURE_CONTRACT_V3"):
        line = "        from backend.engine.feature_contract import FEATURE_CONTRACT_V3\n"
    new_lines.append(line)

with open(sim_path, "w") as f:
    f.writelines(new_lines)
print("Indentation fixed.")
