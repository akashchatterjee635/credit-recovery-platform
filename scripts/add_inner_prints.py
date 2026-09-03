import os

path = "experiments/09_mpc_benchmark.py"
with open(path, "r") as f:
    content = f.read()

content = content.replace("res = mpc.get_action(current_state, t, T_HORIZON, current_tau)", "print(f'  MPC step {t}', flush=True)\n            res = mpc.get_action(current_state, t, T_HORIZON, current_tau)")

with open(path, "w") as f:
    f.write(content)
print("Added inner step prints")
