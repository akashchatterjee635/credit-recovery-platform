import os

bench_path = "experiments/09_mpc_benchmark.py"
with open(bench_path, "r") as f:
    content = f.read()

content = content.replace("sim.policy_environment.step", "sim.policy_env.step")

with open(bench_path, "w") as f:
    f.write(content)
print("Fixed policy_environment -> policy_env")
