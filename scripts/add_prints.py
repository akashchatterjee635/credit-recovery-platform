import sys
print("Starting script...", flush=True)
with open("experiments/09_mpc_benchmark.py", "r") as f:
    content = f.read()

content = "import sys\nprint('top of file', flush=True)\n" + content

with open("experiments/09_mpc_benchmark.py", "w") as f:
    f.write(content)
