import re
path = "experiments/09_mpc_benchmark.py"
with open(path, "r") as f:
    content = f.read()

content = re.sub(r"N_APPLICANTS = 1\b", "N_APPLICANTS = 50", content)
content = content.replace("top of file\n", "")
content = content.replace("print('init adapter...', flush=True)\n    ", "")
content = content.replace("print('loading adapter...', flush=True)\n    ", "")
content = content.replace("print('loading test_df...', flush=True)\n    ", "")
content = content.replace("print('init registry...', flush=True)\n    ", "")
content = content.replace("print('predicting risks...', flush=True)\n    ", "")
content = content.replace("print('starting loop...', flush=True)\n    ", "")

with open(path, "w") as f:
    f.write(content)
