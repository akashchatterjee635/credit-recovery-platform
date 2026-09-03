import re

path = "experiments/09_mpc_benchmark.py"
with open(path, "r", encoding="utf-8") as f:
    code = f.read()

# Add a patch right after shared_router is created
code = code.replace(
    "training_data=train_df,\n    )",
    "training_data=train_df,\n    )\n    # Disable DiCE to speed up the pre-filter search dramatically\n    shared_router._dice = None"
)

with open(path, "w", encoding="utf-8") as f:
    f.write(code)
print("Disabled DiCE cleanly.")
