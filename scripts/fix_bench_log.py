import re

path = "experiments/09_mpc_benchmark.py"
with open(path, "r") as f:
    content = f.read()

# Add a print statement inside the loop
content = content.replace("for i in range(len(sample_df)):", "for i in range(len(sample_df)):\n        print(f'Processing applicant {i+1}/{len(sample_df)}...')")

with open(path, "w") as f:
    f.write(content)
print("Added progress logging")
