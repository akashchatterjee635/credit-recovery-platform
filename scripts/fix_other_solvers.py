import re

# Patch binary_search_solver and dice_solver to accept target_threshold kwarg gracefully
for path in ["backend/engine/solvers/binary_search_solver.py", "backend/engine/solvers/dice_solver.py"]:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    # Find the generate_recourse signature and add target_threshold if missing
    content = re.sub(
        r"def generate_recourse\(self, applicant: pd\.DataFrame\)",
        "def generate_recourse(self, applicant: pd.DataFrame, target_threshold: float = None, **kwargs)",
        content
    )
    # Replace self.threshold uses with effective_threshold
    content = re.sub(
        r"(def generate_recourse[^\n]+\n)",
        r"\1        _threshold = target_threshold if target_threshold is not None else self.threshold\n",
        content,
        count=1
    )
    # Replace self.threshold in method body (only inside generate_recourse, approximately)
    content = content.replace("self.threshold", "_threshold")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"OK  {path}")
