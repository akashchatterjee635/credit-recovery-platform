# Fix 7+8: Correct Moderate disturbance parameters in simulator.py
import re

path = "backend/engine/simulator.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Fix 7: income_shock_min/max for Moderate: 0.80-0.90 (10-20% downside, not 20-50%)
content = content.replace(
    "income_shock_min: float = 0.50   # Rare shock: income drops to 50% - 80% of current",
    "income_shock_min: float = 0.80   # Rare shock: income drops to 80% - 90% of current (10-20% downside)"
)
content = content.replace(
    "income_shock_max: float = 0.80",
    "income_shock_max: float = 0.90"
)

# Fix 8: shifted_threshold for Moderate: 0.27 (not 0.25, which is Severe)
content = content.replace(
    "shifted_threshold: float = 0.25  # Shifted approval/recourse threshold",
    "shifted_threshold: float = 0.27  # Moderate shift (0.27); use 0.25 for Severe"
)

# Also update the docstring reference
content = content.replace(
    "income_shock_min: float = 0.50,  # Severe: larger income drop",
    "income_shock_min: float = 0.50,  # Severe: 20-50% income drop"
)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("OK  simulator.py (Moderate disturbance parameters corrected)")
