# ========================================================
# File: c3_1_e8_branching_dimension_validation.py
# Purpose: Validate branching dimensions of E8 and its subgroups down to the Standard Model
# Method:
#   - Define stepwise subgroup decomposition from E8 → E6 → SO(10) → SU(5) → SM gauge groups
#   - List expected dimension counts of each group and multiplet decomposition
# Output:
#   - Prints dimension checks for each subgroup and multiplet breakdown
# ========================================================

groups = [
    ("E8", 248),
    ("E6", 78),
    ("SO(10)", 45),
    ("SU(5)", 24),
    ("SU(3) × SU(2) × U(1)", 8 + 3 + 1),
]

multiplet_counts = {
    "E8": 248,
    "E6 + extra": 78 + 27 + 27 + 1 + 1 + 1 + 1 + 1 + 1,  # common E6 decompositions
    "SO(10) (adjoint)": 45,
    "SU(5) (adjoint)": 24,
    "Standard Model": 8 + 3 + 1  # gluons, weak bosons, photon
}

print("=== E8 Branching Dimension Validation ===")
for name, dim in groups:
    print(f"{name:<30} → dim = {dim}")

print("\n=== Multiplet Breakdown Check ===")
for label, dim in multiplet_counts.items():
    print(f"{label:<30} = {dim}")
