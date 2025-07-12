import matplotlib.pyplot as plt
import numpy as np

# Triangle vertices
vertices = {
    "Time": (0.5, 1.0),
    "Mass": (0.0, 0.0),
    "Interaction": (1.0, 0.0)
}

# Definitions for each concept
definitions = {
    "Time": r"$\nabla_\tau S > 0$",
    "Mass": r"$m(\tau) \sim \nabla_\tau S$",
    "Interaction": r"$I_{\mu\nu} := \nabla_\mu \nabla_\nu S$"
}

# Create figure and axis
fig, ax = plt.subplots(figsize=(7, 6))
ax.set_xlim(-0.2, 1.2)
ax.set_ylim(-0.2, 1.2)
ax.axis('off')

# Arrow properties
arrow_props = dict(arrowstyle='->', color='black', linewidth=1.5)

# Triangle structure
points = list(vertices.values())

# Draw arrows (clockwise)
for i in range(3):
    start = points[i]
    end = points[(i + 1) % 3]
    ax.annotate("",
                xy=end, xycoords='data',
                xytext=start, textcoords='data',
                arrowprops=arrow_props)

# Labels and Formulae
ax.text(0.5, 1.12, "Time", fontsize=14, ha='center', va='bottom', fontweight='bold')
ax.text(0.5, 1.09, r"$\nabla_\tau S > 0$", fontsize=14, ha='center', va='top')

ax.text(-0.1, -0.175, "Mass", fontsize=14, ha='left', va='bottom', fontweight='bold')
ax.text(-0.1, -0.05, r"$m(\tau) \sim \nabla_\tau S$", fontsize=14, ha='left', va='top')

ax.text(1.1, -0.175, "Interaction", fontsize=14, ha='right', va='bottom', fontweight='bold')
ax.text(1.1, -0.05, r"$I_{\mu\nu} := \nabla_\mu \nabla_\nu S$", fontsize=14, ha='right', va='top')

# Title
# Layout and display
plt.tight_layout()
plt.savefig('7_1_4.png', dpi=300, bbox_inches='tight')
