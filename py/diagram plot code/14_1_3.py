import numpy as np
import matplotlib.pyplot as plt

# Set up figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

# Left: non-quantized phase winding (unstable)
theta = np.linspace(0, 3 * np.pi, 500)
r1 = 1 + 0.1 * np.sin(5 * theta)  # irregular path
x1 = r1 * np.cos(theta)
y1 = r1 * np.sin(theta)

ax1.plot(x1, y1, color='red', lw=2)
ax1.set_title('Unstable Phase Path (Non-Quantized)')
ax1.set_aspect('equal')
ax1.set_xlim([-1.5, 1.5])
ax1.set_ylim([-1.5, 1.5])
ax1.axis('off')
ax1.text(0, -1.2, r"$\oint A_\mu dx^\mu \notin 2\pi \mathbb{Z}$", ha='center', fontsize=12)

# Right: stable quantized loop (2πn)
n = 3
theta_q = np.linspace(0, 2 * np.pi * n, 1000)
r2 = 1.0
x2 = r2 * np.cos(theta_q)
y2 = r2 * np.sin(theta_q)

ax2.plot(x2, y2, color='blue', lw=2)
ax2.set_title('Stable Winding Loop (Quantized)')
ax2.set_aspect('equal')
ax2.set_xlim([-1.5, 1.5])
ax2.set_ylim([-1.5, 1.5])
ax2.axis('off')
ax2.text(0, -1.2, r"$\oint A_\mu dx^\mu = 2\pi n$", ha='center', fontsize=12)

# Annotation
fig.text(0.5, 0.03, 'Topological Quantization as Projectability Criterion (CP3, CP4, CP8)', ha='center', fontsize=10, color='darkred')

plt.tight_layout()
plt.show()
