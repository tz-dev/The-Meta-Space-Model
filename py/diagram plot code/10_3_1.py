import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set up figure
fig, ax = plt.subplots(figsize=(8, 6))

# Define box positions
meta_box = [0.1, 0.75, 0.25, 0.12]    # [x, y, width, height]
m4_box = [0.6, 0.75, 0.25, 0.12]
cond_box = [0.25, 0.25, 0.5, 0.15]

# Draw boxes
ax.add_patch(patches.FancyBboxPatch((meta_box[0], meta_box[1]), meta_box[2], meta_box[3],
                                     boxstyle="round,pad=0.02", edgecolor='blue', facecolor='lightblue', lw=2))
ax.text(0.225, 0.81, r"$\mathcal{M}_{\mathrm{meta}}$", fontsize=12, ha='center')

ax.add_patch(patches.FancyBboxPatch((m4_box[0], m4_box[1]), m4_box[2], m4_box[3],
                                     boxstyle="round,pad=0.02", edgecolor='green', facecolor='lightgreen', lw=2))
ax.text(0.725, 0.81, r"$\mathcal{M}_4$", fontsize=12, ha='center')

ax.add_patch(patches.FancyBboxPatch((cond_box[0], cond_box[1]), cond_box[2], cond_box[3],
                                     boxstyle="round,pad=0.03", edgecolor='red', facecolor='mistyrose', lw=2))
ax.text(0.5, 0.31, r"Projectional Stationarity", fontsize=10, ha='center')
ax.text(0.5, 0.28, r"$\delta S_{\mathrm{proj}}[\pi] = 0$", fontsize=12, ha='center')

# Add projection arrow
ax.annotate("", xy=(meta_box[0] + meta_box[2], meta_box[1] + meta_box[3]/2),
            xytext=(m4_box[0], m4_box[1] + m4_box[3]/2),
            arrowprops=dict(arrowstyle="<-", lw=2))
ax.text(0.475, 0.775, r"$\pi$", fontsize=12, ha='center')

# Add dashed lines to condition box
ax.plot([meta_box[0] + meta_box[2]/2, cond_box[0] + cond_box[2]/2],
        [meta_box[1], cond_box[1] + cond_box[3]], 'k--', lw=1)

ax.plot([m4_box[0] + m4_box[2]/2, cond_box[0] + cond_box[2]/2],
        [m4_box[1], cond_box[1] + cond_box[3]], 'k--', lw=1)

# Cleanup
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

plt.tight_layout()
plt.savefig('10_3_1.png', dpi=300, bbox_inches='tight')