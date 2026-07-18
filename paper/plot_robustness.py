"""Figure: Robustness composite (a) per-step mix accuracy (b) buffer sensitivity on CSTNET-TLS1.3."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
matplotlib.rcParams.update({'font.size': 8, 'font.family': 'sans-serif'})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.6))

# --- Panel (a): Per-step mix accuracy ---
steps = [1, 2, 3, 4, 5]
rar     = [93.42, 92.70, 90.13, 86.89, 84.64]
adapter = [91.30, 90.80, 83.50, 78.40, 74.80]
ft      = [75.58, 72.73, 39.51, 28.16, 22.00]

ax1.plot(steps, rar,     'o-',  color='#228833', lw=2, ms=5, label='RAR', zorder=5)
ax1.plot(steps, adapter, 'D-',  color='#4477AA', lw=2, ms=5, label='Adapter', zorder=4)
ax1.plot(steps, ft,      's--', color='#999999', lw=1.8, ms=5, label='Fine-tune', zorder=3)

ax1.set_xlabel('Incremental step', fontsize=9)
ax1.set_ylabel('Mix accuracy (%)', fontsize=9)
ax1.set_xticks(steps)
ax1.set_ylim(20, 100)
ax1.legend(fontsize=7.5, frameon=True, edgecolor='#cccccc')
ax1.grid(True, alpha=0.25, lw=0.5)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_title('(a) Per-step mix accuracy', fontsize=9, fontweight='bold', loc='left')

# --- Panel (b): Buffer sensitivity ---
labels = ['0.5', '1', '2', '5']
rar_b    = [11.82, 8.49, 9.19, 5.98]
adp_b    = [22.70, 16.40, 11.70, 7.97]
x_b      = range(len(labels))

ax2.plot(x_b, rar_b, 'o-', color='#228833', lw=2, ms=5, label='RAR', zorder=5)
ax2.plot(x_b, adp_b, 'D-', color='#4477AA', lw=2, ms=5, label='Adapter', zorder=4)

ax2.set_xlabel('Memory ratio (%)', fontsize=9)
ax2.set_ylabel('Forgetting (%)', fontsize=9)
ax2.set_xticks(x_b)
ax2.set_xticklabels(labels)
ax2.set_ylim(0, 28)
ax2.legend(fontsize=7.5, frameon=True, edgecolor='#cccccc')
ax2.grid(True, alpha=0.25, lw=0.5)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_title('(b) Forgetting vs memory ratio', fontsize=9, fontweight='bold', loc='left')

plt.tight_layout()
plt.savefig('figures/robustness.pdf', dpi=200, bbox_inches='tight')
plt.savefig('figures/robustness.png', dpi=200, bbox_inches='tight')
print('Saved figures/robustness.pdf')
