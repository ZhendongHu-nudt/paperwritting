"""Figure 5: Per-step mix accuracy on CSTNET-TLS1.3 (T=5, 8 classes/step)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 8, 'font.family': 'sans-serif'})

steps = [1, 2, 3, 4, 5]
rar     = [93.42, 92.70, 90.13, 86.89, 84.64]
adapter = [91.30, 90.80, 83.50, 78.40, 74.80]
ft      = [75.58, 72.73, 39.51, 28.16, 22.00]

fig, ax = plt.subplots(figsize=(3.3, 2.5))
ax.plot(steps, rar,     'o-',  color='#228833', lw=2, ms=5, label='RAR', zorder=5)
ax.plot(steps, adapter, 'D-',  color='#4477AA', lw=2, ms=5, label='Adapter', zorder=4)
ax.plot(steps, ft,      's--', color='#999999', lw=1.8, ms=5, label='Fine-tune', zorder=3)

ax.set_xlabel('Incremental step', fontsize=9)
ax.set_ylabel('Mix accuracy (%)', fontsize=9)
ax.set_xticks(steps)
ax.set_ylim(20, 100)
ax.legend(fontsize=7.5, frameon=True, edgecolor='#cccccc')
ax.grid(True, alpha=0.25, lw=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('figures/per_step_curves.pdf', dpi=200, bbox_inches='tight')
plt.savefig('figures/per_step_curves.png', dpi=200, bbox_inches='tight')
print('Saved figures/per_step_curves.pdf')
