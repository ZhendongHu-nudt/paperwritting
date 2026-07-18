"""Figure 6: Buffer sensitivity on CSTNET-TLS1.3 (RAR vs Adapter)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 8, 'font.family': 'sans-serif'})

labels = ['0.5', '1', '2', '5']   # memory ratio × 100
rar    = [11.82, 8.49, 9.19, 5.98]
adp    = [22.70, 16.40, 11.70, 7.97]
x      = range(len(labels))

fig, ax = plt.subplots(figsize=(3.3, 2.5))
ax.plot(x, rar, 'o-', color='#228833', lw=2, ms=5, label='RAR', zorder=5)
ax.plot(x, adp, 'D-', color='#4477AA', lw=2, ms=5, label='Adapter', zorder=4)

ax.set_xlabel('Memory ratio (%)', fontsize=9)
ax.set_ylabel('Forgetting (%)', fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 28)
ax.legend(fontsize=7.5, frameon=True, edgecolor='#cccccc')
ax.grid(True, alpha=0.25, lw=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('figures/buffer_sensitivity.pdf', dpi=200, bbox_inches='tight')
plt.savefig('figures/buffer_sensitivity.png', dpi=200, bbox_inches='tight')
print('Saved figures/buffer_sensitivity.pdf')
