"""Figure 4: tau sensitivity on CSTNET-TLS1.3 (dual y-axis: forget + mix)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 8, 'font.family': 'sans-serif'})

tau    = [0.25, 0.5, 1.0, 2.0, 4.0]
forget = [9.71, 8.55, 7.91, 7.97, 7.78]
mix    = [91.03, 91.60, 91.95, 91.96, 92.06]

fig, ax1 = plt.subplots(figsize=(3.3, 2.5))
c1, c2 = '#EE6677', '#228833'

ax1.plot(tau, forget, 'o-', color=c1, lw=2, ms=5, zorder=5)
ax1.set_xlabel('Temperature ' + r'$\tau$', fontsize=9)
ax1.set_ylabel('Forgetting (%)', fontsize=9, color=c1)
ax1.tick_params(axis='y', labelcolor=c1)
ax1.set_xscale('log', base=2)
ax1.set_xticks(tau)
ax1.set_xticklabels([str(t) for t in tau])
ax1.set_ylim(6, 11)
ax1.grid(True, alpha=0.25, lw=0.5)
ax1.spines['top'].set_visible(False)

ax2 = ax1.twinx()
ax2.plot(tau, mix, 's--', color=c2, lw=2, ms=5, zorder=4)
ax2.set_ylabel('Mix accuracy (%)', fontsize=9, color=c2)
ax2.tick_params(axis='y', labelcolor=c2)
ax2.set_ylim(89, 94)
ax2.spines['top'].set_visible(False)

plt.tight_layout()
plt.savefig('figures/tau_sensitivity.pdf', dpi=200, bbox_inches='tight')
plt.savefig('figures/tau_sensitivity.png', dpi=200, bbox_inches='tight')
print('Saved figures/tau_sensitivity.pdf')
