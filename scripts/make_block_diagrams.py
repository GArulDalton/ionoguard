import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse, Circle, Polygon, Arc
plt.rcParams.update({'font.family': 'serif', 'font.size': 9})
import os
F = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures') + '/'
def box(ax, x, y, w, h, txt, fc, fs=8.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.04,rounding_size=0.12', fc=fc, ec='#333', lw=0.9))
    ax.text(x + w/2, y + h/2, txt, ha='center', va='center', fontsize=fs)
def arr(ax, x1, y1, x2, y2, c='#333', ls='-', lw=1.0):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle='->', lw=lw, color=c, linestyle=ls))

# ---------- Figure 1: system scenario
fig, ax = plt.subplots(figsize=(7.2, 4.0)); ax.axis('off'); ax.set_xlim(0, 12); ax.set_ylim(0, 7)
t = np.linspace(0, np.pi, 200)
ax.fill_between(6 + 9*np.cos(t), -7.6 + 9*np.sin(t), -2, color='#dfeedd', lw=0)          # Earth
ax.plot(6 + 10.6*np.cos(t), -7.6 + 10.6*np.sin(t), c='#9a7bd1', lw=8, alpha=0.25)         # ionosphere shell
ax.text(10.9, 2.55, 'Ionosphere\n(350 km shell)', fontsize=7.5, color='#5b3d99')
ax.plot(6 + 13.4*np.cos(t), -7.6 + 13.4*np.sin(t), c='0.6', lw=0.8, ls='--')             # orbit
ax.text(0.15, 5.0, 'LEO orbit\n550 km', fontsize=7.5, color='0.35')
ax.add_patch(Ellipse((5.0, 2.95), 0.75, 1.2, angle=-10, fc='#c0392b', alpha=0.55, ec='none'))
ax.text(2.6, 2.75, 'plasma bubble\n(drifting east)', fontsize=7.5, color='#922b21')
arr(ax, 5.45, 2.95, 6.0, 2.95, c='#922b21')
sats = [(2.9, 5.55, 'A'), (6.2, 5.8, 'B'), (9.3, 5.45, 'C')]
for x, y, l in sats:
    ax.add_patch(Polygon([[x-0.25, y], [x, y+0.18], [x+0.25, y], [x, y-0.18]], fc='#2c3e50'))
    ax.plot([x-0.5, x+0.5], [y, y], c='#2980b9', lw=3); ax.text(x, y + 0.3, f'Sat {l}', ha='center', fontsize=7.5)
ue = (6.0, 1.45)
ax.add_patch(FancyBboxPatch((ue[0]-0.1, ue[1]-0.2), 0.2, 0.38, boxstyle='round,pad=0.02', fc='k'))
ax.text(ue[0], ue[1]-0.5, 'Handset (UE)', ha='center', fontsize=7.5)
for (x, y, l), c, ls in zip(sats, ['#c0392b', '#27ae60', '0.5'], ['-', '-', ':']):
    ax.plot([ue[0], x], [ue[1] + 0.2, y - 0.2], c=c, lw=1.3, ls=ls)
ax.text(3.0, 4.3, 'link through bubble:\nstrong scintillation', fontsize=7, color='#c0392b')
ax.text(6.35, 4.3, 'predicted clear link', fontsize=7, color='#1e8449')
for x in (8.7, 9.8, 10.9):
    ax.add_patch(Polygon([[x-0.18, 0.95], [x+0.18, 0.95], [x, 1.35]], fc='#f39c12')); 
ax.text(10.2, 0.55, 'GNSS scintillation\nmonitors (S4 reports)', fontsize=7, ha='center')
box(ax, 0.2, 0.15, 3.0, 0.95, 'Gateway / core network:\nIONOGUARD controller', '#fde3e3', fs=7.8)
arr(ax, 9.3, 0.8, 3.25, 0.6, c='#d68910', ls='--'); ax.text(5.9, 0.35, 'delayed S4 reports + ephemeris', fontsize=7, color='#9c640c', ha='center')
arr(ax, 2.6, 1.12, 6.05, 5.55, c='#7d3c98', ls='--'); ax.text(0.3, 1.9, 'selects Sat B and\nthe coding scheme', fontsize=7, color='#7d3c98')
plt.tight_layout(); plt.savefig(F + 'figA_scenario.png', dpi=300); plt.close()

# ---------- Figure 4: offline learning and calibration pipeline
fig, ax = plt.subplots(figsize=(7.2, 3.0)); ax.axis('off'); ax.set_xlim(0, 12); ax.set_ylim(0, 5)
box(ax, 0.1, 3.3, 2.1, 1.2, 'Training nights\n(simulated or\nrecorded data)', '#e8f0fa', fs=7.2)
box(ax, 0.1, 0.5, 2.1, 1.2, 'Calibration nights\n(disjoint seeds)', '#e8f0fa', fs=7.2)
box(ax, 2.7, 3.3, 2.2, 1.2, 'Drift-compensated\nmap + projected IPPs\n→ features ξ', '#fff2d9', fs=7.2)
box(ax, 5.4, 3.3, 2.2, 1.2, 'Fit gradient-boosted\nregressor g_φ\n(squared loss)', '#e3f4e3', fs=7.2)
box(ax, 2.7, 0.5, 2.2, 1.2, 'Point forecasts σ̃_i\n= g_φ(ξ_i)', '#e3f4e3', fs=7.2)
box(ax, 5.4, 0.5, 2.2, 1.2, 'Assign regime g_i\nfrom S4(σ̃_i): quiet,\nmoderate, disturbed', '#fff2d9', fs=7.0)
box(ax, 8.1, 0.5, 1.9, 1.2, 'Per regime:\ndecision loss l_i(λ)\non grid Λ (Eq. 8)', '#fde3e3', fs=7.0)
box(ax, 8.1, 3.3, 1.9, 1.2, 'λ̂_g = smallest λ\nwith adjusted\nrisk ≤ ε (Eq. 9)', '#fde3e3', fs=7.0)
box(ax, 10.4, 1.9, 1.5, 1.2, 'Deploy\n{g_φ, λ̂_g}\nin Algorithm 1', '#f2f2f2', fs=7.2)
arr(ax, 2.2, 3.9, 2.7, 3.9); arr(ax, 4.9, 3.9, 5.4, 3.9); arr(ax, 2.2, 1.1, 2.7, 1.1); arr(ax, 4.9, 1.1, 5.4, 1.1)
arr(ax, 7.6, 1.1, 8.1, 1.1); arr(ax, 9.05, 1.7, 9.05, 3.3); arr(ax, 6.5, 3.3, 3.9, 1.7)
arr(ax, 10.0, 3.9, 10.6, 3.1)
ax.text(5.6, 2.45, 'trained model', fontsize=7.2, style='italic')
ax.text(6.1, 4.75, 'Stage 1: supervised learning', fontsize=8, ha='center', weight='bold')
ax.text(6.1, 0.12, 'Stage 2: regime-conditional conformal risk calibration (Algorithm 2)', fontsize=8, ha='center', weight='bold')
plt.tight_layout(); plt.savefig(F + 'figB_pipeline.png', dpi=300); plt.close()
print('ok')
