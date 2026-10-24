import json, pickle, os, sys, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({'font.family': 'serif', 'font.size': 9})
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = os.path.join(ROOT, 'figures') + '/'
RES = os.path.join(ROOT, 'results') + '/'
POL = ['MaxElev-FM','A3-OLLA','Risk-ACM','IONOGUARD','IG-noDrift','IG-noCRC','IG-Quantile','IG-noLearn','IG-noSel','Oracle']
LBL = {'Oracle': 'Genie-PL'}
SC = {'S1': 'S1: strong EPB, 0.8 GHz', 'S2': 'S2: strong EPB, 1.99 GHz', 'S3': 'S3: moderate EPB, 0.8 GHz'}
S = json.load(open(RES + 'summary.json'))
which = sys.argv[1:] or ['arch', 'trade', 'dist', 'sweep', 'crc', 'trace']

if 'arch' in which:
    fig, ax = plt.subplots(figsize=(7.2, 3.2)); ax.axis('off'); ax.set_xlim(0, 10); ax.set_ylim(0, 4.4)
    def box(x, y, w, h, txt, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.04,rounding_size=0.12', fc=fc, ec='#333', lw=0.9))
        ax.text(x + w/2, y + h/2, txt, ha='center', va='center', fontsize=8.2)
    def arr(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle='->', lw=1.0, color='#333'))
    box(0.1, 3.2, 2.0, 0.9, 'GNSS monitor network\nS4 reports (every 30 s,\n60 s latency)', '#e8f0fa')
    box(0.1, 1.7, 2.0, 0.9, 'Satellite ephemeris\n(orbit propagation)', '#e8f0fa')
    box(0.1, 0.25, 2.0, 0.9, 'Offline: training &\ncalibration nights', '#f2f2f2')
    box(2.7, 3.2, 2.2, 0.9, '(A) Drift-compensated\nkernel map  +  online\nzonal-drift estimate v̂', '#fff2d9')
    box(2.7, 1.7, 2.2, 0.9, 'IPP projection p_n(t+τ)\nfor all candidates,\nτ ∈ {0,5,10,15} s', '#fff2d9')
    box(5.4, 2.45, 2.0, 0.9, '(B) Gradient-boosted\ncorrection g_φ\n→ point forecast σ̃', '#e3f4e3')
    box(5.4, 0.95, 2.0, 0.9, '(C) Mondrian CRC\nσ̃ + λ̂_g per regime g\n(E[failure] ≤ ε)', '#e3f4e3')
    box(7.9, 1.7, 2.0, 1.4, '(D) Receding-horizon\nsatellite selection\n+ MCS / deferral\n(hysteresis κ)', '#fde3e3')
    arr(2.1, 3.65, 2.7, 3.65); arr(2.1, 2.15, 2.7, 2.15); arr(4.9, 3.65, 5.4, 3.0); arr(4.9, 2.15, 5.4, 2.75)
    arr(6.4, 2.45, 6.4, 1.85); arr(7.4, 1.4, 7.9, 2.1)
    arr(2.1, 0.7, 5.4, 1.2); ax.text(3.6, 0.55, 'λ̂_g from calibration set', fontsize=7.5, ha='center', style='italic')
    arr(2.1, 0.85, 5.6, 2.45); ax.text(3.1, 1.25, 'train g_φ', fontsize=7.5, style='italic')
    ax.text(8.9, 1.45, 'serving satellite,\nMCS or defer', ha='center', va='top', fontsize=7.8)
    plt.tight_layout(); plt.savefig(F + 'fig2_arch.png', dpi=220); plt.close()

if 'trade' in which:
    mk = {'MaxElev-FM':'v','A3-OLLA':'s','Risk-ACM':'D','IONOGUARD':'*','IG-noDrift':'1','IG-noCRC':'2','IG-Quantile':'3','IG-noLearn':'4','IG-noSel':'x','Oracle':'P'}
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.7))
    for ax, sc in zip(axs, SC):
        for p in POL:
            c = 'crimson' if p == 'IONOGUARD' else ('tab:blue' if p.startswith('IG') else 'k')
            ax.scatter(S[sc][p]['bler'][0]*100, S[sc][p]['gp'][0], marker=mk[p], s=80 if p == 'IONOGUARD' else 30, c=c, label=LBL.get(p, p), zorder=3)
        ax.axvline(5, ls='--', c='gray', lw=0.8); ax.set_xscale('log'); ax.set_title(SC[sc], fontsize=8.5)
        ax.set_xticks([1, 2, 5, 10, 20]); ax.set_xticklabels(['1', '2', '5', '10', '20']); ax.minorticks_off(); ax.set_xlim(0.9, 22)
        ax.set_xlabel('Block failure rate (%)'); ax.grid(alpha=0.3, which='both')
    axs[0].set_ylabel('Goodput (Mbit/s)')
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=5, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=(0, 0.14, 1, 1)); plt.savefig(F + 'fig4_tradeoff.png', dpi=220); plt.close()

if 'dist' in which:
    main = ['MaxElev-FM','A3-OLLA','Risk-ACM','IG-Quantile','IONOGUARD','Oracle']
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.7)); w = 0.26; x = np.arange(len(main))
    for i, sc in enumerate(SC):
        axs[0].bar(x + (i-1)*w, [S[sc][p]['dgp'][0] for p in main], w, label=sc)
        axs[1].bar(x + (i-1)*w, [min(S[sc][p]['dbler'][0]*100, 20) for p in main], w, label=sc)
    for ax in axs:
        ax.set_xticks(x); ax.set_xticklabels([LBL.get(p, p) for p in main], rotation=30, ha='right', fontsize=7.5); ax.grid(alpha=0.3, axis='y')
    axs[0].set_ylabel('Goodput, disturbed epochs (Mbit/s)', fontsize=8); axs[1].set_ylabel('BFR, disturbed epochs (%)', fontsize=8)
    axs[1].axhline(5, ls='--', c='gray', lw=0.8); axs[1].set_ylim(0, 20); axs[0].legend(fontsize=7.5)
    plt.tight_layout(); plt.savefig(F + 'fig5_disturbed.png', dpi=220); plt.close()

if 'sweep' in which:
    E = json.load(open(RES + 'extra_sweeps.json'))
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.5))
    for p in ['A3-OLLA', 'Risk-ACM', 'IONOGUARD']:
        ks = ['2.0', '5.0', '10.0']; xs = [2, 5, 10]
        axs[0].plot(xs, [E['eps'][p][k][0] for k in ks], 'o-', label=p); axs[1].plot(xs, [E['eps'][p][k][1] for k in ks], 'o-', label=p)
    axs[0].plot([2, 10], [2, 10], 'k--', lw=0.8, label='target'); axs[0].set_xlabel('Target ε (%)'); axs[0].set_ylabel('Achieved BFR (%)'); axs[0].legend(fontsize=6.5)
    axs[1].set_xlabel('Target ε (%)'); axs[1].set_ylabel('Goodput (Mbit/s)')
    for p, m in [('IONOGUARD', 'o-'), ('IG-noDrift', 's--')]:
        axs[2].plot([30, 60, 300], [E['lat'][p][k] for k in ['30', '60', '300']], m, label=p)
    axs[2].set_xlabel('Monitor data latency L (s)'); axs[2].set_ylabel('Goodput (Mbit/s)'); axs[2].legend(fontsize=7)
    for a in axs: a.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(F + 'fig7_sweeps.png', dpi=220); plt.close()

if 'crc' in which and os.path.exists(RES + 'models_S1.pkl'):
    sys.path.insert(0, '.')
    m = pickle.load(open(RES + 'models_S1.pkl', 'rb'))
    curves = m['crc_curve'][('drift', 'p', 0.05)]; lam = m['lam'][('drift', 'p', 0.05)]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6))
    names = ['G1: S4 < 0.15', 'G2: 0.15–0.40', 'G3: ≥ 0.40']
    for g, c in enumerate(curves):
        if c is None: continue
        n = c[0, 3]; adj = (n*c[:, 1] + 1)/(n + 1)
        axs[0].plot(c[:, 0], adj*100, label=f'{names[g]} (n={int(n)})'); axs[1].plot(c[:, 0], c[:, 2], label=names[g])
        axs[0].axvline(lam[g], ls=':', c=f'C{g}')
    axs[0].axhline(5, ls='--', c='gray'); axs[0].set_xlabel('Inflation λ'); axs[0].set_ylabel('Adjusted calibration risk (%)')
    axs[0].set_yscale('log'); axs[0].legend(fontsize=7); axs[0].grid(alpha=.3, which='both')
    axs[1].set_xlabel('Inflation λ'); axs[1].set_ylabel('Mean scheduled efficiency (b/s/Hz)'); axs[1].grid(alpha=.3); axs[1].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(F + 'fig3_crc.png', dpi=220); plt.close()
    json.dump({'lam': [float(x) for x in lam], 'n': [None if c is None else int(c[0, 3]) for c in curves]}, open(RES + 'crc.json', 'w'))

if 'trace' in which and os.path.exists(RES + 'trace.pkl'):
    T = pickle.load(open(RES + 'trace.pkl', 'rb')); out = {}
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    im = ax.pcolormesh(T['xs'], T['ys'], T['S'], cmap='magma_r', vmin=0, vmax=0.7, shading='auto')
    plt.colorbar(im, ax=ax, label='Vertical S4 at L1 (true field)')
    for sid, tr in T['tracks'].items():
        a = np.array(tr)
        if len(a) < 5: continue
        ax.plot(a[:, 0], a[:, 1], '-', c='deepskyblue', lw=1.4)
        ax.annotate('', xy=(a[-1, 0], a[-1, 1]), xytext=(a[-4, 0], a[-4, 1]), arrowprops=dict(arrowstyle='->', color='deepskyblue', lw=1.2))
    ax.plot(0, 0, 'k^', ms=8); ax.text(15, -60, 'UE (Chennai)', fontsize=8)
    ax.set_xlabel('East (km) at 350-km shell'); ax.set_ylabel('North (km)')
    ax.set_title('EPB field at %02d:%02d LT and 120-s IPP tracks of visible satellites' % (19 + T['t0']//3600, (T['t0'] % 3600)//60), fontsize=8.5)
    ax.set_aspect('equal'); ax.set_xlim(-1100, 1100); ax.set_ylim(-900, 900)
    plt.tight_layout(); plt.savefig(F + 'fig1_snapshot.png', dpi=220); plt.close()
    R = T['res']
    fig, axs = plt.subplots(3, 1, figsize=(7.2, 4.8), sharex=True)
    for ax, p, lab in zip(axs, ['A3-OLLA', 'IG-noSel', 'IONOGUARD'], ['A3-OLLA', 'IG-noSel (max-elevation)', 'IONOGUARD']):
        tr = np.array(R[p]['trace']); sl = (tr[:, 0] >= 3600) & (tr[:, 0] < 14400)
        t = 19 + tr[sl, 0]/3600
        ax.fill_between(t, tr[sl, 2], tr[sl, 3], color='0.85', lw=0, label='range over visible satellites')
        ax.plot(t, tr[sl, 1], lw=0.6, c='crimson', label='serving link')
        ax.set_ylabel('slant S4', fontsize=8); ax.set_ylim(0, 1.02)
        fr = tr[sl, 6].sum()/max(tr[sl, 7].sum(), 1); gp = tr[sl, 5].sum()/sl.sum()/1e6
        ax.set_title(f'{lab}: mean serving S4 = {tr[sl, 1].mean():.3f}, BFR = {100*fr:.1f}%, goodput = {gp:.2f} Mbit/s', fontsize=8)
        out[p] = dict(s4=float(tr[sl, 1].mean()), bfr=float(fr), gp=float(gp), s4_all=float(tr[:, 1].mean()))
    axs[0].legend(fontsize=7, loc='upper right'); axs[-1].set_xlabel('Local time (h)')
    plt.tight_layout(); plt.savefig(F + 'fig6_timeseries.png', dpi=220); plt.close()
    json.dump(out, open(RES + 'trace_stats.json', 'w'), indent=1)
print('figs ok', which)
