"""One S1 test night with per-epoch traces (Figures 2 and 8)."""
import pickle, os, numpy as np
from _common import RES, T
import sim, run
sim.FADE = sim.build_fade_table()
models = pickle.load(open(os.path.join(RES, 'models_S1.pkl'), 'rb'))
n = run.Night(1101, 'strong', 0.8e9, T=T)
res = run.run_night(n, models, policies=['A3-OLLA', 'IONOGUARD', 'IG-noSel'], trace=True)
t0 = min(9000, T - 200)
xs = np.linspace(-1100, 1100, 221); ys = np.linspace(-900, 900, 181); XX, YY = np.meshgrid(xs, ys)
S = n.field.s4(XX.ravel(), YY.ravel(), np.full(XX.size, t0)).reshape(XX.shape)
tracks = {}
for dt in range(120):
    g = n.geo[t0 + dt]
    for i, s in enumerate(g['ids']):
        tracks.setdefault(int(s), []).append((g['x'][i], g['y'][i], g['s4'][i]))
pickle.dump(dict(res=res, S=S, xs=xs, ys=ys, tracks=tracks, t0=t0), open(os.path.join(RES, 'trace.pkl'), 'wb'))
print('trace done')
