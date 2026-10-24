"""Sensitivity studies on S1 test night 0: target eps, monitor density M, latency L (Figure 9)."""
import pickle, json, os, time
from _common import RES, T
import sim, run
sim.FADE = sim.build_fade_table()
models = pickle.load(open(os.path.join(RES, 'models_S1.pkl'), 'rb'))
jobs = [(f'eps{e}', {}, {'eps': e}, ['A3-OLLA', 'Risk-ACM', 'IONOGUARD', 'IG-noSel', 'Oracle']) for e in (0.02, 0.10)]
jobs += [(f'M{M}', {'M': M}, {}, ['IONOGUARD', 'IG-noLearn']) for M in (40, 240)]
jobs += [(f'lat{L}', {'latency': L}, {}, ['IONOGUARD', 'IG-noDrift']) for L in (30, 300)]
for name, nkw, rkw, pols in jobs:
    out = os.path.join(RES, 'nights', f'sweep_{name}.json')
    if os.path.exists(out):
        continue
    t0 = time.time()
    res = run.run_night(run.Night(1101, 'strong', 0.8e9, T=T, **nkw), models, policies=pols, **rkw)
    json.dump({p: {k: float(v) for k, v in d.items()} for p, d in res.items()}, open(out, 'w'))
    print(name, round(time.time() - t0), 's', flush=True)
