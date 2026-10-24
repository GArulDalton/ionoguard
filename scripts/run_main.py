"""Main comparison: 10 schemes x 3 scenarios x 3 held-out test nights (Tables 3-4, Figures 6-7)."""
import pickle, json, os, time
from _common import RES, T, QUICK
import sim, run
sim.FADE = sim.build_fade_table()
SCEN = {'S1': ('strong', 0.8e9, 1000), 'S2': ('strong', 1.99e9, 2000), 'S3': ('moderate', 0.8e9, 3000)}
for sc, (inten, f, b) in SCEN.items():
    models = pickle.load(open(os.path.join(RES, f'models_{sc}.pkl'), 'rb'))
    for s in range(1 if QUICK else 3):
        out = os.path.join(RES, 'nights', f'{sc}_night{s}.json')
        if os.path.exists(out):
            continue
        t0 = time.time()
        res = run.run_night(run.Night(b + 101 + s, inten, f, T=T), models)
        json.dump({p: {k: float(v) for k, v in d.items()} for p, d in res.items()}, open(out, 'w'))
        print(sc, 'night', s, round(time.time() - t0), 's', flush=True)
