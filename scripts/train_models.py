"""Stage 1 + Stage 2 (Algorithm 2): train the correction model and calibrate the
regime-conditional conformal risk parameters for each scenario."""
import pickle, os, time
from _common import RES, T, QUICK
import sim, train
sim.FADE = sim.build_fade_table()
SCEN = {'S1': ('strong', 0.8e9, 1000), 'S2': ('strong', 1.99e9, 2000), 'S3': ('moderate', 0.8e9, 3000)}
for sc, (inten, f, b) in SCEN.items():
    out = os.path.join(RES, f'models_{sc}.pkl')
    if os.path.exists(out):
        continue
    t0 = time.time()
    n_tr = 1 if QUICK else 3
    m = train.build_models([b + 1 + i for i in range(n_tr)], [b + 11 + i for i in range(n_tr)], inten, f, T=T)
    pickle.dump(m, open(out, 'wb')); print(sc, 'models done in', round(time.time() - t0), 's', flush=True)
