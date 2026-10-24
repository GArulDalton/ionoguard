"""Per-epoch runtime of IONOGUARD (Section 7.6)."""
import time, pickle, json, os, numpy as np
from _common import RES
import sim, run
sim.FADE = sim.build_fade_table()
m = pickle.load(open(os.path.join(RES, 'models_S1.pkl'), 'rb'))
n = run.Night(1101, 'strong', 0.8e9, T=3600)
orig = sim.DriftMap.update; tu = []
def upd(self, b):
    t0 = time.perf_counter(); orig(self, b); tu.append(time.perf_counter() - t0)
sim.DriftMap.update = upd
t0 = time.perf_counter(); run.run_night(n, m, policies=['IONOGUARD'], collect_cov=False); tot = time.perf_counter() - t0
json.dump({'ms_epoch': 1000 * (tot - sum(tu)) / n.T, 'ms_drift': 1000 * float(np.mean(sorted(tu)[len(tu) // 2:])), 'n': n.T},
          open(os.path.join(RES, 'timing.json'), 'w'))
print(open(os.path.join(RES, 'timing.json')).read())
