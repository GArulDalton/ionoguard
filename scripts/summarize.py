"""Aggregate per-night outputs into results/summary.json and results/extra_sweeps.json."""
import json, glob, os, numpy as np
from _common import RES
N = os.path.join(RES, 'nights')
S = {}
for sc in ('S1', 'S2', 'S3'):
    D = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(N, f'{sc}_night*.json')))]
    if not D:
        continue
    S[sc] = {'n': len(D)}
    for p in D[0]:
        S[sc][p] = {k: [float(np.mean([d[p][k] for d in D])), float(np.std([d[p][k] for d in D]))]
                    for k in ('gp', 'bler', 'dbler', 'defer', 'ho', 'wviol', 'dgp', 'dfrac')}
if S:
    json.dump(S, open(os.path.join(RES, 'summary.json'), 'w'), indent=1)
L = lambda n: json.load(open(os.path.join(N, f'sweep_{n}.json'))) if os.path.exists(os.path.join(N, f'sweep_{n}.json')) else None
base = json.load(open(os.path.join(N, 'S1_night0.json'))) if os.path.exists(os.path.join(N, 'S1_night0.json')) else None
if base and L('eps0.02'):
    E = {'eps': {}, 'lat': {}, 'M': {}}
    for p in ('A3-OLLA', 'Risk-ACM', 'IONOGUARD'):
        E['eps'][p] = {k: [100 * d[p]['bler'], d[p]['gp']] for k, d in (('2.0', L('eps0.02')), ('5.0', base), ('10.0', L('eps0.1')))}
    for p in ('IONOGUARD', 'IG-noDrift'):
        E['lat'][p] = {k: d[p]['gp'] for k, d in (('30', L('lat30')), ('60', base), ('300', L('lat300')))}
    for M in ('40', '240'):
        E['M'][M] = {p: [v['gp'], v['bler']] for p, v in L('M' + M).items()}
    json.dump(E, open(os.path.join(RES, 'extra_sweeps.json'), 'w'), indent=1)
print('summary written')
