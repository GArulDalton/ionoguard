import numpy as np
import sim
import train
from sim import MCS_SE, MCS_REQ, SUB, HO_LOSS, BW

USER = (13.08, 80.27)
HSTEPS = np.array([0, 5, 10, 15])
GOLD = 0.6180339887


class Night:
    def __init__(self, seed, intensity='strong', f=0.8e9, s0=9.0, T=21600, M=120, latency=60):
        self.rng = np.random.default_rng(seed)
        self.T, self.f = T, f
        const = sim.Constellation(rng=self.rng)
        self.geo = sim.link_geometry(*USER, const, np.arange(T, dtype=float))
        self.field = sim.BubbleField(self.rng, *USER, intensity=intensity, T=T)
        lens = np.array([len(g['ids']) for g in self.geo])
        self.off = np.concatenate([[0], np.cumsum(lens)])
        ilat = np.concatenate([g['ilat'] for g in self.geo]); ilon = np.concatenate([g['ilon'] for g in self.geo])
        x, y = self.field.xy(ilat, ilon)
        tt = np.repeat(np.arange(T), lens)
        s4v = np.empty(len(x))
        for a in range(0, len(x), 20000):
            s4v[a:a + 20000] = self.field.s4(x[a:a + 20000], y[a:a + 20000], tt[a:a + 20000])
        cosx = np.concatenate([g['cosx'] for g in self.geo])
        s4s = sim.slant_s4(s4v, cosx, f)
        snr = sim.mean_snr(np.concatenate([g['rng'] for g in self.geo]), np.concatenate([g['scan'] for g in self.geo]), s0)
        for k, g in enumerate(self.geo):
            sl = slice(self.off[k], self.off[k + 1])
            g['x'], g['y'], g['s4v'], g['s4'], g['snr'] = x[sl], y[sl], s4v[sl], s4s[sl], snr[sl]
            g['pos'] = {int(i): j for j, i in enumerate(g['ids'])}
        self.U = self.rng.random((T, SUB))
        self.mon = sim.Monitors(self.field, np.random.default_rng(seed + 777), M=M, latency=latency)

    def fades(self, t, sid, s4):
        s4g, xg, cdf = sim.FADE
        u = (self.U[t] + GOLD * sid) % 1.0
        i = int(round(min(max(s4, 0), 1) * 100))
        return np.interp(u, cdf[i], xg)


def future_query(night, t, sats):
    rows = []
    for h in HSTEPS:
        tq = t + h
        if tq >= night.T:
            continue
        g = night.geo[tq]
        for s in sats:
            j = g['pos'].get(int(s))
            if j is not None:
                rows.append((int(s), int(h), g['x'][j], g['y'][j], g['cosx'][j], g['snr'][j], g['s4'][j], g['s4v'][j]))
    return rows


def collect_training(night, every=10):
    maps = {'drift': sim.DriftMap(use_drift=True), 'nodrift': sim.DriftMap(use_drift=False)}
    X = {k: [] for k in maps}; Y = []; R = []
    nb = -1
    for t in range(night.T - 10):
        if t % night.mon.cad == 0:
            night.mon.collect(t)
        av = night.mon.available(t)
        if len(av) != nb or (av and av[-1][0] != getattr(night, '_lb', None)):
            for mp in maps.values():
                mp.update(av)
            nb = len(av); night._lb = av[-1][0] if av else None
        if t % every or not av:
            continue
        rows = future_query(night, t, night.geo[t]['ids'])
        if not rows:
            continue
        r = np.array(rows)
        for k, mp in maps.items():
            X[k].append(mp.features(r[:, 2], r[:, 3], t + r[:, 1], t, t / 3600.0))
        Y.append(r[:, 7]); R.append(r)
    return {k: np.vstack(v) for k, v in X.items()}, np.concatenate(Y), np.vstack(R)


POLICIES = ['MaxElev-FM', 'A3-OLLA', 'Risk-ACM', 'IONOGUARD',
            'IG-noDrift', 'IG-noCRC', 'IG-Quantile', 'IG-noLearn', 'IG-noSel', 'Oracle']


def maxelev_choice(g, cur):
    ids, el = g['ids'], g['el']
    best = ids[np.argmax(el)]
    if cur in g['pos']:
        return int(best) if el.max() > el[g['pos'][cur]] + 10 else cur
    return int(best)


def lookahead_choice(g, cur, rows_s4, eps1, hyst=1.05, add=0.1):
    J = {}
    for sid, h, snr, s4 in rows_s4:
        J[sid] = J.get(sid, 0.0) + sim.exp_goodput(snr, s4, eps1)
    if not J:
        return cur
    best = max(J, key=J.get)
    if cur in g['pos'] and cur in J:
        return best if J[best] > hyst * J[cur] + add else cur
    return best


def run_night(night, models, policies=POLICIES, eps=0.05, delta=0.02, collect_cov=True, trace=False):
    T = night.T
    eps1 = (eps - delta) / (1 - delta)
    S4R2 = (2 * sim.K_RICE + 1) / (sim.K_RICE + 1) ** 2
    st = {p: dict(defer=0, cur=-1, bits=0.0, att=0, fail=0, ho=0, lost=0, dbits=0.0, datt=0, dfail=0, dt=0,
                  olla=0.0, a3cnt=0, s4e=0.3, last=None, wa=0, wf=0, wv=0, wn=0) for p in policies}
    MODE = {'IONOGUARD': 'full', 'IG-noDrift': 'nodrift', 'IG-noCRC': 'nocrc', 'IG-Quantile': 'quant',
            'IG-noLearn': 'nolearn', 'IG-noSel': 'full'}
    EPSP = {p: (eps1 if MODE.get(p) == 'quant' else eps) for p in policies}
    fc = {p: MODE[p] for p in policies if p in MODE}
    maps = {'drift': sim.DriftMap(use_drift=True), 'nodrift': sim.DriftMap(use_drift=False)}
    cov = {p: [0, 0] for p in fc}
    mrng = np.random.default_rng(12345)
    nb, lb = -1, None
    for t in range(T):
        g = night.geo[t]
        if t % night.mon.cad == 0:
            night.mon.collect(t)
        av = night.mon.available(t)
        if av and (len(av) != nb or av[-1][0] != lb):
            for mp in maps.values():
                mp.update(av)
            nb, lb = len(av), av[-1][0]
        disturbed = g['s4'].max() > 0.4
        rows = future_query(night, t, g['ids'])
        r = np.array(rows)
        up = {}
        if len(r) and av and fc:
            Fm = {k: mp.features(r[:, 2], r[:, 3], t + r[:, 1], t, t / 3600.0) for k, mp in maps.items()}
            pred = {}
            pd_ = models['drift']['p'].predict(Fm['drift'])
            lam = models['lam']
            pred['full'] = train.apply_lambda(pd_, r[:, 4], night.f, lam[('drift', 'p', eps)])
            pred['nocrc'] = pd_
            if 'IG-noDrift' in fc:
                pnd = models['nodrift']['p'].predict(Fm['nodrift'])
                pred['nodrift'] = train.apply_lambda(pnd, r[:, 4], night.f, lam[('nodrift', 'p', eps)])
            if 'IG-Quantile' in fc:
                pred['quant'] = models['drift']['q'].predict(Fm['drift']) + models['drift']['c']
            pred['nolearn'] = train.apply_lambda(Fm['drift'][:, 0], r[:, 4], night.f, lam[('drift', 'raw', eps)])
            for p, mode in fc.items():
                qv = np.clip(pred[mode], 0, 1)
                up[p] = sim.slant_s4(qv, r[:, 4], night.f)
                if collect_cov:
                    cov[p][0] += np.sum(r[:, 7] <= qv); cov[p][1] += len(qv)
        else:
            for p in fc:
                up[p] = np.ones(len(r))
        meas = None
        if 'A3-OLLA' in policies:
            lin = []
            for j, s in enumerate(g['ids']):
                m = 1 / max(g['s4'][j], 0.05) ** 2
                lin.append(g['snr'][j] + 10 * np.log10(np.mean(mrng.gamma(m, 1 / m, SUB))))
            meas = np.array(lin)
        for p in policies:
            S = st[p]; prev = S['cur']; k = -9
            if p in ('MaxElev-FM', 'Risk-ACM', 'IG-noSel'):
                cur = maxelev_choice(g, prev)
            elif p == 'A3-OLLA':
                b = int(g['ids'][np.argmax(meas)])
                if prev not in g['pos']:
                    cur = b; S['a3cnt'] = 0
                else:
                    S['a3cnt'] = S['a3cnt'] + 1 if meas.max() > meas[g['pos'][prev]] + 3 else 0
                    cur = b if S['a3cnt'] >= 2 else prev
                    if cur != prev:
                        S['a3cnt'] = 0
            elif p == 'Oracle':
                cur = lookahead_choice(g, prev, [(int(a[0]), a[1], a[5], a[6]) for a in rows], eps)
            else:
                cur = lookahead_choice(g, prev, [(int(a[0]), a[1], a[5], s) for a, s in zip(rows, up[p])], EPSP[p])
            ho = (prev != -1 and cur != prev)
            if ho:
                S['ho'] += 1
            S['cur'] = cur
            j = g['pos'][cur]
            snr, s4t = g['snr'][j], g['s4'][j]
            fd = night.fades(t, cur, s4t)
            start = HO_LOSS if ho else 0
            S['lost'] += start
            if p == 'A3-OLLA':
                ok_n = 0; bits = 0.0
                for kk in range(start, SUB):
                    eff = snr - S['olla']
                    idx = np.where(MCS_REQ <= eff)[0]; idx = idx.max() if len(idx) else 0
                    if snr + fd[kk] >= MCS_REQ[idx]:
                        bits += MCS_SE[idx] * BW / SUB; ok_n += 1
                        S['olla'] = max(S['olla'] - 0.5 * eps / (1 - eps), -5)
                    else:
                        S['olla'] = min(S['olla'] + 0.5, 30)
                n = SUB - start; fails = n - ok_n
            else:
                if p == 'MaxElev-FM':
                    idx = np.where(MCS_REQ <= snr - 3.0)[0]; k = idx.max() if len(idx) else 0
                elif p == 'Risk-ACM':
                    if ho:
                        S['s4e'] = 0.3
                    k, _ = sim.best_mcs(snr, min(S['s4e'], 1.0), eps)
                elif p == 'Oracle':
                    k, _ = sim.best_mcs(snr, s4t, eps)
                else:
                    jj = [i for i, a in enumerate(rows) if a[0] == cur and a[1] == 0]
                    k, _ = sim.best_mcs(snr, up[p][jj[0]] if jj else 1.0, EPSP[p])
                if k < 0:
                    S['defer'] += SUB - start; n = 0; fails = 0; bits = 0.0
                else:
                    okv = (snr + fd[start:]) >= MCS_REQ[k]
                    n = SUB - start; fails = int(n - okv.sum()); bits = okv.sum() * MCS_SE[k] * BW / SUB
                if p == 'Risk-ACM':
                    lin = 10 ** (fd / 10)
                    s4t2 = lin.var() / lin.mean() ** 2
                    s4s = np.sqrt(max((1 + s4t2) / (1 + S4R2) - 1, 0))
                    ema = 0.6 * S['s4e'] / 1.15 + 0.4 * s4s if S['last'] is not None else s4s
                    S['s4e'] = 1.15 * max(ema, s4s); S['last'] = s4s
            S['bits'] += bits; S['att'] += n; S['fail'] += fails
            if trace:
                S.setdefault('tr', []).append((t, s4t, g['s4'].min(), g['s4'].max(), k, bits, fails, n))
            S['wa'] += n; S['wf'] += fails
            if (t + 1) % 60 == 0:
                if S['wa']:
                    S['wn'] += 1; S['wv'] += int(S['wf'] / S['wa'] > 2 * eps)
                S['wa'] = S['wf'] = 0
            if disturbed:
                S['dbits'] += bits; S['datt'] += n; S['dfail'] += fails; S['dt'] += 1
    res = {}
    for p in policies:
        S = st[p]
        res[p] = dict(gp=S['bits'] / T / 1e6, bler=S['fail'] / max(S['att'], 1), fpo=S['fail'] / (T * SUB),
                      defer=S['defer'] / (T * SUB), ho=S['ho'] / (T / 60), dgp=S['dbits'] / max(S['dt'], 1) / 1e6,
                      dbler=S['dfail'] / max(S['datt'], 1), dfrac=S['dt'] / T, wviol=S['wv'] / max(S['wn'], 1),
                      cov=(cov[p][0] / cov[p][1]) if p in cov and cov[p][1] else np.nan)
        if trace:
            res[p]['trace'] = S.get('tr', [])
    return res
