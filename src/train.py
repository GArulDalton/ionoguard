import numpy as np, time
import sim, run
from sklearn.ensemble import HistGradientBoostingRegressor as HGB

EPS_LIST = (0.02, 0.05, 0.10)
LAMS = np.round(np.arange(-0.50, 0.601, 0.01), 2)
BINS = np.array([0.0, 0.15, 0.40, 1.01])


def build_models(train_seeds, cal_seeds, intensity, f, delta=0.02, T=21600):
    D = {'drift': ([], []), 'nodrift': ([], [])}
    for s in train_seeds:
        t0 = time.time(); n = run.Night(s, intensity, f, T=T)
        X, Y, R = run.collect_training(n)
        for k in D: D[k][0].append(X[k]); D[k][1].append(Y)
        print('train night', s, len(Y), round(time.time() - t0, 1), flush=True)
    C = {'drift': ([], []), 'nodrift': ([], [])}; CR = []
    for s in cal_seeds:
        n = run.Night(s, intensity, f, T=T)
        X, Y, R = run.collect_training(n, every=15)
        for k in C: C[k][0].append(X[k]); C[k][1].append(Y)
        CR.append(R); print('cal night', s, len(Y), flush=True)
    models = {}
    for k in D:
        X = np.vstack(D[k][0]); Y = np.concatenate(D[k][1])
        q = HGB(loss='quantile', quantile=1 - delta, max_iter=300, learning_rate=0.08, max_leaf_nodes=31, random_state=0).fit(X, Y)
        p = HGB(loss='squared_error', max_iter=300, learning_rate=0.08, random_state=0).fit(X, Y)
        Xc = np.vstack(C[k][0]); Yc = np.concatenate(C[k][1])
        nC = len(Yc); lvl = min(np.ceil((nC + 1) * (1 - delta)) / nC, 1.0)
        c = np.quantile(Yc - q.predict(Xc), lvl); c_raw = np.quantile(Yc - Xc[:, 0], lvl)
        models[k] = dict(q=q, p=p, c=c, c_raw=c_raw, mae_p=np.mean(np.abs(Yc - p.predict(Xc))),
                         mae_raw=np.mean(np.abs(Yc - Xc[:, 0])), ntrain=len(Y), ncal=nC)
        print(k, 'c=%.4f c_raw=%.4f MAE point=%.4f raw=%.4f' % (c, c_raw, models[k]['mae_p'], models[k]['mae_raw']), flush=True)
    R = np.vstack(CR); h0 = R[:, 1] == 0
    models['lam'] = {}; models['crc_curve'] = {}
    for k in ('drift', 'nodrift'):
        Xc = np.vstack(C[k][0])[h0]; Rc = R[h0]
        preds = {'p': models[k]['p'].predict(Xc), 'raw': Xc[:, 0]}
        ytrue = sim.slant_s4(Rc[:, 7], Rc[:, 4], f)
        for pk, sh in preds.items():
            for eps in EPS_LIST:
                lam, curve = crc_lambda(sh, ytrue, Rc[:, 4], Rc[:, 5], f, eps)
                models['lam'][(k, pk, eps)] = lam; models['crc_curve'][(k, pk, eps)] = curve
                print('CRC', k, pk, eps, 'lambda=', lam, flush=True)
    return models


def crc_lambda(shat_v, ytrue_slant, cosx, snr, f, eps):
    """Mondrian (regime-conditional) conformal risk control."""
    s_pt = sim.slant_s4(np.clip(shat_v, 0, 1), cosx, f)
    b = np.digitize(s_pt, BINS) - 1
    lams, curves = [], []
    for j in range(len(BINS) - 1):
        m = b == j; n = m.sum()
        if n < 50:
            lams.append(0.3); curves.append(None); continue
        rows = []
        for lam in LAMS:
            s_up = sim.slant_s4(np.clip(shat_v[m] + lam, 0, 1), cosx[m], f)
            k = sim.pick_mcs_vec(snr[m], s_up, eps)
            tx = k >= 0
            loss = np.where(tx, sim.outage_prob(ytrue_slant[m], snr[m] - sim.MCS_REQ[np.maximum(k, 0)]), 0.0)
            rows.append((lam, loss.mean(), np.where(tx, sim.MCS_SE[np.maximum(k, 0)], 0).mean(), n))
        c = np.array(rows)
        adj = (n * c[:, 1] + 1.0) / (n + 1)
        lam_star = LAMS[-1]
        for i in range(len(LAMS)):
            if np.all(adj[i:] <= eps):
                lam_star = LAMS[i]; break
        lams.append(lam_star); curves.append(c)
    return np.array(lams), curves


def apply_lambda(shat_v, cosx, f, lams):
    s_pt = sim.slant_s4(np.clip(shat_v, 0, 1), cosx, f)
    b = np.clip(np.digitize(s_pt, BINS) - 1, 0, len(BINS) - 2)
    return shat_v + lams[b]
