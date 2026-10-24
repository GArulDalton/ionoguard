"""IONOGUARD simulator: plasma-bubble-aware predictive satellite selection and
link adaptation for equatorial direct-to-cell (D2C) LEO networks."""
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

RE = 6371.0
H_ION = 350.0
MU = 398600.4418
WE = 7.2921159e-5
F_REF = 1.575e9

MCS_SE = np.array([0.15, 0.23, 0.38, 0.60, 0.88, 1.18, 1.48, 1.91, 2.41, 2.73, 3.32])
MCS_REQ = np.array([-6.7, -4.7, -2.3, 0.2, 2.4, 4.3, 5.9, 8.1, 10.3, 11.7, 14.1])
BW = 5e6
SUB = 20
HO_LOSS = 2
K_RICE = 10 ** (10 / 10)


def build_fade_table(n=120000, seed=0):
    rng = np.random.default_rng(seed)
    s4g = np.round(np.arange(0.0, 1.0001, 0.01), 2)
    xg = np.arange(-40.0, 12.01, 0.1)
    K = K_RICE
    los = np.sqrt(K / (K + 1)); sig = np.sqrt(1 / (2 * (K + 1)))
    h = los + sig * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    ric = np.abs(h) ** 2
    cdf = np.zeros((len(s4g), len(xg)))
    for i, s4 in enumerate(s4g):
        if s4 < 1e-3:
            sc = np.ones(n)
        else:
            m = 1.0 / s4 ** 2
            sc = rng.gamma(m, 1.0 / m, n)
        f = 10 * np.log10(sc * ric + 1e-12); f.sort()
        cdf[i] = np.searchsorted(f, xg) / n
    return s4g, xg, cdf


FADE = None


def outage_prob(s4, margin_db):
    s4g, xg, cdf = FADE
    i = np.clip(np.round(np.asarray(s4) * 100).astype(int), 0, 100)
    j = np.clip(np.round((-np.asarray(margin_db) - xg[0]) / 0.1).astype(int), 0, len(xg) - 1)
    return cdf[i, j]


def best_mcs(snr_db, s4, eps):
    margins = snr_db - MCS_REQ
    p = outage_prob(np.full(len(MCS_REQ), s4), margins)
    ok = np.where(p <= eps)[0]
    if not len(ok):
        return -1, 0.0          # infeasible: defer
    k = ok.max()
    return k, p[k]


def exp_goodput(snr_db, s4, eps):
    k, p = best_mcs(snr_db, s4, eps)
    if k < 0:
        return 0.0
    return MCS_SE[k] * (1 - p)


def best_mcs_vec(snr_db, s4):
    snr_db = np.asarray(snr_db)[:, None]; s4 = np.asarray(s4)[:, None]
    return outage_prob(np.broadcast_to(s4, (len(s4), len(MCS_REQ))), snr_db - MCS_REQ[None, :])


def pick_mcs_vec(snr_db, s4, eps):
    P = best_mcs_vec(snr_db, s4)
    ok = P <= eps
    return np.where(ok.any(1), len(MCS_REQ) - 1 - np.argmax(ok[:, ::-1], axis=1), -1)


def lla_to_ecef(lat, lon, alt=0.0):
    la, lo = np.radians(lat), np.radians(lon); r = RE + alt
    return np.array([r * np.cos(la) * np.cos(lo), r * np.cos(la) * np.sin(lo), r * np.sin(la)])


def ecef_to_latlon(p):
    r = np.linalg.norm(p, axis=-1)
    return np.degrees(np.arcsin(p[..., 2] / r)), np.degrees(np.arctan2(p[..., 1], p[..., 0]))


class Constellation:
    def __init__(self, h=550.0, inc=53.0, P=72, S=22, F=1, rng=None):
        self.a = RE + h; self.h = h; self.inc = np.radians(inc)
        self.n = np.sqrt(MU / self.a ** 3)
        p = np.repeat(np.arange(P), S); s = np.tile(np.arange(S), P)
        off_raan = rng.uniform(0, 2 * np.pi) if rng is not None else 0.0
        off_u = rng.uniform(0, 2 * np.pi) if rng is not None else 0.0
        self.raan = 2 * np.pi * p / P + off_raan
        self.u0 = 2 * np.pi * s / S + 2 * np.pi * F * p / (P * S) + off_u
        self.N = P * S

    def ecef(self, t):
        t = np.atleast_1d(t)[:, None]
        u = self.u0[None, :] + self.n * t
        cO, sO = np.cos(self.raan)[None, :], np.sin(self.raan)[None, :]
        cu, su = np.cos(u), np.sin(u); ci, si = np.cos(self.inc), np.sin(self.inc)
        x = self.a * (cu * cO - su * ci * sO); y = self.a * (cu * sO + su * ci * cO); z = self.a * (su * si)
        th = -WE * t
        return np.stack([x * np.cos(th) - y * np.sin(th), x * np.sin(th) + y * np.cos(th), z], -1)


def link_geometry(user_lat, user_lon, const, t_arr, el_mask=25.0, chunk=600):
    u = lla_to_ecef(user_lat, user_lon); up = u / np.linalg.norm(u)
    out = []
    for c0 in range(0, len(t_arr), chunk):
        tt = t_arr[c0:c0 + chunk]
        P = const.ecef(tt)
        d = P - u[None, None, :]
        dn = np.linalg.norm(d, axis=-1)
        sel = np.degrees(np.arcsin((d @ up) / dn))
        for k in range(len(tt)):
            vis = np.where(sel[k] >= el_mask)[0]
            dk = d[k, vis] / dn[k, vis][:, None]
            ud = dk @ u; rI = RE + H_ION
            s = -ud + np.sqrt(ud ** 2 - u @ u + rI ** 2)
            ipp = u[None, :] + s[:, None] * dk
            ilat, ilon = ecef_to_latlon(ipp)
            cosx = np.sum(ipp / rI * dk, axis=1)
            sp = P[k, vis]
            scan = np.degrees(np.arccos(np.sum(sp / np.linalg.norm(sp, axis=1)[:, None] * dk, axis=1)))
            out.append(dict(ids=vis, el=sel[k, vis], rng=dn[k, vis], ilat=ilat, ilon=ilon, cosx=cosx, scan=scan))
    return out


def mean_snr(rng_km, scan_deg, s0):
    return s0 - 20 * np.log10(rng_km / 550.0) + 12 * np.log10(np.cos(np.radians(scan_deg)))


class BubbleField:
    def __init__(self, rng, lat0, lon0, intensity='strong', T=21600):
        self.lat0, self.lon0, self.T = lat0, lon0, T
        nb = rng.poisson(4 if intensity == 'strong' else 2) + 1
        lo, hi = (0.25, 0.70) if intensity == 'strong' else (0.15, 0.45)
        self.t_on = rng.uniform(1800, 10800, nb)
        self.t_dec = rng.uniform(14400, 19800, nb)
        self.x0 = rng.uniform(-2600, 400, nb)
        self.w = rng.uniform(50, 160, nb)
        self.A = rng.uniform(lo, hi, nb)
        self.yc = rng.uniform(-500, 300, nb)
        self.sy = rng.uniform(450, 900, nb)
        self.vk = rng.uniform(0.9, 1.1, nb)
        v0 = rng.uniform(110, 170)
        tt = np.arange(T + 400)
        vm = v0 * (1 - 0.55 * tt / T)
        ou = np.zeros_like(vm); th, sg = 1 / 900.0, 18.0
        for i in range(1, len(tt)):
            ou[i] = ou[i - 1] - th * ou[i - 1] + sg * np.sqrt(2 * th) * rng.standard_normal()
        self.v = np.maximum(vm + ou, 10.0)
        self.X = np.concatenate([[0], np.cumsum(self.v[:-1])]) / 1000.0
        g = gaussian_filter(rng.standard_normal((300, 700)), 4.0)
        g = g / g.std() * 0.35
        self.tex = np.exp(g - 0.35 ** 2 / 2)
        self.tx0, self.ty0, self.dg = -4500.0, -1500.0, 10.0
        self.bg = 0.03

    def xy(self, lat, lon):
        return (RE * np.cos(np.radians(self.lat0)) * np.radians(lon - self.lon0), RE * np.radians(lat - self.lat0))

    def s4(self, x, y, t):
        x = np.asarray(x, float); y = np.asarray(y, float)
        ti = np.clip(np.asarray(t).astype(int), 0, len(self.X) - 1)
        X = self.X[ti]
        val = np.zeros(np.broadcast(x, y, X).shape)
        for k in range(len(self.A)):
            a = self.A[k] / (1 + np.exp(-(np.asarray(t) - self.t_on[k]) / 400.0))
            a = a * np.exp(-np.maximum(0, np.asarray(t) - self.t_dec[k]) / 2400.0)
            xk = self.x0[k] + self.vk[k] * X
            val = val + a * np.exp(-0.5 * ((x - xk) / self.w[k]) ** 2) * np.exp(-0.5 * ((y - self.yc[k]) / self.sy[k]) ** 2)
        xr = (x - X - self.tx0) / self.dg; yr = (y - self.ty0) / self.dg
        tex = map_coordinates(self.tex, [np.broadcast_to(yr, val.shape).ravel(), np.broadcast_to(xr, val.shape).ravel()],
                              order=1, mode='wrap').reshape(val.shape)
        return np.clip(self.bg + val * tex, 0, 1.0)


def slant_s4(s4v, cosx, f):
    return np.clip(s4v * (1.0 / np.maximum(cosx, 0.2)) ** 0.5 * (f / F_REF) ** -1.5, 0, 1.0)


class Monitors:
    def __init__(self, field, rng, M=120, cadence=30, latency=60, radius=1100.0):
        self.f, self.rng, self.M = field, rng, M
        self.cad, self.lat, self.R = cadence, latency, radius
        self.batches = []

    def collect(self, t):
        r = self.R * np.sqrt(self.rng.uniform(0, 1, self.M)); a = self.rng.uniform(0, 2 * np.pi, self.M)
        x, y = r * np.cos(a), r * np.sin(a)
        tru = self.f.s4(x, y, np.full(self.M, t))
        meas = np.clip(tru * (1 + 0.10 * self.rng.standard_normal(self.M)) + 0.02 * self.rng.standard_normal(self.M), 0, 1.2)
        self.batches.append((t, x, y, meas))

    def available(self, t, window=300):
        return [b for b in self.batches if b[0] + self.lat <= t and b[0] + self.lat > t - window]


class DriftMap:
    VGRID = np.arange(0, 260, 10.0)

    def __init__(self, bw=60.0, use_drift=True):
        self.bw, self.use_drift = bw, use_drift
        self.v = 100.0 if use_drift else 0.0
        self.last_n = 0; self.obs = None

    @staticmethod
    def _nw(xo, yo, so, to, xq, yq, tq, v, bw):
        dx = xq[:, None] - (xo[None, :] + v * (tq[:, None] - to[None, :]) / 1000.0)
        dy = yq[:, None] - yo[None, :]
        w = np.exp(-0.5 * (dx ** 2 + dy ** 2) / bw ** 2)
        ws = w.sum(1)
        est = (w @ so) / np.maximum(ws, 1e-9)
        return np.where(ws > 1e-3, est, 0.05), ws

    def update(self, batches):
        if not batches:
            self.obs = None; return
        self.obs = (np.concatenate([b[1] for b in batches]), np.concatenate([b[2] for b in batches]),
                    np.concatenate([b[3] for b in batches]), np.concatenate([np.full(len(b[1]), b[0]) for b in batches]))
        if self.use_drift and len(batches) >= 3 and batches[-1][0] != getattr(self, 'tl', -1):
            self.tl = batches[-1][0]
            new, old = batches[-1], batches[:-1]
            xo = np.concatenate([b[1] for b in old]); yo = np.concatenate([b[2] for b in old])
            so = np.concatenate([b[3] for b in old]); to = np.concatenate([np.full(len(b[1]), b[0]) for b in old])
            tq = np.full(len(new[1]), new[0])
            best, bv = 1e9, self.v
            for v in self.VGRID:
                e, ws = self._nw(xo, yo, so, to, new[1], new[2], tq, v, self.bw)
                m = ws > 0.3
                if m.sum() < 10:
                    continue
                err = np.mean((e[m] - new[3][m]) ** 2)
                if err < best:
                    best, bv = err, v
            self.v = 0.7 * self.v + 0.3 * bv
        self.last_n = len(batches)

    def query(self, x, y, t):
        x = np.atleast_1d(x); y = np.atleast_1d(y); t = np.broadcast_to(t, x.shape).astype(float)
        if self.obs is None:
            return np.full(x.shape, 0.05), np.zeros(x.shape)
        xo, yo, so, to = self.obs
        return self._nw(xo, yo, so, to, x, y, t, self.v if self.use_drift else 0.0, self.bw)

    def features(self, x, y, t, t_now, hour):
        est, ws = self.query(x, y, t)
        e_w, _ = self.query(np.atleast_1d(x) - 60, y, t)
        e_e, _ = self.query(np.atleast_1d(x) + 60, y, t)
        lead = np.broadcast_to(t - (self.obs[3].max() if self.obs is not None else t_now), np.shape(est))
        return np.column_stack([est, np.log1p(ws), lead, np.full(len(est), self.v), np.full(len(est), hour),
                                e_w, e_e, np.maximum.reduce([est, e_w, e_e])])
