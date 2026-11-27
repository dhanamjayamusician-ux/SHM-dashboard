"""ndt_core.py - simulation core (same equations/steps as ndt_shm_sim.py), returns results as a dict."""
import numpy as np

DEFAULTS = dict(
    Lx=2.0, Ly=1.0, H=0.20, dx=0.2, dy=0.2,
    fc0=35.0, fc_design=30.0, xd=1.2, yd=0.6, sd_def=0.15, red_def=0.6,
    V_ref1=4200.0, fc_ref1=35.0, V_ref2=3500.0, fc_ref2=17.5,
    sigma_t=0.5e-6, V_class=(3000.0, 3500.0, 4500.0),
    kR=0.0225, sigma_R=2.0, R_lim=(10.0, 60.0), w_upv=0.6,
    c0=299792458.0, er_true=7.0, er_assumed=7.0, f_gpr=2.0e9, t_max=4.0e-9, dt=1.0e-11, dx_g=0.02,
    x_bar=(0.2, 0.6, 1.0, 1.4, 1.8), d_bar=(0.06, 0.04, 0.06, 0.06, 0.06), dia_bar=0.016,
    A_bar=1.0, x_v=1.2, d_v=0.12, A_v=-0.6, A_w=-0.4, sigma_g=0.02, min_cover=0.040,
    gate_bar=(0.6e-9, 1.8e-9), gate_void=(1.9e-9, 3.0e-9),
    void_thr=-0.3, void_apex_frac=0.7, peak_frac=0.5, min_sep=0.2,
    seed_upv=12345, seed_rh=67890, seed_gpr=24680,
)


def randn_lcg(n, seed):
    """Portable N(0,1)-like noise (Park-Miller LCG, Irwin-Hall sum of 12)."""
    m = 2147483647
    a = 16807
    s = seed
    z = np.zeros(n)
    for i in range(n):
        acc = 0.0
        for _ in range(12):
            s = (a * s) % m
            acc += s / m
        z[i] = acc - 6.0
    return z


def ricker(tau, f):
    a = (np.pi * f * tau) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


def run_sim(overrides=None, noise_fn=randn_lcg):
    p = dict(DEFAULTS)
    if overrides:
        p.update(overrides)
    g = lambda k: p[k]
    Lx, Ly, H, dx, dy = g('Lx'), g('Ly'), g('H'), g('dx'), g('dy')
    x_bar = np.array(g('x_bar'), dtype=float)
    d_bar = np.array(g('d_bar'), dtype=float)
    c0, dt, t_max, dx_g = g('c0'), g('dt'), g('t_max'), g('dx_g')
    gate_bar, gate_void = g('gate_bar'), g('gate_void')

    # ---------------- validation (raises ValueError) ----------------
    if min(Lx, Ly, H, dx, dy, dx_g, dt, t_max) <= 0:
        raise ValueError("Geometry, spacing and time inputs must be > 0.")
    if g('fc_ref1') <= 0 or g('fc_ref2') <= 0 or g('V_ref1') == g('V_ref2'):
        raise ValueError("Calibration points invalid.")
    b_c = np.log(g('fc_ref1') / g('fc_ref2')) / (g('V_ref1') - g('V_ref2'))
    if b_c <= 0:
        raise ValueError("Calibration must give strength increasing with velocity.")
    a_c = g('fc_ref1') * np.exp(-b_c * g('V_ref1'))
    if not (0 <= g('red_def') < 1) or g('fc0') <= 0 or g('sd_def') <= 0:
        raise ValueError("Defect inputs invalid (0 <= loss < 1, fc0 > 0, radius > 0).")
    if g('er_true') <= 1 or g('er_assumed') <= 1:
        raise ValueError("Relative permittivity must be > 1.")
    if x_bar.size != d_bar.size:
        raise ValueError("x_bar and d_bar must have the same size.")
    if np.any(x_bar < 0) or np.any(x_bar > Lx) or g('x_v') < 0 or g('x_v') > Lx:
        raise ValueError("Rebar/void positions must lie within 0..Lx.")
    if np.any(d_bar <= g('dia_bar') / 2) or g('d_v') <= 0 or g('d_v') >= H:
        raise ValueError("Rebar depth must exceed bar radius; void depth must be inside the member.")
    if not (0 <= g('w_upv') <= 1):
        raise ValueError("w_upv must be within [0, 1].")
    if 2 * H / (c0 / np.sqrt(g('er_true'))) >= t_max or gate_void[1] > t_max:
        raise ValueError("Time window too short for back wall / gates.")

    # ---------------- grid and ground truth ----------------
    nx = int(round(Lx / dx)) + 1
    ny = int(round(Ly / dy)) + 1
    n_pts = nx * ny
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)
    r2 = (X - g('xd')) ** 2 + (Y - g('yd')) ** 2
    fc_true = g('fc0') * (1 - g('red_def') * np.exp(-r2 / (2 * g('sd_def') ** 2)))
    V_true = np.log(fc_true / a_c) / b_c
    R_true = np.sqrt(fc_true / g('kR'))

    # ---------------- UPV ----------------
    noise_t = noise_fn(n_pts, g('seed_upv')).reshape((ny, nx), order='F')
    t_meas = H / V_true + g('sigma_t') * noise_t
    if np.any(t_meas <= 0):
        raise ValueError("Non-positive UPV transit time (noise too large).")
    V_meas = H / t_meas
    fc_UPV = a_c * np.exp(b_c * V_meas)

    # ---------------- rebound ----------------
    noise_R = noise_fn(n_pts, g('seed_rh')).reshape((ny, nx), order='F')
    R_lim = g('R_lim')
    R_meas = np.clip(np.floor(R_true + g('sigma_R') * noise_R + 0.5), R_lim[0], R_lim[1])
    fc_RH = g('kR') * R_meas ** 2

    # ---------------- fusion ----------------
    w = g('w_upv')
    fc_comb = w * fc_UPV + (1 - w) * fc_RH
    flag_fc = fc_comb < g('fc_design')
    V_class = g('V_class')
    upv_cls = np.full(V_meas.shape, 4)
    upv_cls[V_meas >= V_class[0]] = 3
    upv_cls[V_meas >= V_class[1]] = 2
    upv_cls[V_meas >= V_class[2]] = 1

    # ---------------- GPR ----------------
    nt = int(round(t_max / dt)) + 1
    t = np.arange(nt) * dt
    ng = int(round(Lx / dx_g)) + 1
    x_g = np.linspace(0, Lx, ng)
    Xg, Tg = np.meshgrid(x_g, t)
    v_true = c0 / np.sqrt(g('er_true'))
    B = np.zeros((nt, ng))
    for k in range(x_bar.size):
        Rk = np.sqrt(d_bar[k] ** 2 + (Xg - x_bar[k]) ** 2)
        B = B + g('A_bar') * (d_bar[k] / Rk) * ricker(Tg - 2 * Rk / v_true, g('f_gpr'))
    Rv = np.sqrt(g('d_v') ** 2 + (Xg - g('x_v')) ** 2)
    B = B + g('A_v') * (g('d_v') / Rv) * ricker(Tg - 2 * Rv / v_true, g('f_gpr'))
    B = B + g('A_w') * ricker(Tg - 2 * H / v_true, g('f_gpr'))
    B = B + g('sigma_g') * noise_fn(nt * ng, g('seed_gpr')).reshape((nt, ng), order='F')

    Bbr = B - B.mean(axis=1, keepdims=True)
    v_est = c0 / np.sqrt(g('er_assumed'))

    g1 = (t >= gate_bar[0] - dt / 2) & (t <= gate_bar[1] + dt / 2)
    tg1 = t[g1]
    sub1 = Bbr[g1, :]
    Amax = sub1.max(axis=0)
    iR = sub1.argmax(axis=0)
    thr = g('peak_frac') * Amax.max()
    cand = np.array([i for i in range(1, ng - 1)
                     if Amax[i] >= Amax[i - 1] and Amax[i] >= Amax[i + 1] and Amax[i] > thr], dtype=int)
    if cand.size:
        cand = cand[np.argsort(-Amax[cand], kind='stable')]
    acc = []
    for i in cand:
        if all(abs(x_g[i] - x_g[j]) >= g('min_sep') - 1e-9 for j in acc):
            acc.append(i)
    acc = np.sort(np.array(acc, dtype=int))
    x_pick = x_g[acc]
    t_pick = tg1[iR[acc]]
    d_est = v_est * t_pick / 2
    cover_est = d_est - g('dia_bar') / 2
    d_tr = np.array([d_bar[np.argmin(np.abs(x_bar - xp))] for xp in x_pick])

    g2 = (t >= gate_void[0] - dt / 2) & (t <= gate_void[1] + dt / 2)
    tg2 = t[g2]
    sub2 = Bbr[g2, :]
    Amin = sub2.min()
    void_found = bool(Amin < g('void_thr'))
    rows = np.flatnonzero((sub2 <= g('void_apex_frac') * Amin).any(axis=1))
    c_i = sub2[rows[0], :].argmin()
    r_i = sub2[:, c_i].argmin()
    x_v_est = x_g[c_i]
    d_v_est = v_est * tg2[r_i] / 2

    return dict(p=p, nx=nx, ny=ny, n_pts=n_pts, x=x, y=y, X=X, Y=Y,
                fc_true=fc_true, V_meas=V_meas, R_meas=R_meas, fc_UPV=fc_UPV, fc_RH=fc_RH,
                fc_comb=fc_comb, flag_fc=flag_fc, upv_cls=upv_cls,
                t=t, x_g=x_g, Bbr=Bbr, v_est=v_est, x_bar=x_bar,
                x_pick=x_pick, t_pick=t_pick, d_est=d_est, d_tr=d_tr, cover_est=cover_est,
                void_found=void_found, Amin=float(Amin), x_v_est=float(x_v_est), d_v_est=float(d_v_est))
