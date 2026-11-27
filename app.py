import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from ndt_core import DEFAULTS, randn_lcg, run_sim

st.set_page_config(page_title="UPV + Rebound + GPR Simulator", layout="wide")
st.title("Combined UPV + Rebound Hammer + GPR Health-Monitoring Simulator")
st.caption("Synthetic concrete member 2.0 m x 1.0 m x 0.20 m. Idealised model: not validated against real test data.")


@st.cache_data(show_spinner="Generating noise...")
def cached_noise(n, seed):
    return randn_lcg(n, seed)


with st.sidebar:
    st.header("Inputs")
    st.subheader("Defect (ground truth)")
    xd = st.slider("Defect x (m)", 0.0, 2.0, DEFAULTS['xd'], 0.1)
    yd = st.slider("Defect y (m)", 0.0, 1.0, DEFAULTS['yd'], 0.1)
    sd_def = st.slider("Defect radius s (m)", 0.05, 0.50, DEFAULTS['sd_def'], 0.05)
    red_def = st.slider("Strength loss at centre (-)", 0.0, 0.9, DEFAULTS['red_def'], 0.05)
    fc0 = st.number_input("Sound strength (MPa)", 10.0, 80.0, DEFAULTS['fc0'], 1.0)
    fc_design = st.number_input("Acceptance limit (MPa)", 5.0, 80.0, DEFAULTS['fc_design'], 1.0)

    st.subheader("Measurement noise")
    sigma_t_us = st.slider("UPV time noise std (us)", 0.0, 3.0, DEFAULTS['sigma_t'] * 1e6, 0.1)
    sigma_R = st.slider("Rebound noise std (R units)", 0.0, 6.0, DEFAULTS['sigma_R'], 0.5)
    sigma_g = st.slider("GPR noise std (-)", 0.0, 0.2, DEFAULTS['sigma_g'], 0.01)

    st.subheader("Fusion")
    w_upv = st.slider("UPV weight in combined strength", 0.0, 1.0, DEFAULTS['w_upv'], 0.05)

    st.subheader("GPR")
    er_true = st.number_input("True permittivity er", 2.0, 15.0, DEFAULTS['er_true'], 0.5)
    er_assumed = st.number_input("Assumed permittivity er", 2.0, 15.0, DEFAULTS['er_assumed'], 0.5)
    d_nom = st.slider("Nominal rebar depth (mm)", 20, 100, 60, 5)
    d_low = st.slider("Bar 2 (shallow) depth (mm)", 20, 100, 40, 5)
    x_v = st.slider("Void x (m)", 0.0, 2.0, DEFAULTS['x_v'], 0.1)
    d_v_mm = st.slider("Void depth (mm)", 80, 180, 120, 10)
    min_cover_mm = st.slider("Minimum cover (mm)", 20, 60, 40, 5)

    st.subheader("Noise seeds")
    seed_upv = st.number_input("UPV seed", 1, 2_000_000_000, DEFAULTS['seed_upv'])
    seed_rh = st.number_input("Rebound seed", 1, 2_000_000_000, DEFAULTS['seed_rh'])
    seed_gpr = st.number_input("GPR seed", 1, 2_000_000_000, DEFAULTS['seed_gpr'])

d_bar = [d_nom / 1000.0] * 5
d_bar[1] = d_low / 1000.0
over = dict(xd=xd, yd=yd, sd_def=sd_def, red_def=red_def, fc0=fc0, fc_design=fc_design,
            sigma_t=sigma_t_us * 1e-6, sigma_R=sigma_R, sigma_g=sigma_g, w_upv=w_upv,
            er_true=er_true, er_assumed=er_assumed, d_bar=tuple(d_bar), x_v=x_v, d_v=d_v_mm / 1000.0,
            min_cover=min_cover_mm / 1000.0, seed_upv=int(seed_upv), seed_rh=int(seed_rh), seed_gpr=int(seed_gpr))

try:
    r = run_sim(over, noise_fn=cached_noise)
except ValueError as e:
    st.error(f"Invalid input: {e}")
    st.stop()

p = r['p']
n_flag = int(r['flag_fc'].sum())
n_low = int((r['cover_est'] < p['min_cover']).sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Strength-deficient points", f"{n_flag} / {r['n_pts']}")
c2.metric("Low-cover bars", f"{n_low} / {r['x_pick'].size}")
c3.metric("GPR void", "detected" if r['void_found'] else "not detected")
c4.metric("Status", "ATTENTION" if (n_flag or n_low or r['void_found']) else "No defect indicated")

# ---------------- plots ----------------
x, y, X, Y, dx, dy = r['x'], r['y'], r['X'], r['Y'], p['dx'], p['dy']
fig1, axs = plt.subplots(2, 2, figsize=(12, 8))
maps = [r['V_meas'], r['R_meas'], r['fc_comb']]
ttl = ['UPV velocity (direct transmission)', 'Rebound number', 'Combined strength estimate']
cbl = ['V (m/s)', 'R (-)', 'f_c (MPa)']
for k in range(3):
    ax = axs.flat[k]
    im = ax.imshow(maps[k], origin='lower', extent=[x[0] - dx / 2, x[-1] + dx / 2, y[0] - dy / 2, y[-1] + dy / 2],
                   aspect='equal', cmap='viridis')
    ax.plot(X[r['flag_fc']], Y[r['flag_fc']], 'ro', markersize=9, mfc='none', mew=1.5, label='f_c < f_c,design')
    fig1.colorbar(im, ax=ax, label=cbl[k])
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_title(ttl[k]); ax.grid(True)
    ax.legend(loc='upper left', fontsize=8)
ax = axs.flat[3]
ft = r['fc_true'].ravel()
ax.plot(ft, r['fc_UPV'].ravel(), 'b.', label='UPV')
ax.plot(ft, r['fc_RH'].ravel(), 'g.', label='Rebound')
ax.plot(ft, r['fc_comb'].ravel(), 'ko', mfc='none', label='Combined')
ax.plot([0, 1.1 * p['fc0']], [0, 1.1 * p['fc0']], 'r--', label='1:1')
ax.set_xlabel('f_c true (MPa)'); ax.set_ylabel('f_c estimated (MPa)')
ax.set_title('Estimated vs true strength'); ax.grid(True); ax.legend()
fig1.tight_layout()

t, x_g, Bbr, v_est = r['t'], r['x_g'], r['Bbr'], r['v_est']
fig2, axs2 = plt.subplots(1, 2, figsize=(12, 5))
ax = axs2[0]
ax.imshow(Bbr, extent=[x_g[0], x_g[-1], v_est * t[-1] / 2, 0], aspect='auto', cmap='gray', vmin=-1, vmax=1)
ax.plot(r['x_pick'], r['d_est'], 'r+', markersize=10, mew=1.5, label='Rebar apex pick')
ax.plot(r['x_v_est'] if r['void_found'] else np.nan, r['d_v_est'] if r['void_found'] else np.nan,
        'cs', markersize=10, mfc='none', mew=1.5, label='Void pick')
ax.set_xlabel('Scan position x (m)'); ax.set_ylabel('Apparent depth (m)')
ax.set_title('GPR B-scan (background removed)'); ax.grid(True); ax.legend(loc='lower right')
ax = axs2[1]
i1 = np.argmin(np.abs(x_g - p['x_v']))
i2 = np.argmin(np.abs(x_g - p['x_bar'][1]))
ax.plot(t * 1e9, Bbr[:, i1], label=f'x = {x_g[i1]:.2f} m (void location)')
ax.plot(t * 1e9, Bbr[:, i2], label=f'x = {x_g[i2]:.2f} m (shallow bar)')
ax.set_xlabel('Two-way time (ns)'); ax.set_ylabel('Amplitude (-)')
ax.set_title('A-scans'); ax.grid(True); ax.legend()
fig2.tight_layout()

tab1, tab2, tab3 = st.tabs(["UPV + Rebound", "GPR", "Tables"])
with tab1:
    st.pyplot(fig1)
with tab2:
    st.pyplot(fig2)
with tab3:
    ft_ = r['fc_true']
    rm = lambda A: float(np.sqrt(np.mean((A - ft_) ** 2)))
    summ = pd.DataFrame({
        'Quantity': ['fc true', 'UPV velocity', 'Rebound number', 'fc from UPV', 'fc from rebound', 'fc combined'],
        'Mean': [a.mean() for a in (ft_, r['V_meas'], r['R_meas'], r['fc_UPV'], r['fc_RH'], r['fc_comb'])],
        'Min': [a.min() for a in (ft_, r['V_meas'], r['R_meas'], r['fc_UPV'], r['fc_RH'], r['fc_comb'])],
        'Max': [a.max() for a in (ft_, r['V_meas'], r['R_meas'], r['fc_UPV'], r['fc_RH'], r['fc_comb'])],
        'Unit': ['MPa', 'm/s', '-', 'MPa', 'MPa', 'MPa']}).round(2)
    st.subheader("UPV + rebound summary")
    st.dataframe(summ, hide_index=True)
    st.write(f"RMSE vs true f_c (MPa): UPV {rm(r['fc_UPV']):.2f} | Rebound {rm(r['fc_RH']):.2f} | "
             f"Combined {rm(r['fc_comb']):.2f}")
    names = ['Excellent', 'Good', 'Medium', 'Doubtful']
    st.write("UPV quality classes: " + ", ".join(f"{names[k-1]} {int((r['upv_cls'] == k).sum())}" for k in range(1, 5)))

    idx = np.flatnonzero(r['flag_fc'].ravel(order='F'))
    flagged = pd.DataFrame({
        'x (m)': X.ravel(order='F')[idx], 'y (m)': Y.ravel(order='F')[idx],
        'fc_comb (MPa)': r['fc_comb'].ravel(order='F')[idx], 'fc_true (MPa)': ft_.ravel(order='F')[idx]}).round(2)
    st.subheader("Flagged points")
    st.dataframe(flagged, hide_index=True)

    st.subheader(f"GPR rebar picks (er assumed = {p['er_assumed']:.1f}, v = {v_est * 1e-9:.4f} m/ns)")
    gpr = pd.DataFrame({
        'Bar': np.arange(1, r['x_pick'].size + 1), 'x (m)': r['x_pick'], 't (ns)': r['t_pick'] * 1e9,
        'd_est (mm)': r['d_est'] * 1e3, 'd_true (mm)': r['d_tr'] * 1e3,
        'err (mm)': (r['d_est'] - r['d_tr']) * 1e3, 'cover (mm)': r['cover_est'] * 1e3,
        'Status': np.where(r['cover_est'] < p['min_cover'], 'LOW COVER', 'OK')}).round(3)
    st.dataframe(gpr, hide_index=True)
    if r['void_found']:
        st.write(f"Void/anomaly: YES at x = {r['x_v_est']:.2f} m, depth = {r['d_v_est'] * 1e3:.0f} mm "
                 f"(true x = {p['x_v']:.2f} m, depth = {p['d_v'] * 1e3:.0f} mm), amplitude {r['Amin']:.2f}.")
    else:
        st.write(f"Void/anomaly: not detected (min amplitude {r['Amin']:.2f} > {p['void_thr']:.2f}).")

with st.expander("Model limitations"):
    st.markdown(
        "- Synthetic, idealised model: no coupling, roughness, moisture, carbonation or rebar effects on UPV.\n"
        "- Strength curves and fusion weight are assumed placeholders; real use needs core calibration.\n"
        "- GPR is 2D with point-like targets and constant permittivity; one scan line only.\n"
        "- Estimates use the same curves that generated the data, so accuracy is optimistic.")
plt.close('all')
