"""
streamlit_app.py
Web app: fit CAP-FT, EIR-FT and EIR-fPLR performance curves from manufacturer
points and produce IESVE-ready coefficients + plots, all in the browser.

Run locally:   streamlit run streamlit_app.py
"""
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Performance Curve Fitter", layout="wide")
st.title("Performance Curve Fitter")
st.caption("CAP-FT · EIR-FT · EIR-fPLR  —  fit from manufacturer points, get IESVE-ready coefficients.")

# ----------------------------------------------------------------- settings
with st.sidebar:
    st.header("Settings")
    TDATUM = st.radio("Datum (must match IESVE toggle)", [32.0, 0.0],
                      format_func=lambda d: f"{d:g}   ({'°F' if d == 32 else '°C'})")
    c1, c2 = st.columns(2)
    TLET_R = c1.number_input("Rated Tlet", value=44.0, step=1.0)
    TODB_R = c2.number_input("Rated Todb", value=95.0, step=1.0)
    PLR_ORDER = st.radio("EIR-fPLR order", [2, 3], index=1,
                         format_func=lambda o: f"{o}  ({'quadratic' if o == 2 else 'cubic'})")
    st.info("Data stays in your browser session — nothing is stored. "
            "For confidential manufacturer/project data, run locally or deploy privately.")

# ----------------------------------------------------------------- inputs
temp_default = pd.DataFrame({
    "Tlet": [40, 40, 40, 40, 44, 44, 44, 44, 48, 48, 48, 48, 52, 52, 52, 52],
    "Todb": [75, 85, 95, 105] * 4,
    "Capacity": [555, 512, 459, 396, 578, 544, 500, 446,
                 601, 576, 541, 496, 624, 608, 582, 546],
    "kW_per_ton": [0.908, 0.958, 1.048, 1.178, 0.860, 0.910, 1.000, 1.130,
                   0.812, 0.862, 0.952, 1.082, 0.764, 0.814, 0.904, 1.034],
})
plr_default = pd.DataFrame({"PLR": [1.00, 0.90, 0.75, 0.50, 0.25],
                            "EIR_fPLR": [1.000, 0.885, 0.715, 0.470, 0.270]})

ca, cb = st.columns([3, 2])
with ca:
    st.subheader("Temperature grid")
    st.caption("Each row = capacity AND kW/ton at (Tlet, Todb). Feeds CAP-FT and EIR-FT together. "
               "Include the rated point; use ≥3 distinct values of each temperature.")
    temp_df = st.data_editor(temp_default, num_rows="dynamic",
                             use_container_width=True, key="temp")
with cb:
    st.subheader("Part-load points")
    st.caption("EIR multiplier vs PLR (fraction of full-load power), normalized to 1.0 at PLR = 1.")
    plr_df = st.data_editor(plr_default, num_rows="dynamic",
                            use_container_width=True, key="plr")

# ----------------------------------------------------------------- fitting
def _diag(t, p):
    r2 = 1 - np.sum((t - p) ** 2) / np.sum((t - t.mean()) ** 2)
    return r2, float(np.max(np.abs((p - t) / t)) * 100)

def rated_value(Tl, To, val, r1, r2):
    m = (Tl == r1) & (To == r2)
    if m.any():
        return float(val[m][0])
    a, b = Tl - TDATUM, To - TDATUM
    X = np.column_stack([np.ones_like(a), a, a**2, b, b**2, a * b])
    c, *_ = np.linalg.lstsq(X, val, rcond=None)
    ar, br = r1 - TDATUM, r2 - TDATUM
    return float(np.array([1, ar, ar**2, br, br**2, ar * br]) @ c)

def fit_biquad(Tl, To, val, rv):
    a, b = Tl - TDATUM, To - TDATUM
    y = val / rv
    X = np.column_stack([np.ones_like(a), a, a**2, b, b**2, a * b])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    ar, br = TLET_R - TDATUM, TODB_R - TDATUM
    Cn = float(np.array([1, ar, ar**2, br, br**2, ar * br]) @ coef)
    return coef, Cn, _diag(y, X @ coef)

def fit_plr(p, m, order):
    p = np.asarray(p, float); m = np.asarray(m, float)
    X = np.vander(p, order + 1, increasing=True)
    coef, *_ = np.linalg.lstsq(X, m, rcond=None)
    return coef, float(np.sum(coef)), _diag(m, X @ coef)

def biquad_eval(coef, Cn, Tl, To):
    a, b = Tl - TDATUM, To - TDATUM
    return (coef[0] + coef[1]*a + coef[2]*a**2
            + coef[3]*b + coef[4]*b**2 + coef[5]*a*b) / Cn

def plr_eval(coef, Cn, p):
    return np.polynomial.polynomial.polyval(p, coef) / Cn

# ----------------------------------------------------------------- run
temp_df = temp_df.dropna()
plr_df = plr_df.dropna()
Tl = temp_df["Tlet"].to_numpy(float); To = temp_df["Todb"].to_numpy(float)
cap = temp_df["Capacity"].to_numpy(float); kwt = temp_df["kW_per_ton"].to_numpy(float)

problems = []
if len(temp_df) < 6:
    problems.append("Need at least 6 temperature points (a 3×3 grid is the safe minimum).")
if len(set(Tl)) < 3 or len(set(To)) < 3:
    problems.append("Need at least 3 distinct values of *each* temperature, or the fit is singular.")
if len(plr_df) < PLR_ORDER + 1:
    problems.append(f"Need at least {PLR_ORDER + 1} part-load points for an order-{PLR_ORDER} fit.")

if problems:
    for p in problems:
        st.warning(p)
    st.stop()

cap_rv = rated_value(Tl, To, cap, TLET_R, TODB_R)
kwt_rv = rated_value(Tl, To, kwt, TLET_R, TODB_R)
cap_coef, cap_Cn, cap_d = fit_biquad(Tl, To, cap, cap_rv)
eir_coef, eir_Cn, eir_d = fit_biquad(Tl, To, kwt, kwt_rv)
plr_coef, plr_Cn, plr_d = fit_plr(plr_df["PLR"], plr_df["EIR_fPLR"], PLR_ORDER)

# ----------------------------------------------------------------- results
st.divider()
st.subheader("Fitted coefficients")
labels = ["C00", "C10", "C20", "C01", "C02", "C11"]

def show_temp(name, coef, Cn, diag):
    df = pd.DataFrame({
        "coef": labels,
        "IESVE form (enter + let IESVE set Cnorm)": coef,
        "EnergyPlus form (normalization baked in)": coef / Cn,
    })
    st.markdown(f"**{name}**  —  R² = {diag[0]:.5f}, max error = {diag[1]:.2f}%, Cnorm = {Cn:.6f}")
    st.dataframe(df, hide_index=True, use_container_width=True)

g1, g2 = st.columns(2)
with g1:
    show_temp("CAP-FT (capacity)", cap_coef, cap_Cn, cap_d)
with g2:
    show_temp("EIR-FT (energy input)", eir_coef, eir_Cn, eir_d)

st.markdown(f"**EIR-fPLR**  —  R² = {plr_d[0]:.5f}, max error = {plr_d[1]:.2f}%, "
            f"Cnorm = {plr_Cn:.6f}  (order {PLR_ORDER})")
st.dataframe(pd.DataFrame({"term": [f"PLR^{i}" for i in range(len(plr_coef))],
                           "coef": plr_coef}), hide_index=True)

st.caption(f"Valid range — Tlet [{Tl.min():.0f}, {Tl.max():.0f}], "
           f"Todb [{To.min():.0f}, {To.max():.0f}], PLR [{plr_df['PLR'].min():.2f}, 1.0]. "
           "Do not trust the curves outside this box.")

# copy-paste block
txt = ["CAP-FT (IESVE form):"] + [f"  {l} = {v:.8f}" for l, v in zip(labels, cap_coef)]
txt += ["EIR-FT (IESVE form):"] + [f"  {l} = {v:.8f}" for l, v in zip(labels, eir_coef)]
txt += ["EIR-fPLR:"] + [f"  a{i} = {v:.8f}" for i, v in enumerate(plr_coef)]
st.code("\n".join(txt), language="text")

# ----------------------------------------------------------------- plots
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
to_grid = np.linspace(To.min() - 3, To.max() + 3, 100)
tlets = sorted(set(Tl))
colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(tlets)))
for tl, c in zip(tlets, colors):
    ax[0].plot(to_grid, biquad_eval(cap_coef, cap_Cn, tl, to_grid), color=c, lw=1.6, label=f"{tl:.0f}° LWT")
    ax[1].plot(to_grid, biquad_eval(eir_coef, eir_Cn, tl, to_grid), color=c, lw=1.6)
    sel = Tl == tl
    ax[0].scatter(To[sel], cap[sel] / cap_rv, color=c, s=26, zorder=5, edgecolor="k", linewidth=0.4)
    ax[1].scatter(To[sel], kwt[sel] / kwt_rv, color=c, s=26, zorder=5, edgecolor="k", linewidth=0.4)
for a, t in zip(ax[:2], ["CAP-FT (capacity)", "EIR-FT (energy input)"]):
    a.axhline(1, color="0.6", ls=":", lw=1); a.axvline(TODB_R, color="0.6", ls=":", lw=1)
    a.set_xlabel("Outdoor / condenser dry-bulb"); a.set_ylabel("Multiplier (× rated)")
    a.set_title(t); a.grid(alpha=0.25)
ax[0].legend(fontsize=8, frameon=False)
pg = np.linspace(0.2, 1.0, 100)
ax[2].plot(pg, plr_eval(plr_coef, plr_Cn, pg), color="#D85A30", lw=1.8)
ax[2].scatter(plr_df["PLR"], plr_df["EIR_fPLR"], color="#D85A30", s=28, zorder=5, edgecolor="k", linewidth=0.4)
ax[2].axhline(1, color="0.6", ls=":", lw=1); ax[2].axvline(1, color="0.6", ls=":", lw=1)
ax[2].set_xlabel("Part-load ratio"); ax[2].set_ylabel("EIR multiplier"); ax[2].set_title("EIR-fPLR (part load)")
ax[2].grid(alpha=0.25)
fig.tight_layout()
st.pyplot(fig)

buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
st.download_button("Download plot (PNG)", buf.getvalue(), "curves.png", "image/png")
