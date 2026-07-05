"""
streamlit_app.py — NECB Performance Curve Explorer
Two modes:
  1. NECB curves   — pick equipment, see the code formula + coefficients + graphs
                     (exact coefficients from NECB 2020, Subsection 8.4.5)
  2. Create graphs — enter your own points, get the three fitted graphs
All temperatures in deg F, Imperial units.

Run:  streamlit run streamlit_app.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="NECB Curve Explorer", layout="wide")

st.markdown("""
<style>
.block-container{padding-top:2.2rem; max-width:1150px;}
h1,h2,h3{color:#12303a;}
.hero{background:linear-gradient(90deg,#0f6e56,#1D9E75); color:#fff;
      padding:1.1rem 1.4rem; border-radius:14px; margin-bottom:1.4rem;}
.hero h1{color:#fff; margin:0; font-size:1.7rem;}
.hero p{color:#dff3ec; margin:.25rem 0 0; font-size:.95rem;}
.card{background:#f6f9fb; border:1px solid #e3e8ee; border-radius:14px;
      padding:1rem 1.3rem; margin-bottom:1.1rem;}
.badge{display:inline-block; background:#e1f5ee; color:#0f6e56;
       padding:3px 12px; border-radius:20px; font-size:.78rem; font-weight:600;
       letter-spacing:.02em; margin-bottom:.4rem;}
.coef{display:inline-block; background:#fff; border:1px solid #e3e8ee;
      border-radius:8px; padding:5px 11px; margin:3px 5px 3px 0; font-size:.9rem;
      font-family:ui-monospace,Menlo,monospace;}
.coef b{color:#0f6e56;}
section[data-testid="stSidebar"]{background:#0f2a24;}
section[data-testid="stSidebar"] *{color:#e8f3ee;}
</style>
""", unsafe_allow_html=True)

NECB = {
    "ashp": {
        "name": "Air-source heat pump", "sec": "8.4.5.7",
        "capft": {"kind": "cubic1", "c": [0.2536714, 0.0104351, 0.0001861, -0.0000015]},
        "eirft": {"kind": "cubic1", "c": [2.4600298, -0.0622539, 0.0008800, -0.0000046]},
        "eirplr": {"c": [0.0856522, 0.9388137, -0.1834361, 0.1589702]},
        "xr": (-10.0, 60.0), "xlab": "Outdoor dry-bulb (°F)",
    },
    "dx": {
        "name": "Direct-expansion cooling", "sec": "8.4.5.4",
        "capft": {"c": [0.8740302, -0.0011416, 0.0001711, -0.0029570, 0.0000102, -0.0000592]},
        "eirft": {"c": [-1.0639310, 0.0306584, -0.0001269, 0.0154213, 0.0000497, -0.0002096]},
        "eirplr": {"c": [0.2012301, -0.0312175, 1.9504979, -1.1205105]},
        "lines": [60, 63, 67, 72], "xr": (75.0, 115.0),
        "xlab": "Outdoor dry-bulb (°F)", "linelab": "Entering wet-bulb",
    },
    "chiller": {
        "name": "Electric chiller", "sec": "8.4.5.5",
        "lines": [40, 44, 48], "xr": (65.0, 95.0),
        "xlab": "Condenser water supply (°F)", "linelab": "Chilled water supply",
        "types": {
            "Air-cooled — Scroll": {
                "capft": [0.40070684, 0.01861548, 0.00007199, 0.00177296, -0.00002014, -0.00008273],
                "eirft": [0.99006553, -0.00584144, 0.00016454, -0.00661136, 0.00016808, -0.00022501],
                "eirplr": [0.06369119, 0.58488832, 0.35280274]},
            "Air-cooled — Reciprocating": {
                "capft": [0.57617295, 0.02063133, 0.00007769, -0.00351183, 0.00000312, -0.00007865],
                "eirft": [0.66534403, -0.01383821, 0.00014736, 0.00712808, 0.00004571, -0.00010326],
                "eirplr": [0.1143742, 0.5459334, 0.34229861]},
            "Air-cooled — Screw": {
                "capft": [-0.09464899, 0.0383407, -0.00009205, 0.00378007, -0.00001375, -0.00015464],
                "eirft": [0.013545636, 0.02292946, -0.00016107, -0.00235396, 0.00012991, -0.00018585],
                "eirplr": [0.03648722, 0.73474298, 0.21994748]},
            "Water-cooled — Scroll": {
                "capft": [0.36131454, 0.01855477, 0.00003011, 0.00093592, -0.00001518, -0.00005481],
                "eirft": [1.00121431, -0.01026981, 0.00016703, -0.0128136, 0.00014613, -0.00021959],
                "eirplr": [0.04411957, 0.64036703, 0.31955532]},
            "Water-cooled — Reciprocating": {
                "capft": [0.58531422, 0.01539593, 0.00007296, -0.00212462, -0.00000715, -0.00004597],
                "eirft": [0.46140041, -0.0882156, 0.00008223, 0.00926607, 0.00005722, -0.00011594],
                "eirplr": [0.08144133, 0.41927141, 0.49939604]},
            "Water-cooled — Screw": {
                "capft": [0.332669598, 0.00729116, -0.00049938, 0.01598983, -0.00028254, 0.00052346],
                "eirft": [0.66625406, 0.00068584, 0.00028496, -0.00341677, 0.00025484, -0.00048195],
                "eirplr": [0.33018833, 0.23554291, 0.46070828]},
            "Water-cooled — Centrifugal": {
                "capft": [-0.29861975, 0.02996076, -0.00080125, 0.01736268, -0.00032606, 0.00063139],
                "eirft": [0.51777196, -0.00400363, 0.00002026, 0.00698793, 0.0000829, -0.00015467],
                "eirplr": [0.17149273, 0.58820208, 0.23737257]},
        },
    },
    "boiler": {
        "name": "Boiler", "sec": "8.4.5.2",
        "types": {"Non-condensing": [0.082597, 0.996764, -0.079361],
                  "Condensing": [0.00533, 0.904, 0.09066]},
    },
    "furnace": {
        "name": "Furnace", "sec": "8.4.5.3",
        "types": {"Atmospheric": [0.0186100, 1.0942090, -0.1128190],
                  "Condensing": [0.00533, 0.904, 0.09066]},
    },
}

def biquad(c, x1, x2):
    return c[0] + c[1]*x1 + c[2]*x1**2 + c[3]*x2 + c[4]*x2**2 + c[5]*x1*x2
def cubic1(c, x):
    return c[0] + c[1]*x + c[2]*x**2 + c[3]*x**3
def polyplr(c, p):
    return sum(c[i]*p**i for i in range(len(c)))
def fit_biquad(x1, x2, y):
    X = np.column_stack([np.ones_like(x1), x1, x1**2, x2, x2**2, x1*x2])
    return np.linalg.lstsq(X, y, rcond=None)[0]
def fit_poly(x, y, order):
    return np.linalg.lstsq(np.vander(x, order+1, increasing=True), y, rcond=None)[0]

GREEN, CORAL, BLUE = "#1D9E75", "#D85A30", "#378ADD"
FAM = plt.cm.viridis(np.linspace(0.15, 0.85, 4))

def new_ax(title, xlab, ylab):
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.set_title(title, fontsize=11); ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    ax.grid(alpha=0.25); return fig, ax

st.markdown('<div class="hero"><h1>NECB Performance Curve Explorer</h1>'
            '<p>Part-load curves from NECB 2020, Subsection 8.4.5 · all values in °F / Imperial</p></div>',
            unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Mode")
    mode = st.radio("mode", ["NECB curves", "Create graphs from points"],
                    label_visibility="collapsed")

if mode == "NECB curves":
    family = st.selectbox("Equipment family", ["Heat pumps & cooling", "Boilers & furnaces"])
    if family == "Heat pumps & cooling":
        unit = st.selectbox("Equipment",
                            ["Air-source heat pump (8.4.5.7)", "Electric chiller (8.4.5.5)",
                             "Direct-expansion cooling (8.4.5.4)"])
        key = {"A": "ashp", "E": "chiller", "D": "dx"}[unit[0]]
    else:
        unit = st.selectbox("Equipment", ["Boiler (8.4.5.2)", "Furnace (8.4.5.3)"])
        key = "boiler" if unit.startswith("B") else "furnace"

    eq = NECB[key]
    st.markdown(f'<span class="badge">NECB {eq["sec"]}</span>', unsafe_allow_html=True)

    if key in ("boiler", "furnace"):
        typ = st.selectbox("Type", list(eq["types"].keys()))
        c = eq["types"][typ]
        st.markdown(f"#### {eq['name']} — fuel part-load curve")
        st.latex(r"FHeatPLC = a + b\,PLR + c\,PLR^2")
        st.markdown(
            f'<div class="card"><span class="coef">a = <b>{c[0]:.6g}</b></span>'
            f'<span class="coef">b = <b>{c[1]:.6g}</b></span>'
            f'<span class="coef">c = <b>{c[2]:.6g}</b></span></div>', unsafe_allow_html=True)
        p = np.linspace(0.1, 1.0, 100)
        fig, ax = new_ax(f"{eq['name']} ({typ})", "Part-load ratio", "FHeatPLC")
        ax.plot(p, polyplr(c, p), color=CORAL, lw=2)
        st.pyplot(fig, use_container_width=False)
        st.caption("Fuel_partload = Fuel_design × FHeatPLC. Boilers and furnaces have only this "
                   "single part-load curve in the code — no CAP-FT or EIR-FT.")
    else:
        if key == "chiller":
            typ = st.selectbox("Chiller type", list(eq["types"].keys()))
            cf = eq["types"][typ]["capft"]; ef = eq["types"][typ]["eirft"]; ep = eq["types"][typ]["eirplr"]
        else:
            cf = eq["capft"]["c"]; ef = eq["eirft"]["c"]; ep = eq["eirplr"]["c"]

        with st.expander("Plot ranges (adjust)"):
            r1, r2 = st.columns(2)
            xlo = r1.number_input("x-axis min (°F)", value=float(eq["xr"][0]))
            xhi = r2.number_input("x-axis max (°F)", value=float(eq["xr"][1]))
        xg = np.linspace(xlo, xhi, 100)
        cols = st.columns(3)

        with cols[0]:
            st.markdown("**CAP-FT** — capacity")
            fig, ax = new_ax("CAP-FT", eq["xlab"], "× rated")
            if key == "ashp":
                st.latex(r"a+b\,t+c\,t^2+d\,t^3")
                ax.plot(xg, cubic1(cf, xg), color=GREEN, lw=2)
            else:
                st.latex(r"a+b\,v_1+c\,v_1^2+d\,v_2+e\,v_2^2+f\,v_1v_2")
                for tl, col in zip(eq["lines"], FAM):
                    ax.plot(xg, biquad(cf, tl, xg), color=col, lw=1.7, label=f"{tl}°")
                ax.legend(title=eq["linelab"], fontsize=7, title_fontsize=7)
            st.pyplot(fig, use_container_width=True)

        with cols[1]:
            st.markdown("**EIR-FT** — energy input")
            fig, ax = new_ax("EIR-FT", eq["xlab"], "× rated")
            if key == "ashp":
                st.latex(r"a+b\,t+c\,t^2+d\,t^3")
                ax.plot(xg, cubic1(ef, xg), color=BLUE, lw=2)
            else:
                st.latex(r"a+b\,v_1+c\,v_1^2+d\,v_2+e\,v_2^2+f\,v_1v_2")
                for tl, col in zip(eq["lines"], FAM):
                    ax.plot(xg, biquad(ef, tl, xg), color=col, lw=1.7, label=f"{tl}°")
                ax.legend(title=eq["linelab"], fontsize=7, title_fontsize=7)
            st.pyplot(fig, use_container_width=True)

        with cols[2]:
            st.markdown("**EIR-fPLR** — part load")
            st.latex(r"a+b\,PLR+c\,PLR^2" + (r"+d\,PLR^3" if len(ep) == 4 else ""))
            fig, ax = new_ax("EIR-fPLR", "Part-load ratio", "× full-load")
            pg = np.linspace(0.1, 1.0, 100)
            ax.plot(pg, polyplr(ep, pg), color=CORAL, lw=2)
            st.pyplot(fig, use_container_width=True)

        def coefrow(name, c, labels):
            chips = "".join(f'<span class="coef">{l} = <b>{v:.6g}</b></span>' for l, v in zip(labels, c))
            return f'<div style="margin-bottom:.5rem;"><b>{name}</b><br>{chips}</div>'
        L6, Lp = ["a", "b", "c", "d", "e", "f"], ["a", "b", "c", "d"]
        st.markdown('<div class="card">' + coefrow("CAP-FT", cf, L6 if key != "ashp" else Lp)
                    + coefrow("EIR-FT", ef, L6 if key != "ashp" else Lp)
                    + coefrow("EIR-fPLR", ep, Lp[:len(ep)]) + '</div>', unsafe_allow_html=True)

else:
    st.markdown("#### Create graphs from your points")
    st.caption("Enter points for each curve; the app fits and plots them. CAP-FT and EIR-FT take "
               "two variables (v1, v2) and a value; EIR-fPLR takes PLR and a value. All °F / Imperial.")
    cap_def = pd.DataFrame({"v1": [67, 67, 67, 67], "v2": [75, 85, 95, 105], "value": [1.10, 1.05, 1.00, 0.95]})
    eir_def = pd.DataFrame({"v1": [67, 67, 67, 67], "v2": [75, 85, 95, 105], "value": [0.90, 0.95, 1.00, 1.08]})
    plr_def = pd.DataFrame({"PLR": [1.00, 0.75, 0.50, 0.25], "value": [1.00, 0.72, 0.47, 0.27]})

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**CAP-FT points**")
        capp = st.data_editor(cap_def, num_rows="dynamic", key="capp", use_container_width=True)
    with c2:
        st.markdown("**EIR-FT points**")
        eirp = st.data_editor(eir_def, num_rows="dynamic", key="eirp", use_container_width=True)
    with c3:
        st.markdown("**EIR-fPLR points**")
        plrp = st.data_editor(plr_def, num_rows="dynamic", key="plrp", use_container_width=True)

    g1, g2, g3 = st.columns(3)

    def plot_biquad(df, title, container):
        df = df.dropna()
        with container:
            if len(df) < 6 or df["v1"].nunique() < 3 or df["v2"].nunique() < 3:
                st.info("Need ≥6 points and ≥3 distinct values of each variable.")
                return
            x1 = df["v1"].to_numpy(float); x2 = df["v2"].to_numpy(float); y = df["value"].to_numpy(float)
            c = fit_biquad(x1, x2, y)
            xg = np.linspace(x2.min()-3, x2.max()+3, 80)
            fig, ax = new_ax(title, "v2", "value")
            for tl, col in zip(sorted(set(x1)), FAM):
                ax.plot(xg, biquad(c, tl, xg), color=col, lw=1.6, label=f"v1={tl:g}")
                sel = x1 == tl
                ax.scatter(x2[sel], y[sel], color=col, s=26, edgecolor="k", linewidth=0.4, zorder=5)
            ax.legend(fontsize=7)
            st.pyplot(fig, use_container_width=True)

    plot_biquad(capp, "CAP-FT (fitted)", g1)
    plot_biquad(eirp, "EIR-FT (fitted)", g2)
    with g3:
        pdf = plrp.dropna()
        if len(pdf) < 3:
            st.info("Need ≥3 part-load points.")
        else:
            p = pdf["PLR"].to_numpy(float); v = pdf["value"].to_numpy(float)
            order = 3 if len(pdf) >= 4 else 2
            c = fit_poly(p, v, order)
            pg = np.linspace(0.2, 1.0, 80)
            fig, ax = new_ax("EIR-fPLR (fitted)", "Part-load ratio", "value")
            ax.plot(pg, np.polynomial.polynomial.polyval(pg, c), color=CORAL, lw=1.8)
            ax.scatter(p, v, color=CORAL, s=28, edgecolor="k", linewidth=0.4, zorder=5)
            st.pyplot(fig, use_container_width=True)
