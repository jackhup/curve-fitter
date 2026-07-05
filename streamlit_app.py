"""
streamlit_app.py — NECB Performance Curve Explorer (v3)
Modes:
  1. NECB curves   — equipment curves from NECB 2020, Subsection 8.4.5, with
                     formulas, variable definitions, COP and capacity graphs
  2. Create graphs — fit your own points (cubic-in-T or biquadratic),
                     with named variables and COP outputs
All temperatures in °F, Imperial units.
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
.block-container{padding-top:2.2rem; max-width:1200px;}
h1,h2,h3{color:#12303a;}
.hero{background:linear-gradient(90deg,#0f6e56,#1D9E75); color:#fff;
      padding:1.1rem 1.4rem; border-radius:14px; margin-bottom:1.4rem;}
.hero h1{color:#fff; margin:0; font-size:1.7rem;}
.hero p{color:#dff3ec; margin:.25rem 0 0; font-size:.95rem;}
.card{background:#f6f9fb; border:1px solid #e3e8ee; border-radius:14px;
      padding:1rem 1.3rem; margin-bottom:1.1rem;}
.badge{display:inline-block; background:#e1f5ee; color:#0f6e56;
       padding:3px 12px; border-radius:20px; font-size:.78rem; font-weight:600; margin-bottom:.4rem;}
.coef{display:inline-block; background:#fff; border:1px solid #e3e8ee; border-radius:8px;
      padding:5px 11px; margin:3px 5px 3px 0; font-size:.9rem;
      font-family:ui-monospace,Menlo,monospace;}
.coef b{color:#0f6e56;}
.vars{background:#fffdf4; border:1px solid #efe7c7; border-radius:10px;
      padding:.6rem .9rem; font-size:.86rem; margin:.3rem 0 .8rem;}
section[data-testid="stSidebar"]{background:#0f2a24;}
section[data-testid="stSidebar"] *{color:#e8f3ee;}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ NECB data
NECB = {
    "ashp": {
        "name": "Air-source heat pump", "sec": "8.4.5.7", "mode": "heating",
        "capft": [0.2536714, 0.0104351, 0.0001861, -0.0000015],
        "eirft": [2.4600298, -0.0622539, 0.0008800, -0.0000046],
        "eirplr": [0.0856522, 0.9388137, -0.1834361, 0.1589702],
        "xr": (-10.0, 60.0), "xlab": "Outdoor air dry-bulb  t_odb  (°F)",
        "vars": "t = **t_odb** — outdoor air dry-bulb temperature (°F). "
                "Heating capacity and power both depend only on outdoor temperature.",
        "cop_default": 3.3,
    },
    "dx": {
        "name": "Direct-expansion cooling", "sec": "8.4.5.4", "mode": "cooling",
        "capft": [0.8740302, -0.0011416, 0.0001711, -0.0029570, 0.0000102, -0.0000592],
        "eirft": [-1.0639310, 0.0306584, -0.0001269, 0.0154213, 0.0000497, -0.0002096],
        "eirplr": [0.2012301, -0.0312175, 1.9504979, -1.1205105],
        "lines": [60, 63, 67, 72], "xr": (75.0, 115.0),
        "xlab": "Outdoor air dry-bulb  t_odb  (°F)", "linelab": "t_wb (°F)",
        "v1n": "t_wb", "v2n": "t_odb",
        "vars": "**t_wb** — entering coil wet-bulb temperature (°F), the air condition at the "
                "evaporator coil. **t_odb** — outdoor air dry-bulb temperature (°F) at the condenser.",
        "cop_default": 3.5,
    },
    "chiller": {
        "name": "Electric chiller", "sec": "8.4.5.5", "mode": "cooling",
        "lines": [40, 44, 48], "xr": (65.0, 95.0),
        "xlab": "Condenser water supply  t_cws  (°F)", "linelab": "t_chws (°F)",
        "v1n": "t_chws", "v2n": "t_cws",
        "vars": "**t_chws** — chilled water supply temperature (°F), the water leaving the "
                "evaporator. **t_cws** — condenser water supply temperature (°F), the water "
                "entering the condenser (for air-cooled machines this represents the outdoor "
                "air condition at the condenser).",
        "cop_default": 5.0,
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
    "boiler": {"name": "Boiler", "sec": "8.4.5.2",
               "types": {"Non-condensing": [0.082597, 0.996764, -0.079361],
                         "Condensing": [0.00533, 0.904, 0.09066]}},
    "furnace": {"name": "Furnace", "sec": "8.4.5.3",
                "types": {"Atmospheric": [0.0186100, 1.0942090, -0.1128190],
                          "Condensing": [0.00533, 0.904, 0.09066]}},
}

# ------------------------------------------------------------------ math
def biquad(c, x1, x2):
    return c[0] + c[1]*x1 + c[2]*x1**2 + c[3]*x2 + c[4]*x2**2 + c[5]*x1*x2
def cubic1(c, x):
    return c[0] + c[1]*x + c[2]*x**2 + c[3]*x**3
def polyplr(c, p):
    return sum(c[i]*np.asarray(p)**i for i in range(len(c)))
def fit_biquad(x1, x2, y):
    X = np.column_stack([np.ones_like(x1), x1, x1**2, x2, x2**2, x1*x2])
    return np.linalg.lstsq(X, y, rcond=None)[0]
def fit_cubic1(x, y):
    return np.linalg.lstsq(np.vander(x, 4, increasing=True), y, rcond=None)[0]
def fit_poly(x, y, order):
    return np.linalg.lstsq(np.vander(x, order+1, increasing=True), y, rcond=None)[0]

GREEN, CORAL, BLUE, PURPLE = "#1D9E75", "#D85A30", "#378ADD", "#7F77DD"
FAM = plt.cm.viridis(np.linspace(0.15, 0.85, 4))

def new_ax(title, xlab, ylab):
    fig, ax = plt.subplots(figsize=(5.2, 3.7))
    ax.set_title(title, fontsize=10.5)
    ax.set_xlabel(xlab, fontsize=9); ax.set_ylabel(ylab, fontsize=9)
    ax.tick_params(labelsize=8); ax.grid(alpha=0.25)
    return fig, ax

COP_ID = (r"COP_{op}=\dfrac{Q_{op}}{P_{op}}"
          r"=COP_{rated}\cdot\dfrac{PLR}{EIR\_FT \cdot EIR\_FPLR}")

# ------------------------------------------------------------------ layout
st.markdown('<div class="hero"><h1>NECB Performance Curve Explorer</h1>'
            '<p>Part-load curves from NECB 2020, Subsection 8.4.5 · all values in °F / Imperial</p></div>',
            unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Mode")
    mode = st.radio("mode", ["NECB curves", "Create graphs from points"],
                    label_visibility="collapsed")

# =============================================================== MODE 1: NECB
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

    # ---------------- boiler / furnace ----------------
    if key in ("boiler", "furnace"):
        typ = st.selectbox("Type", list(eq["types"].keys()))
        c = eq["types"][typ]
        st.markdown(f"#### {eq['name']} — fuel part-load curve")
        st.latex(r"FHeatPLC = a + b\,PLR + c\,PLR^2")
        st.markdown('<div class="vars"><b>PLR</b> — part-load ratio, Q_partload / Q_design '
                    '(fraction of design heating output). <b>FHeatPLC</b> — fuel heating '
                    'part-load curve: the fraction of design fuel input consumed at that PLR. '
                    'Fuel_partload = Fuel_design × FHeatPLC.</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="card"><span class="coef">a = <b>{c[0]:.6g}</b></span>'
                    f'<span class="coef">b = <b>{c[1]:.6g}</b></span>'
                    f'<span class="coef">c = <b>{c[2]:.6g}</b></span></div>', unsafe_allow_html=True)
        p = np.linspace(0.1, 1.0, 100)
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = new_ax(f"{eq['name']} ({typ})", "Part-load ratio  PLR  (–)",
                             "FHeatPLC — fraction of design fuel input (–)")
            ax.plot(p, polyplr(c, p), color=CORAL, lw=2)
            st.pyplot(fig, use_container_width=True)
        with col2:
            fig, ax = new_ax("Part-load efficiency", "Part-load ratio  PLR  (–)",
                             "Efficiency ÷ rated efficiency (–)")
            ax.plot(p, p / polyplr(c, p), color=GREEN, lw=2)
            ax.axhline(1, color="0.6", ls=":", lw=1)
            st.pyplot(fig, use_container_width=True)
        st.caption("Right graph: PLR ÷ FHeatPLC = efficiency relative to rated — shows the "
                   "part-load efficiency penalty (or gain, for condensing at low load).")

    # ---------------- ASHP / DX / chiller ----------------
    else:
        if key == "chiller":
            typ = st.selectbox("Chiller type", list(eq["types"].keys()))
            cf, ef, ep = (eq["types"][typ][k] for k in ("capft", "eirft", "eirplr"))
        else:
            cf, ef, ep = eq["capft"], eq["eirft"], eq["eirplr"]

        cA, cB, cC = st.columns(3)
        xlo = cA.number_input("x-axis min (°F)", value=float(eq["xr"][0]))
        xhi = cB.number_input("x-axis max (°F)", value=float(eq["xr"][1]))
        cop_r = cC.number_input("Rated COP (for COP graphs)", value=float(eq["cop_default"]),
                                min_value=0.5, step=0.1)
        xg = np.linspace(xlo, xhi, 120)
        pg = np.linspace(0.15, 1.0, 120)

        # formulas + variable definitions
        if key == "ashp":
            st.latex(r"CAP\_FT = a + b\,t + c\,t^2 + d\,t^3 \qquad EIR\_FT = a + b\,t + c\,t^2 + d\,t^3")
        else:
            v1, v2 = eq["v1n"], eq["v2n"]
            st.latex(rf"CAP\_FT,\;EIR\_FT = a + b\,{v1} + c\,{v1}^2 + d\,{v2} + e\,{v2}^2 + f\,{v1}\,{v2}")
        st.latex(r"EIR\_FPLR = a + b\,PLR + c\,PLR^2" + (r" + d\,PLR^3" if len(ep) == 4 else ""))
        st.latex(COP_ID)
        st.markdown(f'<div class="vars">{eq["vars"]} '
                    '<b>PLR</b> — part-load ratio = Q_operating / Q_available (load as a fraction '
                    'of the capacity available at the current temperatures, not rated capacity). '
                    '<b>CAP_FT</b> — available capacity ÷ rated capacity. '
                    '<b>EIR_FT, EIR_FPLR</b> — power-input multipliers on rated efficiency. '
                    '<b>Q_available</b> = Q_rated × CAP_FT; <b>P_operating</b> = P_rated × CAP_FT × '
                    'EIR_FT × EIR_FPLR.</div>', unsafe_allow_html=True)

        def temp_curve(coefs, tl=None):
            if key == "ashp":
                return cubic1(coefs, xg)
            return biquad(coefs, tl, xg)

        # ---- row 1: the three code curves ----
        r1 = st.columns(3)
        with r1[0]:
            fig, ax = new_ax("CAP_FT — capacity vs temperature", eq["xlab"],
                             "CAP_FT = Q_available / Q_rated (–)")
            if key == "ashp":
                ax.plot(xg, temp_curve(cf), color=GREEN, lw=2)
            else:
                for tl, col in zip(eq["lines"], FAM):
                    ax.plot(xg, temp_curve(cf, tl), color=col, lw=1.7, label=f"{tl}")
                ax.legend(title=eq["linelab"], fontsize=7, title_fontsize=7)
            ax.axhline(1, color="0.6", ls=":", lw=1)
            st.pyplot(fig, use_container_width=True)
        with r1[1]:
            fig, ax = new_ax("EIR_FT — energy input vs temperature", eq["xlab"],
                             "EIR_FT = power multiplier (–)")
            if key == "ashp":
                ax.plot(xg, temp_curve(ef), color=BLUE, lw=2)
            else:
                for tl, col in zip(eq["lines"], FAM):
                    ax.plot(xg, temp_curve(ef, tl), color=col, lw=1.7, label=f"{tl}")
                ax.legend(title=eq["linelab"], fontsize=7, title_fontsize=7)
            ax.axhline(1, color="0.6", ls=":", lw=1)
            st.pyplot(fig, use_container_width=True)
        with r1[2]:
            fig, ax = new_ax("EIR_FPLR — energy input vs part load",
                             "Part-load ratio  PLR  (–)", "EIR_FPLR = power multiplier (–)")
            ax.plot(pg, polyplr(ep, pg), color=CORAL, lw=2)
            ax.axhline(1, color="0.6", ls=":", lw=1)
            st.pyplot(fig, use_container_width=True)

        # ---- row 2: derived COP and capacity graphs ----
        st.markdown("#### Derived performance — what this means for COP and available capacity")
        mid = eq["lines"][1] if key != "ashp" else None
        r2 = st.columns(3)
        with r2[0]:
            fig, ax = new_ax("COP vs temperature (full load, PLR = 1)", eq["xlab"],
                             "COP_operating (W/W)")
            eirfplr1 = float(polyplr(ep, 1.0))
            if key == "ashp":
                ax.plot(xg, cop_r / (temp_curve(ef) * eirfplr1), color=PURPLE, lw=2)
            else:
                for tl, col in zip(eq["lines"], FAM):
                    ax.plot(xg, cop_r / (temp_curve(ef, tl) * eirfplr1),
                            color=col, lw=1.7, label=f"{tl}")
                ax.legend(title=eq["linelab"], fontsize=7, title_fontsize=7)
            ax.axhline(cop_r, color="0.6", ls=":", lw=1)
            st.pyplot(fig, use_container_width=True)
            st.caption("COP = COP_rated / (EIR_FT × EIR_FPLR(1)). Dotted line = rated COP.")
        with r2[1]:
            fig, ax = new_ax("COP vs part load (at rated temperatures)",
                             "Part-load ratio  PLR  (–)", "COP_operating (W/W)")
            ax.plot(pg, cop_r * pg / polyplr(ep, pg), color=PURPLE, lw=2)
            ax.axhline(cop_r, color="0.6", ls=":", lw=1)
            st.pyplot(fig, use_container_width=True)
            st.caption("COP = COP_rated × PLR / EIR_FPLR, with EIR_FT = 1 at rated temperatures.")
        with r2[2]:
            fig, ax = new_ax("Q_operating vs part load", "Part-load ratio  PLR  (–)",
                             "Q_operating / Q_rated (–)")
            if key == "ashp":
                for t_pick, col in zip(np.linspace(xlo, xhi, 4), FAM):
                    capft = float(cubic1(cf, t_pick))
                    ax.plot(pg, capft * pg, color=col, lw=1.7, label=f"{t_pick:.0f}°F")
                ax.legend(title="t_odb", fontsize=7, title_fontsize=7)
            else:
                for tl, col in zip(eq["lines"], FAM):
                    capft = float(biquad(cf, tl, (xlo + xhi) / 2))
                    ax.plot(pg, capft * pg, color=col, lw=1.7, label=f"{tl}")
                ax.legend(title=eq["linelab"], fontsize=7, title_fontsize=7)
            st.pyplot(fig, use_container_width=True)
            st.caption("Q_operating = Q_rated × CAP_FT × PLR. The line's endpoint at PLR = 1 is "
                       "Q_available at that temperature — capacity shrinks or grows with conditions.")

        def coefrow(name, c, labels):
            chips = "".join(f'<span class="coef">{l} = <b>{v:.6g}</b></span>' for l, v in zip(labels, c))
            return f'<div style="margin-bottom:.5rem;"><b>{name}</b><br>{chips}</div>'
        L6, Lp = ["a", "b", "c", "d", "e", "f"], ["a", "b", "c", "d"]
        st.markdown('<div class="card">' + coefrow("CAP_FT", cf, L6 if key != "ashp" else Lp)
                    + coefrow("EIR_FT", ef, L6 if key != "ashp" else Lp)
                    + coefrow("EIR_FPLR", ep, Lp[:len(ep)]) + '</div>', unsafe_allow_html=True)

# ================================================== MODE 2: fit from points
else:
    st.markdown("#### Create graphs from your points")
    ctype = st.radio("Curve type",
                     ["Function of one temperature — cubic (e.g. air-source heat pump)",
                      "Function of two temperatures — biquadratic (e.g. chiller, water-source HP)"],
                     horizontal=False)
    onevar = ctype.startswith("Function of one")

    VARS = {
        "t_odb — outdoor air dry-bulb (°F)": "t_odb",
        "t_wb — entering coil wet-bulb (°F)": "t_wb",
        "t_chws — chilled water supply (°F)": "t_chws",
        "t_hws — hot water supply (°F)": "t_hws",
        "t_cws — condenser water supply (°F)": "t_cws",
        "t_src — source water temperature (°F)": "t_src",
    }
    cop_r = st.number_input("Rated COP (for the COP graphs)", value=4.0, min_value=0.5, step=0.1)

    if onevar:
        vsel = st.selectbox("Temperature variable", list(VARS.keys()), index=0)
        vn = VARS[vsel]
        st.markdown(f'<div class="vars">Enter capacity and power multipliers versus <b>{vsel}</b>. '
                    'Values are ratios to the rated point (1.0 at rated conditions). '
                    'Need ≥4 points for the cubic fit.</div>', unsafe_allow_html=True)
        cap_def = pd.DataFrame({vn: [17.0, 32.0, 47.0, 60.0], "CAP_FT": [0.55, 0.76, 1.00, 1.22]})
        eir_def = pd.DataFrame({vn: [17.0, 32.0, 47.0, 60.0], "EIR_FT": [1.55, 1.20, 1.00, 0.90]})
    else:
        c1v, c2v = st.columns(2)
        v1sel = c1v.selectbox("First temperature (curve family lines)",
                              list(VARS.keys()), index=2)
        v2sel = c2v.selectbox("Second temperature (x-axis)", list(VARS.keys()), index=4)
        v1n, v2n = VARS[v1sel], VARS[v2sel]
        st.markdown(f'<div class="vars">Enter multipliers at grid points of <b>{v1sel}</b> and '
                    f'<b>{v2sel}</b>, as ratios to the rated point. Need ≥6 points and ≥3 distinct '
                    'values of each temperature for the biquadratic fit.</div>', unsafe_allow_html=True)
        cap_def = pd.DataFrame({v1n: [44, 44, 44, 40, 40, 40, 48, 48, 48],
                                v2n: [75, 85, 95, 75, 85, 95, 75, 85, 95],
                                "CAP_FT": [1.09, 1.05, 1.00, 1.05, 1.01, 0.96, 1.13, 1.09, 1.04]})
        eir_def = pd.DataFrame({v1n: [44, 44, 44, 40, 40, 40, 48, 48, 48],
                                v2n: [75, 85, 95, 75, 85, 95, 75, 85, 95],
                                "EIR_FT": [0.88, 0.94, 1.00, 0.91, 0.97, 1.04, 0.85, 0.91, 0.97]})
    plr_def = pd.DataFrame({"PLR": [1.00, 0.75, 0.50, 0.25], "EIR_FPLR": [1.00, 0.72, 0.47, 0.27]})

    e1, e2, e3 = st.columns(3)
    with e1:
        st.markdown("**CAP_FT points** — capacity ÷ rated")
        capp = st.data_editor(cap_def, num_rows="dynamic", key="capp", use_container_width=True)
    with e2:
        st.markdown("**EIR_FT points** — power multiplier")
        eirp = st.data_editor(eir_def, num_rows="dynamic", key="eirp", use_container_width=True)
    with e3:
        st.markdown("**EIR_FPLR points** — power vs part load")
        plrp = st.data_editor(plr_def, num_rows="dynamic", key="plrp", use_container_width=True)

    st.latex(COP_ID)

    capp, eirp, plrp = capp.dropna(), eirp.dropna(), plrp.dropna()
    ok, fits = True, {}

    if onevar:
        for nm, df, col in (("cap", capp, "CAP_FT"), ("eir", eirp, "EIR_FT")):
            if len(df) < 4:
                st.info(f"{col}: need ≥4 points for a cubic fit."); ok = False
            else:
                fits[nm] = fit_cubic1(df[vn].to_numpy(float), df[col].to_numpy(float))
    else:
        for nm, df, col in (("cap", capp, "CAP_FT"), ("eir", eirp, "EIR_FT")):
            if len(df) < 6 or df[v1n].nunique() < 3 or df[v2n].nunique() < 3:
                st.info(f"{col}: need ≥6 points and ≥3 distinct values of each temperature."); ok = False
            else:
                fits[nm] = fit_biquad(df[v1n].to_numpy(float), df[v2n].to_numpy(float),
                                      df[col].to_numpy(float))
    if len(plrp) < 3:
        st.info("EIR_FPLR: need ≥3 part-load points."); ok = False
    else:
        fits["plr"] = fit_poly(plrp["PLR"].to_numpy(float), plrp["EIR_FPLR"].to_numpy(float),
                               3 if len(plrp) >= 4 else 2)

    if ok:
        pg = np.linspace(0.15, 1.0, 100)
        if onevar:
            x = capp[vn].to_numpy(float)
            xg = np.linspace(x.min()-3, x.max()+3, 100)
            xlab = vsel
            def curve(c, tl=None): return cubic1(c, xg)
            lines = [None]
        else:
            x2 = capp[v2n].to_numpy(float)
            xg = np.linspace(x2.min()-3, x2.max()+3, 100)
            xlab = v2sel
            lines = sorted(capp[v1n].unique())[:4]
            def curve(c, tl): return biquad(c, tl, xg)

        r1 = st.columns(3)
        specs = [("cap", "CAP_FT (fitted)", "CAP_FT = Q_available / Q_rated (–)", capp, "CAP_FT", GREEN),
                 ("eir", "EIR_FT (fitted)", "EIR_FT = power multiplier (–)", eirp, "EIR_FT", BLUE)]
        for (nm, title, ylab, df, col, colr), cont in zip(specs, r1[:2]):
            with cont:
                fig, ax = new_ax(title, xlab, ylab)
                if onevar:
                    ax.plot(xg, curve(fits[nm]), color=colr, lw=2)
                    ax.scatter(df[vn], df[col], color=colr, s=26, edgecolor="k",
                               linewidth=0.4, zorder=5)
                else:
                    for tl, cc in zip(lines, FAM):
                        ax.plot(xg, curve(fits[nm], tl), color=cc, lw=1.6, label=f"{v1n}={tl:g}")
                        sel = df[v1n] == tl
                        ax.scatter(df.loc[sel, v2n], df.loc[sel, col], color=cc, s=26,
                                   edgecolor="k", linewidth=0.4, zorder=5)
                    ax.legend(fontsize=7)
                st.pyplot(fig, use_container_width=True)
        with r1[2]:
            fig, ax = new_ax("EIR_FPLR (fitted)", "Part-load ratio  PLR  (–)",
                             "EIR_FPLR = power multiplier (–)")
            ax.plot(pg, np.polynomial.polynomial.polyval(pg, fits["plr"]), color=CORAL, lw=1.8)
            ax.scatter(plrp["PLR"], plrp["EIR_FPLR"], color=CORAL, s=28,
                       edgecolor="k", linewidth=0.4, zorder=5)
            st.pyplot(fig, use_container_width=True)

        st.markdown("#### Derived COP and capacity")
        eirfplr1 = float(np.polynomial.polynomial.polyval(1.0, fits["plr"]))
        r2 = st.columns(3)
        with r2[0]:
            fig, ax = new_ax("COP vs temperature (PLR = 1)", xlab, "COP_operating (W/W)")
            if onevar:
                ax.plot(xg, cop_r / (curve(fits["eir"]) * eirfplr1), color=PURPLE, lw=2)
            else:
                for tl, cc in zip(lines, FAM):
                    ax.plot(xg, cop_r / (curve(fits["eir"], tl) * eirfplr1),
                            color=cc, lw=1.6, label=f"{v1n}={tl:g}")
                ax.legend(fontsize=7)
            ax.axhline(cop_r, color="0.6", ls=":", lw=1)
            st.pyplot(fig, use_container_width=True)
        with r2[1]:
            fig, ax = new_ax("COP vs part load (rated temps)", "Part-load ratio  PLR  (–)",
                             "COP_operating (W/W)")
            ax.plot(pg, cop_r * pg / np.polynomial.polynomial.polyval(pg, fits["plr"]),
                    color=PURPLE, lw=2)
            ax.axhline(cop_r, color="0.6", ls=":", lw=1)
            st.pyplot(fig, use_container_width=True)
        with r2[2]:
            fig, ax = new_ax("Q_operating vs part load", "Part-load ratio  PLR  (–)",
                             "Q_operating / Q_rated (–)")
            if onevar:
                for t_pick, cc in zip(np.linspace(xg.min(), xg.max(), 4), FAM):
                    ax.plot(pg, float(cubic1(fits["cap"], t_pick)) * pg, color=cc, lw=1.6,
                            label=f"{t_pick:.0f}°F")
            else:
                xmid = float(np.median(xg))
                for tl, cc in zip(lines, FAM):
                    ax.plot(pg, float(biquad(fits["cap"], tl, xmid)) * pg, color=cc, lw=1.6,
                            label=f"{v1n}={tl:g}")
            ax.legend(fontsize=7)
            st.pyplot(fig, use_container_width=True)
        st.caption("Endpoint of each Q line at PLR = 1 is Q_available at that temperature.")
