# app_v2.py
# ============================================================
# Plataforma IA 2.0 — Producto + Empaque + Claims + Producto Nuevo (Cold Start)
# Incluye:
# ✅ Modelos: éxito (clasificación) + ventas (regresión) con RandomForest
# ✅ Claims Lab (recomendaciones + score)
# ✅ Pack Vision+ (imagen -> métricas -> quick wins -> 3-second choice)
# ✅ Insights (rankings + distribuciones)
# ✅ Producto Nuevo (cold start + comparables p25/p50/p75 + launch score)
# ✅ Recomendaciones What-If para Producto Nuevo (coldstart_recommendations)
# ✅ Reporte Ejecutivo descargable (TXT + CSV inputs)
# ✅ Market Intelligence
# ============================================================

# ============================================================
# 🧠 PRODUCT INTELLIGENCE PLATFORM — CORE ENGINE (FIXED)
# Base + Modelos + Pack Vision + Shelf & Emotion Engine
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, mean_absolute_error

# ----------------------------
# Config
# ----------------------------
st.set_page_config(
    page_title="AI Product Intelligence Platform",
    layout="wide"
)

DATA_PATH_DEFAULT = "mercado_cereales_5000_con_ventas.csv"

# ----------------------------
# Helpers
# ----------------------------
def clip(v,a,b):
    return float(np.clip(v,a,b))

def safe_percent(x):
    return f"{x*100:.2f}%"

def df_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")

# ============================================================
# 📦 PACK VISION ENGINE
# ============================================================

def image_metrics(img):
    im = img.convert("RGB")
    arr = np.asarray(im).astype(np.float32)

    gray = (0.299*arr[...,0] + 0.587*arr[...,1] + 0.114*arr[...,2])

    brightness = float(np.mean(gray)/255)
    contrast = float(np.std(gray)/255)

    rg = arr[...,0] - arr[...,1]
    yb = 0.5*(arr[...,0]+arr[...,1]) - arr[...,2]
    colorfulness = float((np.std(rg)+0.3*np.std(yb))/255)

    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:,1:-1] = gray[:,2:] - gray[:,:-2]
    gy[1:-1,:] = gray[2:,:] - gray[:-2,:]
    mag = np.sqrt(gx**2 + gy**2)

    edge_density = float(np.mean(mag > np.percentile(mag,85)))

    return {
        "brightness": brightness,
        "contrast": contrast,
        "colorfulness": colorfulness,
        "edge_density": edge_density
    }


def pack_scores_from_metrics(m):

    legibility = clip(70*m["contrast"] + 30*(1-abs(m["edge_density"]-0.18)/0.18),0,1)*100
    pop = clip(0.55*m["contrast"] + 0.45*m["colorfulness"],0,1)*100
    clarity = clip(0.6*m["contrast"] + 0.4*(1-m["edge_density"]),0,1)*100

    return {
        "legibility": round(legibility,1),
        "pop": round(pop,1),
        "clarity": round(clarity,1)
    }

# ============================================================
# 🧲 SHELF & EMOTION ENGINE
# ============================================================

def shelf_scores(pack_scores, metrics):

    attention = clip(0.6*pack_scores["pop"] + 0.4*(metrics["contrast"]*100),0,100)
    recall = clip(0.5*attention + 0.5*pack_scores["clarity"],0,100)
    choice = clip(0.45*attention + 0.35*pack_scores["clarity"] + 0.2*pack_scores["legibility"],0,100)

    emotion_energy = clip(pack_scores["pop"]/100,0,1)
    emotion_trust = clip(pack_scores["legibility"]/100,0,1)

    emotion = emotion_energy*0.6 + emotion_trust*0.4

    return {
        "attention": attention,
        "recall": recall,
        "choice": choice,
        "emotion": emotion*100
    }

# ============================================================
# 🔥 HEATMAP PROXY
# ============================================================

def simple_heatmap(img):
    im = img.convert("L")
    arr = np.asarray(im).astype(float)

    gx = np.zeros_like(arr)
    gy = np.zeros_like(arr)

    gx[:,1:-1] = arr[:,2:] - arr[:,:-2]
    gy[1:-1,:] = arr[2:,:] - arr[:-2,:]

    mag = np.sqrt(gx**2 + gy**2)
    mag = (mag - mag.min())/(mag.max()+1e-6)

    heat = np.stack([mag, mag*0.5, mag*0], axis=-1)
    heat = (heat*255).astype(np.uint8)

    return Image.fromarray(heat)

# ============================================================
# 🧮 MNL CHOICE SIMULATION
# ============================================================

def mnl_choice(df, beta):

    U = (
        beta["att"]*df["attention"] +
        beta["rec"]*df["recall"] +
        beta["emo"]*df["emotion"] +
        beta["price"]*df["price"]
    )

    U = U - U.max()
    expU = np.exp(U)
    probs = expU/expU.sum()

    df["choice_prob"] = probs
    return df.sort_values("choice_prob", ascending=False)

# ============================================================
# 📊 DATA LOADER
# ============================================================

@st.cache_data
def load_data(path):

    df = pd.read_csv(path)

    for c in ["marca","categoria","canal"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.lower().str.strip()

    # validación mínima
    needed = ["precio","competencia","demanda","tendencia","margen_pct",
              "conexion_score","rating_conexion","sentiment_score",
              "exito","ventas_unidades"]

    missing = [c for c in needed if c not in df.columns]
    if missing:
        st.error(f"Faltan columnas en dataset: {missing}")
        st.stop()

    return df

# ============================================================
# 🤖 MODELS — FIXED
# ============================================================

@st.cache_resource
def train_models(df):

    features = [
        "precio","competencia","demanda","tendencia",
        "margen_pct","conexion_score",
        "rating_conexion","sentiment_score",
        "marca","canal"
    ]

    X = df[features].copy()
    y_cls = df["exito"].astype(int).copy()
    y_reg = df["ventas_unidades"].astype(float).copy()

    num_cols = features[:-2]
    cat_cols = ["marca","canal"]

    pre = ColumnTransformer([
        ("num","passthrough",num_cols),
        ("cat",OneHotEncoder(handle_unknown="ignore"),cat_cols)
    ])

    # ---------- Clasificación ----------
    clf = Pipeline([
        ("pre", pre),
        ("model", RandomForestClassifier(n_estimators=300, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cls, test_size=0.2, random_state=42, stratify=y_cls
    )

    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    proba = clf.predict_proba(X_test)[:,1]

    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, proba)
    cm = confusion_matrix(y_test, pred)

    # ---------- Regresión ----------
    reg = Pipeline([
        ("pre", pre),
        ("model", RandomForestRegressor(n_estimators=300, random_state=42))
    ])

    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )

    reg.fit(X_train_s, y_train_s)

    pred_sales = reg.predict(X_test_s)
    mae = mean_absolute_error(y_test_s, pred_sales)

    return clf, reg, acc, auc, cm, mae

# ============================================================
# 🚀 BOOT
# ============================================================

if Path(DATA_PATH_DEFAULT).exists():
    df = load_data(DATA_PATH_DEFAULT)
else:
    st.error("No encuentro el dataset base")
    st.stop()

success_model, sales_model, ACC, AUC, CM, MAE = train_models(df)

# ============================================================
# 📡 MARKET INTELLIGENCE LAYER
# ============================================================

@st.cache_data
def load_mi_tables():
    tables = {}
    names = [
        "market_trends.csv",
        "market_claims.csv",
        "market_reviews.csv",
        "market_prices.csv"
    ]

    for n in names:
        if Path(n).exists():
            tables[n] = pd.read_csv(n)

    return tables


def mi_category_score(mi_tables, categoria):

    score = 0

    if "market_trends.csv" in mi_tables:
        t = mi_tables["market_trends.csv"]
        s = t[t["categoria"] == categoria]["trend_index"].mean()
        if pd.notna(s):
            score += 0.4 * s

    if "market_reviews.csv" in mi_tables:
        r = mi_tables["market_reviews.csv"]
        s = r[r["categoria"] == categoria]["sentiment"].mean()
        if pd.notna(s):
            score += 30 * s

    if "market_claims.csv" in mi_tables:
        c = mi_tables["market_claims.csv"]
        g = c[c["categoria"] == categoria]["growth_pct"].mean()
        if pd.notna(g):
            score += g * 0.3

    return clip(score, 0, 100)


def apply_mi_adjustment(row, mi_score):

    row = row.copy()

    row["demanda"] = clip(row["demanda"] * (1 + mi_score/200), 0, 100)
    row["tendencia"] = clip(row["tendencia"] * (1 + mi_score/250), 0, 100)

    return row


# ============================================================
# 🏷️ CLAIMS INTELLIGENCE ENGINE
# ============================================================

BASE_CLAIMS = {
    "fit": ["alto en proteína","sin azúcar añadida","alto en fibra","integral"],
    "kids": ["con vitaminas","sabor chocolate","energía diaria"],
    "premium": ["ingredientes seleccionados","calidad premium"],
    "value": ["rinde más","mejor precio"]
}


def get_claims_for_segment(segmento, mi_tables=None):

    base = BASE_CLAIMS.get(segmento, [])

    if mi_tables and "market_claims.csv" in mi_tables:
        mc = mi_tables["market_claims.csv"]
        extra = mc.sort_values("growth_pct", ascending=False)["claim"].head(5).tolist()
        base = list(dict.fromkeys(base + extra))

    return base


def claims_score(claims):

    if not claims:
        return 0

    return clip(60 + 8*len(claims), 0, 100)


# ============================================================
# 🧊 COLD START ENGINE (PRODUCTO NUEVO)
# ============================================================

def cold_start_predict(input_row, success_model, sales_model):

    df_row = pd.DataFrame([input_row])

    prob = float(success_model.predict_proba(df_row)[0][1])
    sales = float(sales_model.predict(df_row)[0])

    return prob, sales


# ============================================================
# 🚀 WHAT-IF RECOMMENDATION ENGINE
# ============================================================

def whatif_recommendations(base_row, success_model, sales_model):

    scenarios = []

    for dp in [-0.15,-0.1,0,0.1,0.15]:
        for dm in [-10,0,10]:
            r = base_row.copy()
            r["precio"] *= (1+dp)
            r["margen_pct"] = clip(r["margen_pct"]+dm,0,90)

            p,s = cold_start_predict(r, success_model, sales_model)

            scenarios.append({
                **r,
                "prob": p,
                "sales": s
            })

    out = pd.DataFrame(scenarios)
    out["score"] = out["prob"]*0.65 + (out["sales"]/out["sales"].max())*0.35

    return out.sort_values("score", ascending=False).head(10)


# ============================================================
# 💼 INVESTOR ENGINE
# ============================================================

def investor_metrics(price, cost, volume, cpa, retention):

    margin = price - cost
    ltv = margin * retention
    net = ltv - cpa

    return {
        "unit_margin": margin,
        "ltv": ltv,
        "ltv_net": net,
        "revenue": price * volume
    }


# ============================================================
# 📄 EXECUTIVE REPORT ENGINE
# ============================================================

def build_exec_report():

    lines = []
    lines.append("AI PRODUCT INTELLIGENCE REPORT")
    lines.append("="*40)

    if "last_sim" in st.session_state:
        lines.append("\nSIMULADOR:")
        for k,v in st.session_state.last_sim.items():
            lines.append(f"{k}: {v}")

    if "last_new" in st.session_state:
        lines.append("\nPRODUCTO NUEVO:")
        for k,v in st.session_state.last_new.items():
            lines.append(f"{k}: {v}")

    return "\n".join(lines).encode("utf-8")


# ============================================================
# 🔁 SHELF LEARNING LOG
# ============================================================

if "shelf_learning" not in st.session_state:
    st.session_state.shelf_learning = []


def log_shelf_learning(row):
    st.session_state.shelf_learning.append(row)
# ============================================================
# 🖥️ UI — BLOQUE 3 COMPLETO (DATASET UPLOADER + ROI + FIXES)
# ============================================================

# ----------------------------
# Sidebar: Dataset uploader
# ----------------------------
st.sidebar.title("📂 Datos")

uploaded_file = st.sidebar.file_uploader(
    "Sube tu dataset CSV (con ventas)",
    type=["csv"],
    key="dataset_uploader"
)

if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.sidebar.success("Dataset cargado desde upload ✅")
else:
    if Path(DATA_PATH_DEFAULT).exists():
        df = load_data(DATA_PATH_DEFAULT)
        st.sidebar.info(f"Usando dataset local: {DATA_PATH_DEFAULT}")
    else:
        st.error("No hay dataset disponible. Sube un CSV o agrega el archivo base al repo.")
        st.stop()

# Entrenar modelos con el df seleccionado
success_model, sales_model, ACC, AUC, CM, MAE = train_models(df)

# ----------------------------
# Header
# ----------------------------
st.title("🧠 AI Product Intelligence Platform")
st.caption("Predicción de éxito + ventas + pack/claims + Shelf & Emotion (3s) + Producto nuevo + Reporte")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Registros", f"{len(df):,}")
k2.metric("Precisión", f"{ACC*100:.2f}%")
k3.metric("AUC", f"{AUC:.3f}")
k4.metric("Éxito base", f"{df['exito'].mean()*100:.1f}%")
k5.metric("MAE ventas", f"{MAE:,.0f} u.")
st.divider()

# ----------------------------
# Claims library (simple + estable)
# ----------------------------
CLAIMS_LIBRARY = {
    "fit": ["Alto en proteína","Alto en fibra","Sin azúcar añadida","Integral","Sin colorantes artificiales","Bajo en calorías"],
    "kids": ["Con vitaminas y minerales","Sabor chocolate","Energía para su día","Sin conservadores","Hecho con granos"],
    "premium": ["Ingredientes seleccionados","Hecho con avena real","Calidad premium","Sabor intenso","Receta artesanal"],
    "value": ["Rinde más","Ideal para la familia","Gran sabor a mejor precio","Económico y práctico"]
}

def claims_score(claims_sel):
    if not claims_sel:
        return 0.0
    n = len(claims_sel)
    base = 85.0 if n <= 3 else max(60.0, 85.0 - (n-3)*8.0)
    return float(np.clip(base, 0, 100))

def df_to_csv_bytes(_df):
    return _df.to_csv(index=False).encode("utf-8")

# ----------------------------
# Tabs
# ----------------------------
tab_sim, tab_ins, tab_pack, tab_shelf, tab_new, tab_rep, tab_data, tab_diag = st.tabs([
    "🧪 Simulador",
    "📊 Insights",
    "📦 Pack Vision+",
    "🧲 Shelf & Emotion (3s)",
    "🧊 Producto Nuevo",
    "📄 Reporte",
    "📂 Datos",
    "🧠 Diagnóstico"
])

# ============================================================
# 🧪 SIMULADOR ✅ (con ROI financiero + ROI unidades)
# ============================================================
with tab_sim:
    st.subheader("🧪 Simulador (éxito + ventas + pack + claims + conexión)")

    marcas = sorted(df["marca"].unique().tolist())
    canales = sorted(df["canal"].unique().tolist())

    c1, c2, c3 = st.columns(3)
    marca = c1.selectbox("Marca", marcas, 0, key="sim_marca")
    canal = c2.selectbox("Canal", canales, 0, key="sim_canal")
    segmento = c3.selectbox("Segmento", ["fit","kids","premium","value"], 0, key="sim_seg")

    st.markdown("### Variables de negocio")
    b1, b2, b3, b4, b5 = st.columns(5)
    precio = b1.number_input("Precio", 1.0, 9999.0, float(df["precio"].median()), step=1.0, key="sim_precio")
    margen = b2.slider("Margen %", 0, 90, int(np.clip(df["margen_pct"].median(),0,90)), key="sim_margen")
    competencia = b3.slider("Competencia (1-10)", 1, 10, int(df["competencia"].median()), key="sim_comp")
    demanda = b4.slider("Demanda (0-100)", 0, 100, int(df["demanda"].median()), key="sim_dem")
    tendencia = b5.slider("Tendencia (0-100)", 0, 100, int(df["tendencia"].median()), key="sim_tend")

    st.markdown("### Pack + Claims")
    img = st.file_uploader("Sube empaque (opcional)", type=["png","jpg","jpeg"], key="sim_pack")

    if img:
        im = Image.open(img)
        st.image(im, caption="Empaque", use_container_width=True)
        m = image_metrics(im)
        ps = pack_scores_from_metrics(m)
        sh = shelf_scores(ps, m)
        conexion_pack = float(sh["choice"])
        pA, pB, pC, pD = st.columns(4)
        pA.metric("Legibilidad", f"{ps['legibility']}/100")
        pB.metric("Pop", f"{ps['pop']}/100")
        pC.metric("Claridad", f"{ps['clarity']}/100")
        pD.metric("Elección 3s", f"{conexion_pack:.1f}/100")
    else:
        ps = {"legibility": 60.0, "pop": 60.0, "clarity": 60.0}
        sh = {"attention": 60.0, "recall": 60.0, "choice": 60.0, "emotion": 60.0}
        conexion_pack = 60.0

    claim_opts = CLAIMS_LIBRARY.get(segmento, [])
    claims_sel = st.multiselect("Claims (elige 2-3)", claim_opts, default=claim_opts[:2], key="sim_claims")
    cscore = claims_score(claims_sel)
    st.metric("Claims Score", f"{cscore:.1f}/100")

    # Conexión final (proxy)
    conexion_score = clip(0.45*demanda + 0.35*conexion_pack + 0.20*cscore, 0, 100)

    row = {
        "precio": float(precio),
        "competencia": float(competencia),
        "demanda": float(demanda),
        "tendencia": float(tendencia),
        "margen_pct": float(margen),
        "conexion_score": float(conexion_score),
        "rating_conexion": 7.0,
        "sentiment_score": 1.0,
        "marca": str(marca).lower(),
        "canal": str(canal).lower(),
    }
    entrada = pd.DataFrame([row])

    s1, s2, s3 = st.columns(3)
    s1.metric("Conexión final", f"{conexion_score:.1f}/100")
    s2.metric("Pack choice (3s)", f"{conexion_pack:.1f}/100")
    s3.metric("Claims", f"{len(claims_sel)}")

    if st.button("🚀 Simular", key="sim_btn"):
        prob = float(success_model.predict_proba(entrada)[0][1])
        ventas = float(sales_model.predict(entrada)[0])
        ventas = max(0.0, ventas)

        # ROI FINANCIERO + UNIDADES
        ventas_u = float(ventas)
        precio_u = float(precio)
        margen_pct_u = float(margen)

        ingresos = ventas_u * precio_u
        utilidad_bruta = ventas_u * (precio_u * (margen_pct_u / 100.0))
        margen_unitario = precio_u * (margen_pct_u / 100.0)

        st.markdown("## 🎯 Resultado simulación")
        r1, r2, r3 = st.columns(3)
        r1.metric("Prob. Éxito", f"{prob*100:.2f}%")
        r2.metric("Ventas (u.)", f"{ventas_u:,.0f}")
        r3.metric("MAE ref (u.)", f"{MAE:,.0f}")

        st.markdown("### 💰 Unit economics (Simulación)")
        u1, u2, u3, u4 = st.columns(4)
        u1.metric("Ingresos ($)", f"${ingresos:,.0f}")
        u2.metric("Utilidad bruta ($)", f"${utilidad_bruta:,.0f}")
        u3.metric("Margen unitario ($)", f"${margen_unitario:.2f}")
        u4.metric("Margen %", f"{margen_pct_u:.1f}%")

        st.markdown("### 🎯 ROI (Financiero + Unidades)")
        rr1, rr2, rr3, rr4 = st.columns(4)

        inversion = rr1.number_input("Inversión ($) (opcional)", 0.0, 1e12, 0.0, step=1000.0, key="sim_inversion")
        meta_unidades = rr2.number_input("Meta unidades (opcional)", 0.0, 1e12, 0.0, step=100.0, key="sim_meta_u")
        baseline_unidades = rr3.number_input(
            "Baseline unidades (opcional)",
            0.0, 1e12,
            float(np.median(df["ventas_unidades"])),
            step=100.0,
            key="sim_baseline_u"
        )

        t1, t2, t3 = st.columns(3)
        cumplimiento = None
        uplift_vs_base = None
        roi_fin = None

        if meta_unidades > 0:
            cumplimiento = ventas_u / meta_unidades
            t1.metric("Cumplimiento vs meta", f"{cumplimiento*100:.1f}%")
        else:
            t1.metric("Cumplimiento vs meta", "—")

        if baseline_unidades > 0:
            uplift_vs_base = (ventas_u - baseline_unidades) / baseline_unidades
            t2.metric("Uplift vs baseline", f"{uplift_vs_base*100:.1f}%")
        else:
            t2.metric("Uplift vs baseline", "—")

        t3.metric("Ventas predichas", f"{ventas_u:,.0f} u.")

        if inversion > 0:
            roi_fin = (utilidad_bruta - inversion) / inversion
            payback_units = inversion / max(margen_unitario, 1e-6)
            rr4.metric("ROI financiero", f"{roi_fin*100:.1f}%")
            st.caption(f"Payback aprox.: **{payback_units:,.0f} u.** (con margen unitario).")
        else:
            rr4.metric("ROI financiero", "—")
            st.caption("Tip: mete inversión para calcular ROI financiero y payback.")

        st.markdown("### 📌 Inputs usados")
        st.dataframe(entrada, use_container_width=True)

        st.session_state.last_sim = {
            **row,
            "claims": claims_sel,
            "claims_score": float(cscore),
            "pack_choice": float(conexion_pack),
            "prob_exito": float(prob),
            "ventas_unidades": float(ventas_u),
            "ingresos": float(ingresos),
            "utilidad_bruta": float(utilidad_bruta),
            "roi_fin": float(roi_fin) if roi_fin is not None else None,
            "cumplimiento_meta": float(cumplimiento) if cumplimiento is not None else None,
            "uplift_vs_baseline": float(uplift_vs_base) if uplift_vs_base is not None else None,
        }

# ============================================================
# 📊 INSIGHTS ✅ (fix Altair SchemaValidationError)
# ============================================================
with tab_ins:
    st.subheader("📊 Insights (rankings + distribuciones)")

    left, right = st.columns(2)

    with left:
        st.markdown("**Ranking por marca (Conexión promedio)**")
        ins = df.groupby("marca")["conexion_score"].mean().sort_values(ascending=False).round(2)
        st.dataframe(ins.to_frame("conexion_score"), use_container_width=True)

        st.markdown("**Ranking por marca (Éxito %)**")
        ex = (df.groupby("marca")["exito"].mean() * 100).sort_values(ascending=False).round(1)
        st.dataframe(ex.to_frame("exito_%"), use_container_width=True)

    with right:
        st.markdown("**Ranking marca + canal (Ventas promedio)**")
        vm = df.groupby(["marca", "canal"])["ventas_unidades"].mean().sort_values(ascending=False).round(0)
        st.dataframe(vm.head(25).to_frame("ventas_unidades"), use_container_width=True)

    st.divider()
    d1, d2 = st.columns(2)

    def _bar_from_bins(bin_counts: pd.Series, title: str):
        st.markdown(f"**{title}**")
        dfp = bin_counts.reset_index()
        dfp = dfp.iloc[:, :2].copy()
        dfp.columns = ["bucket", "count"]
        dfp["bucket"] = dfp["bucket"].astype(str)
        st.bar_chart(dfp.set_index("bucket"), use_container_width=True)

    with d1:
        bins = pd.cut(df["conexion_score"], [0, 20, 40, 60, 80, 100], include_lowest=True)
        dist = bins.value_counts().sort_index()
        _bar_from_bins(dist, "Distribución: Conexión emocional (bucket)")

    with d2:
        bins2 = pd.cut(df["ventas_unidades"].clip(0, 40000), [0, 2000, 5000, 10000, 20000, 40000], include_lowest=True)
        dist2 = bins2.value_counts().sort_index()
        _bar_from_bins(dist2, "Distribución: Ventas unidades (bucket)")

# ============================================================
# 📦 PACK VISION+
# ============================================================
with tab_pack:
    st.subheader("📦 Pack Vision+ (pack suelto + heatmap + quick wins)")

    img = st.file_uploader("Sube imagen del empaque", type=["png","jpg","jpeg"], key="pack_upl")

    if not img:
        st.info("Sube un pack para ver scores + heatmap + quick wins.")
    else:
        im = Image.open(img)
        st.image(im, caption="Empaque", use_container_width=True)

        m = image_metrics(im)
        ps = pack_scores_from_metrics(m)
        sh = shelf_scores(ps, m)

        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Legibilidad", f"{ps['legibility']}/100")
        a2.metric("Shelf Pop", f"{ps['pop']}/100")
        a3.metric("Claridad", f"{ps['clarity']}/100")
        a4.metric("Elección (3s)", f"{sh['choice']:.1f}/100")

        st.markdown("### Heatmap (proxy visual)")
        hm = simple_heatmap(im)
        st.image(hm, caption="Heatmap proxy", use_container_width=True)

        st.markdown("### Quick wins")
        wins = []
        if ps["legibility"] < 60: wins.append("Sube contraste texto/fondo y tipografía más gruesa.")
        if ps["clarity"] < 60: wins.append("Reduce ruido visual y deja aire; 2–3 claims máximo.")
        if ps["pop"] < 60: wins.append("Agrega color acento / jerarquía fuerte del beneficio principal.")
        if not wins: wins.append("Está sólido. Ajusta jerarquía: Marca → Beneficio → Variedad → Credencial.")
        for w in wins:
            st.write("•", w)

# ============================================================
# 🧲 SHELF & EMOTION (3s)
# ============================================================
with tab_shelf:
    st.subheader("🧲 Shelf & Emotion (3s)")
    st.caption("Pack suelto vs competidores o foto de anaquel con ROIs manuales (ligero).")

    mode = st.radio("Modo", ["Pack suelto vs competidores", "Foto de anaquel + ROIs"], horizontal=True, key="shelf_mode")

    beta_att = st.slider("Peso Atención", 0.0, 2.0, 1.0, 0.05, key="b_att")
    beta_rec = st.slider("Peso Recordación", 0.0, 2.0, 0.8, 0.05, key="b_rec")
    beta_emo = st.slider("Peso Emoción", 0.0, 2.0, 0.7, 0.05, key="b_emo")
    beta_price = st.slider("Penalización Precio", -0.02, 0.0, -0.005, 0.0005, key="b_price")
    beta = {"att": beta_att, "rec": beta_rec, "emo": beta_emo, "price": beta_price}

    if mode == "Pack suelto vs competidores":
        your_pack = st.file_uploader("Tu pack", type=["png","jpg","jpeg"], key="your_pack")
        comp_packs = st.file_uploader("Competidores (2–6)", type=["png","jpg","jpeg"], accept_multiple_files=True, key="comp_packs")

        if your_pack and comp_packs:
            items = []

            im = Image.open(your_pack)
            m = image_metrics(im); ps = pack_scores_from_metrics(m); sh = shelf_scores(ps,m)
            items.append({"name":"TU_PACK","attention":sh["attention"],"recall":sh["recall"],"emotion":sh["emotion"],"choice":sh["choice"],"price":0.0})

            for i,f in enumerate(comp_packs[:6], start=1):
                cim = Image.open(f)
                cm = image_metrics(cim); cps = pack_scores_from_metrics(cm); csh = shelf_scores(cps,cm)
                items.append({"name":f"COMP_{i}","attention":csh["attention"],"recall":csh["recall"],"emotion":csh["emotion"],"choice":csh["choice"],"price":0.0})

            d = pd.DataFrame(items)

            st.markdown("### Precios (para MNL)")
            d.loc[d["name"]=="TU_PACK","price"] = st.number_input("Precio TU_PACK", 0.0, 9999.0, 0.0, key="price_you")
            for i in range(1, len(d)):
                d.loc[i,"price"] = st.number_input(f"Precio {d.loc[i,'name']}", 0.0, 9999.0, 0.0, key=f"price_{i}")

            st.markdown("### Simulación de elección MNL")
            ranked = mnl_choice(d, beta)
            st.dataframe(ranked[["name","choice_prob","attention","recall","emotion","price"]], use_container_width=True)

        else:
            st.info("Sube tu pack y al menos 2 competidores.")

    else:
        shelf_img = st.file_uploader("Foto de anaquel", type=["png","jpg","jpeg"], key="shelf_photo")
        if shelf_img:
            shelf_im = Image.open(shelf_img)
            st.image(shelf_im, caption="Anaquel", use_container_width=True)

            def crop_roi(im, roi):
                W,H = im.size
                x1 = int(W*roi[0]/100); y1 = int(H*roi[1]/100)
                x2 = int(W*roi[2]/100); y2 = int(H*roi[3]/100)
                x1 = max(0,min(W-1,x1)); x2 = max(1,min(W,x2))
                y1 = max(0,min(H-1,y1)); y2 = max(1,min(H,y2))
                if x2 <= x1+2 or y2 <= y1+2:
                    return None
                return im.crop((x1,y1,x2,y2))

            labels = ["TU_PACK","COMP_1","COMP_2","COMP_3"]
            rois = []
            cols = st.columns(4)
            for i in range(4):
                with cols[i]:
                    st.markdown(f"**{labels[i]} ROI**")
                    x1 = st.number_input("x1%", 0.0, 100.0, 5.0, key=f"roi_{i}_x1")
                    y1 = st.number_input("y1%", 0.0, 100.0, 5.0, key=f"roi_{i}_y1")
                    x2 = st.number_input("x2%", 0.0, 100.0, 25.0, key=f"roi_{i}_x2")
                    y2 = st.number_input("y2%", 0.0, 100.0, 40.0, key=f"roi_{i}_y2")
                    rois.append((x1,y1,x2,y2))

            items = []
            crops = []
            for i,roi in enumerate(rois):
                crop = crop_roi(shelf_im, roi)
                if crop is None:
                    continue
                crops.append((labels[i], crop))
                m = image_metrics(crop); ps = pack_scores_from_metrics(m); sh = shelf_scores(ps,m)
                items.append({"name":labels[i],"attention":sh["attention"],"recall":sh["recall"],"emotion":sh["emotion"],"choice":sh["choice"],"price":0.0})

            if items:
                st.markdown("### ROIs recortados")
                cols2 = st.columns(min(4,len(crops)))
                for j,(lab,imgc) in enumerate(crops):
                    cols2[j].image(imgc, caption=lab, use_container_width=True)

                d = pd.DataFrame(items)

                st.markdown("### Precios (para MNL)")
                for i in range(len(d)):
                    d.loc[i,"price"] = st.number_input(f"Precio {d.loc[i,'name']}", 0.0, 9999.0, 0.0, key=f"shelf_price_{i}")

                ranked = mnl_choice(d, beta)
                st.dataframe(ranked[["name","choice_prob","attention","recall","emotion","price"]], use_container_width=True)
            else:
                st.warning("No pude recortar ROIs válidos. Ajusta coordenadas.")
        else:
            st.info("Sube una foto de anaquel.")

# ============================================================
# 🧊 PRODUCTO NUEVO (simple)
# ============================================================
with tab_new:
    st.subheader("🧊 Producto Nuevo (Cold Start)")
    st.caption("Predice éxito y ventas sin histórico usando atributos + pack + claims (proxy).")

    categorias = sorted(df["categoria"].unique().tolist())
    canales = sorted(df["canal"].unique().tolist())

    c1,c2,c3 = st.columns(3)
    categoria = c1.selectbox("Categoría comparable", categorias, key="new_cat")
    canal = c2.selectbox("Canal", canales, key="new_canal")
    segmento = c3.selectbox("Segmento", ["fit","kids","premium","value"], key="new_seg")

    b1,b2,b3,b4,b5 = st.columns(5)
    precio = b1.number_input("Precio", 1.0, 9999.0, float(df["precio"].median()), step=1.0, key="new_precio")
    margen = b2.slider("Margen %", 0, 90, int(np.clip(df["margen_pct"].median(),0,90)), key="new_margen")
    competencia = b3.slider("Competencia (1-10)", 1, 10, int(df["competencia"].median()), key="new_comp")
    demanda = b4.slider("Demanda (0-100)", 0, 100, int(df["demanda"].median()), key="new_dem")
    tendencia = b5.slider("Tendencia (0-100)", 0, 100, int(df["tendencia"].median()), key="new_tend")

    claim_opts = CLAIMS_LIBRARY.get(segmento, [])
    claims_sel = st.multiselect("Claims (2-3)", claim_opts, default=claim_opts[:2], key="new_claims")
    cscore = claims_score(claims_sel)

    img = st.file_uploader("Sube empaque (opcional)", type=["png","jpg","jpeg"], key="new_pack")
    if img:
        im = Image.open(img)
        st.image(im, caption="Empaque nuevo", use_container_width=True)
        m = image_metrics(im)
        ps = pack_scores_from_metrics(m)
        sh = shelf_scores(ps, m)
        pack_choice = float(sh["choice"])
        pack_emotion = float(sh["emotion"])
    else:
        pack_choice = 60.0
        pack_emotion = 60.0

    conexion_score = clip(0.45*demanda + 0.35*pack_choice + 0.20*cscore, 0, 100)

    entrada = pd.DataFrame([{
        "precio": float(precio),
        "competencia": float(competencia),
        "demanda": float(demanda),
        "tendencia": float(tendencia),
        "margen_pct": float(margen),
        "conexion_score": float(conexion_score),
        "rating_conexion": 7.0,
        "sentiment_score": 1.0,
        "marca": "nueva",
        "canal": str(canal).lower(),
    }])

    prob = float(success_model.predict_proba(entrada)[0][1])
    ventas = float(sales_model.predict(entrada)[0])

    st.markdown("## 🎯 Resultado (Producto Nuevo)")
    o1,o2,o3,o4 = st.columns(4)
    o1.metric("Prob. éxito", f"{prob*100:.2f}%")
    o2.metric("Ventas (punto)", f"{ventas:,.0f} u.")
    o3.metric("Pack choice (3s)", f"{pack_choice:.1f}/100")
    o4.metric("Claims score", f"{cscore:.1f}/100")

    st.session_state.last_new = {
        "categoria": categoria,
        "canal": canal,
        "segmento": segmento,
        "precio": float(precio),
        "margen_pct": float(margen),
        "competencia": float(competencia),
        "demanda": float(demanda),
        "tendencia": float(tendencia),
        "claims": claims_sel,
        "claims_score": float(cscore),
        "pack_choice": float(pack_choice),
        "pack_emotion": float(pack_emotion),
        "conexion_score_proxy": float(conexion_score),
        "prob_exito": float(prob),
        "ventas_point": float(ventas),
    }

# ============================================================
# 📄 REPORTE (TXT + CSV inputs)
# ============================================================
with tab_rep:
    st.subheader("📄 Reporte Ejecutivo descargable")

    def build_report():
        lines = []
        lines.append("AI PRODUCT INTELLIGENCE PLATFORM — REPORTE EJECUTIVO")
        lines.append(f"Fecha UTC: {datetime.utcnow().isoformat()}")
        lines.append("")
        lines.append("=== MÉTRICAS MODELO ===")
        lines.append(f"Registros: {len(df)}")
        lines.append(f"ACC: {ACC:.4f}")
        lines.append(f"AUC: {AUC:.4f}")
        lines.append(f"MAE ventas: {MAE:.2f} u.")
        lines.append("")

        if "last_sim" in st.session_state:
            s = st.session_state.last_sim
            lines.append("=== ÚLTIMA SIMULACIÓN ===")
            for k,v in s.items():
                lines.append(f"{k}: {v}")
            lines.append("")

        if "last_new" in st.session_state:
            n = st.session_state.last_new
            lines.append("=== ÚLTIMO PRODUCTO NUEVO ===")
            for k,v in n.items():
                lines.append(f"{k}: {v}")
            lines.append("")

        return "\n".join(lines)

    rep = build_report()
    st.download_button(
        "📥 Descargar reporte (TXT)",
        data=rep,
        file_name="reporte_product_intelligence.txt",
        mime="text/plain",
        key="dl_report_txt"
    )

    rows = []
    if "last_sim" in st.session_state:
        rows.append({"type":"simulador", **st.session_state.last_sim})
    if "last_new" in st.session_state:
        rows.append({"type":"producto_nuevo", **st.session_state.last_new})

    if rows:
        outdf = pd.DataFrame(rows)
        st.download_button(
            "📥 Descargar inputs (CSV)",
            data=df_to_csv_bytes(outdf),
            file_name="reporte_inputs.csv",
            mime="text/csv",
            key="dl_report_csv"
        )
        st.dataframe(outdf, use_container_width=True)
    else:
        st.info("Aún no hay simulaciones guardadas.")

# ============================================================
# 📂 DATOS
# ============================================================
with tab_data:
    st.subheader("📂 Datos")
    st.download_button(
        "📥 Descargar dataset",
        data=df_to_csv_bytes(df),
        file_name="dataset_con_ventas.csv",
        mime="text/csv",
        key="dl_dataset"
    )
    st.dataframe(df.head(300), use_container_width=True)

# ============================================================
# 🧠 DIAGNÓSTICO
# ============================================================
with tab_diag:
    st.subheader("🧠 Diagnóstico")
    cm_df = pd.DataFrame(CM, index=["Real 0","Real 1"], columns=["Pred 0","Pred 1"])
    st.dataframe(cm_df, use_container_width=True)
    st.write(f"MAE ventas: **{MAE:,.0f}** unidades.")