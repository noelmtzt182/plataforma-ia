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
# BLOQUE 1 — CORE (Imports + Helpers + Loaders + Models)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, mean_absolute_error


# ----------------------------
# Config general
# ----------------------------
st.set_page_config(
    page_title="Plataforma IA | Producto + Empaque + Claims + Market",
    layout="wide"
)

DATA_PATH_DEFAULT = "mercado_cereales_5000_con_ventas.csv"
MARKET_PATH_DEFAULT = "market_intel.csv"


# ----------------------------
# Helpers básicos
# ----------------------------
def clip(v, a, b):
    return float(max(a, min(b, v)))

def df_to_csv_bytes(df_: pd.DataFrame) -> bytes:
    return df_.to_csv(index=False).encode("utf-8")

def _clean_str_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()

def safe_percent(x: float) -> str:
    return f"{x*100:.2f}%"

def bar_df_from_value_counts(vc: pd.Series) -> pd.DataFrame:
    """Convierte value_counts() a DataFrame simple para evitar errores de Altair."""
    d = vc.reset_index()
    if d.shape[1] >= 2:
        d = d.iloc[:, :2].copy()
        d.columns = ["bucket", "count"]
    else:
        d = pd.DataFrame({"bucket": ["(sin buckets)"], "count": [0]})
    d["bucket"] = d["bucket"].astype(str)
    d["count"] = pd.to_numeric(d["count"], errors="coerce").fillna(0)
    return d


# ----------------------------
# Dataset requerido (base)
# ----------------------------
REQUIRED_BASE = {
    "marca", "categoria", "canal",
    "precio", "costo", "margen", "margen_pct",
    "competencia", "demanda", "tendencia", "estacionalidad",
    "rating_conexion", "comentario", "sentiment_score",
    "conexion_score", "conexion_alta", "score_latente", "exito"
}

REQUIRED_SALES = {"ventas_unidades", "ventas_ingresos", "utilidad"}


# ============================================================
# LOADERS
# ============================================================

@st.cache_data
def load_data(path_or_file) -> pd.DataFrame:
    df = pd.read_csv(path_or_file).copy()
    df.columns = [c.strip().lower() for c in df.columns]

    # Normaliza strings
    for c in ["marca", "categoria", "canal", "estacionalidad", "comentario"]:
        if c in df.columns:
            df[c] = _clean_str_series(df[c])

    missing = sorted(list(REQUIRED_BASE - set(df.columns)))
    if missing:
        raise ValueError(f"Faltan columnas base en el CSV: {missing}")

    # Numéricas
    num_cols = [
        "precio", "costo", "margen", "margen_pct",
        "competencia", "demanda", "tendencia",
        "rating_conexion", "sentiment_score",
        "conexion_score", "conexion_alta",
        "score_latente", "exito"
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ["ventas_unidades", "ventas_ingresos", "utilidad"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Limpieza mínima
    df = df.dropna(
        subset=[
            "marca", "categoria", "canal",
            "precio", "competencia", "demanda", "tendencia",
            "margen_pct", "conexion_score", "rating_conexion",
            "sentiment_score", "exito"
        ]
    )

    df["exito"] = df["exito"].astype(int)
    return df


# ============================================================
# Market Intelligence Loader (Módulo A)
# ============================================================

def load_market_intel(file_or_path) -> pd.DataFrame:
    mdf = pd.read_csv(file_or_path).copy()
    mdf.columns = [c.strip().lower() for c in mdf.columns]

    required = {
        "categoria", "marca", "canal",
        "precio", "competencia_skus", "demanda_idx", "tendencia_idx"
    }

    missing = sorted(list(required - set(mdf.columns)))
    if missing:
        raise ValueError(f"market_intel.csv: faltan columnas requeridas: {missing}")

    for c in ["categoria", "marca", "canal"]:
        mdf[c] = mdf[c].astype(str).str.lower().str.strip()

    for c in ["precio", "competencia_skus", "demanda_idx", "tendencia_idx"]:
        mdf[c] = pd.to_numeric(mdf[c], errors="coerce")

    # Opcionales (si existen)
    for c in ["share_proxy", "rating_promedio", "sentiment_promedio"]:
        if c in mdf.columns:
            mdf[c] = pd.to_numeric(mdf[c], errors="coerce")

    mdf = mdf.dropna(subset=["categoria", "marca", "canal", "precio", "competencia_skus", "demanda_idx", "tendencia_idx"])
    mdf["precio"] = mdf["precio"].clip(lower=0)
    mdf["competencia_skus"] = mdf["competencia_skus"].clip(lower=0)
    mdf["demanda_idx"] = mdf["demanda_idx"].clip(lower=0)
    mdf["tendencia_idx"] = mdf["tendencia_idx"].clip(lower=0)
    return mdf


# ============================================================
# Claims Engine (ligero)
# ============================================================

CLAIMS_LIBRARY = {
    "fit": [
        ("Alto en proteína", 0.90),
        ("Sin azúcar añadida", 0.88),
        ("Alto en fibra", 0.86),
        ("Integral", 0.80),
        ("Sin colorantes artificiales", 0.72),
    ],
    "kids": [
        ("Con vitaminas y minerales", 0.86),
        ("Sabor chocolate", 0.82),
        ("Energía para su día", 0.78),
        ("Hecho con granos", 0.74),
        ("Sin conservadores", 0.70),
    ],
    "premium": [
        ("Ingredientes seleccionados", 0.82),
        ("Hecho con avena real", 0.76),
        ("Calidad premium", 0.70),
        ("Receta artesanal", 0.64),
    ],
    "value": [
        ("Rinde más", 0.78),
        ("Ideal para la familia", 0.72),
        ("Gran sabor a mejor precio", 0.74),
        ("Económico y práctico", 0.66),
    ],
}

CANAL_CLAIM_BOOST = {
    "retail": {
        "Sin azúcar añadida": 1.04,
        "Alto en fibra": 1.03,
        "Integral": 1.02
    },
    "marketplace": {
        "Alto en proteína": 1.05,
        "Sin colorantes artificiales": 1.04,
        "Ingredientes seleccionados": 1.03
    },
}

def recommend_claims(segment: str, canal: str, max_claims: int = 6):
    seg = str(segment).lower().strip()
    can = str(canal).lower().strip()

    items = CLAIMS_LIBRARY.get(seg, [])[:]
    scored = []
    for claim, base in items:
        boost = CANAL_CLAIM_BOOST.get(can, {}).get(claim, 1.0)
        scored.append((claim, float(base) * float(boost)))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:max_claims]

def claims_score(selected_claims, canal: str) -> float:
    if not selected_claims:
        return 0.0
    can = str(canal).lower().strip()

    boosts = [CANAL_CLAIM_BOOST.get(can, {}).get(c, 1.0) for c in selected_claims]
    base = float(np.mean(boosts))

    # Penaliza demasiados claims
    n = len(selected_claims)
    clarity_penalty = 1.0 if n <= 3 else max(0.65, 1.0 - 0.12 * (n - 3))

    score = 75.0 * base * clarity_penalty
    return float(np.clip(score, 0, 100))


# ============================================================
# Pack emotion proxy (ligero)
# ============================================================
def pack_emotion_score(pack_legibility: float, pack_pop: float, pack_clarity: float, claims_score_val: float, copy_tone: int):
    visual = 0.40*(pack_pop/100.0) + 0.30*(pack_clarity/100.0) + 0.15*(pack_legibility/100.0)
    claims = 0.15*(claims_score_val/100.0)
    tone_boost = 0.06 if copy_tone > 0 else (-0.06 if copy_tone < 0 else 0.0)
    score = (visual + claims + tone_boost) * 100.0
    return float(np.clip(score, 0, 100))


# ============================================================
# MODELS (Éxito + Ventas) — robusto contra errores de longitud
# ============================================================
@st.cache_resource
def train_models(df: pd.DataFrame):
    features = [
        "precio", "competencia", "demanda", "tendencia", "margen_pct",
        "conexion_score", "rating_conexion", "sentiment_score",
        "marca", "canal"
    ]

    X = df[features].copy()
    y_cls = df["exito"].astype(int).copy()

    num_cols = [
        "precio", "competencia", "demanda", "tendencia", "margen_pct",
        "conexion_score", "rating_conexion", "sentiment_score"
    ]
    cat_cols = ["marca", "canal"]

    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    # ---- Clasificación: éxito
    clf = Pipeline(steps=[
        ("preprocessor", pre),
        ("model", RandomForestClassifier(
            n_estimators=350, random_state=42, class_weight="balanced_subsample"
        )),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cls, test_size=0.2, random_state=42, stratify=y_cls
    )

    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    proba = clf.predict_proba(X_test)[:, 1]
    ACC = accuracy_score(y_test, pred)
    AUC = roc_auc_score(y_test, proba)
    CM = confusion_matrix(y_test, pred)

    # ---- Regresión: ventas (si existen)
    if not REQUIRED_SALES.issubset(set(df.columns)):
        raise ValueError(f"CSV sin ventas. Faltan: {sorted(list(REQUIRED_SALES - set(df.columns)))}")

    y_reg = df["ventas_unidades"].astype(float).copy()

    reg = Pipeline(steps=[
        ("preprocessor", pre),
        ("model", RandomForestRegressor(n_estimators=350, random_state=42)),
    ])

    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )

    # ✅ IMPORTANTÍSIMO: entrenar con yr_train (no con df completa)
    reg.fit(Xr_train, yr_train)
    yhat = reg.predict(Xr_test)
    MAE = mean_absolute_error(yr_test, yhat)

    return clf, reg, ACC, AUC, CM, MAE

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

# ============================================================
# 📈 Market Intelligence — Loader
# ============================================================

MARKET_PATH_DEFAULT = "market_intel.csv"

st.sidebar.title("📈 Market Intelligence")

market_file = st.sidebar.file_uploader(
    "Sube market_intel.csv",
    type=["csv"],
    key="market_uploader"
)

market_df = None

try:
    if market_file is not None:
        market_df = load_market_intel(market_file)
        st.sidebar.success("Market Intel cargado ✅")
    elif Path(MARKET_PATH_DEFAULT).exists():
        market_df = load_market_intel(MARKET_PATH_DEFAULT)
        st.sidebar.info("Market Intel desde repo")
    else:
        st.sidebar.warning("Sin Market Intel cargado")
except Exception as e:
    st.sidebar.error(f"Error market intel: {e}")

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
tab_sim, tab_ins, tab_pack, tab_shelf, tab_new, tab_market, tab_rep, tab_data, tab_diag = st.tabs([
    "🧪 Simulador",
    "📊 Insights",
    "📦 Pack Vision+",
    "🧲 Shelf & Emotion",
    "🧊 Producto Nuevo",
    "📈 Market Intelligence",
    "📄 Reporte",
    "📂 Datos",
    "🧠 Diagnóstico"
])
# ============================================================
# 🧪 Simulador (con ROI persistente)
# ============================================================
with tab_sim:
    st.subheader("🧪 Simulador What-If (éxito + ventas + pack + claims + conexión)")
    st.caption("Tip: Da click en **Simular** y luego ajusta ROI (inversión/meta/baseline) sin que se borre el resultado.")

    # ------------------------------------------------
    # Persistencia: guarda último resultado de simulación
    # ------------------------------------------------
    if "last_sim" not in st.session_state:
        st.session_state.last_sim = None

    marcas = sorted(df["marca"].unique().tolist())
    canales = sorted(df["canal"].unique().tolist())

    # =========================
    # Inputs principales
    # =========================
    c1, c2, c3 = st.columns(3)
    marca = c1.selectbox("Marca", marcas, 0, key="sim_marca")
    canal = c2.selectbox("Canal", canales, 0, key="sim_canal")
    segmento = c3.selectbox("Segmento objetivo", ["fit", "kids", "premium", "value"], 0, key="sim_segmento")

    st.markdown("### Variables de negocio")
    b1, b2, b3, b4, b5 = st.columns(5)

    precio = b1.number_input("Precio", min_value=1.0, max_value=99999.0, value=float(df["precio"].median()), step=1.0, key="sim_precio")
    competencia = b2.slider("Competencia (1-10)", 1, 10, int(df["competencia"].median()), key="sim_competencia")
    demanda = b3.slider("Demanda (10-100)", 10, 100, int(df["demanda"].median()), key="sim_demanda")
    tendencia = b4.slider("Tendencia (20-100)", 20, 100, int(df["tendencia"].median()), key="sim_tendencia")
    margen_pct = b5.slider("Margen %", 0, 90, int(np.clip(df["margen_pct"].median(), 0, 90)), key="sim_margen_pct")

    st.markdown("### Empaque + Claims")
    p1, p2, p3 = st.columns(3)
    pack_legibility_score = p1.slider("Pack legibilidad (0-100)", 0, 100, 65, key="sim_pack_legibility")
    pack_shelf_pop_score = p2.slider("Pack shelf pop (0-100)", 0, 100, 70, key="sim_pack_pop")
    pack_clarity_score = p3.slider("Pack claridad (0-100)", 0, 100, 65, key="sim_pack_clarity")

    # Claims recomendados
    recs = recommend_claims(segmento, canal, max_claims=6)
    claim_options = [c for c, _ in recs]
    selected_claims = st.multiselect(
        "Selecciona claims (ideal 2-3)",
        claim_options,
        default=claim_options[:2],
        key="sim_claims",
    )
    cscore = claims_score(selected_claims, canal)

    # Copy tone (proxy)
    copy = st.text_input("Copy corto (opcional)", value="Energía y nutrición para tu día", key="sim_copy")
    pos_kw = ["energía", "nutrición", "saludable", "delicioso", "me encanta", "premium", "calidad", "proteína", "fibra"]
    neg_kw = ["caro", "no", "malo", "rechazo", "no me gusta", "pésimo", "horrible"]
    t = copy.lower()
    tone = 0
    if any(k in t for k in pos_kw):
        tone += 1
    if any(k in t for k in neg_kw):
        tone -= 1
    copy_tone = 1 if tone > 0 else (-1 if tone < 0 else 0)

    # Emotion / conexión (proxy)
    pack_emotion = pack_emotion_score(
        pack_legibility_score,
        pack_shelf_pop_score,
        pack_clarity_score,
        cscore,
        copy_tone
    )

    uplift = clip((pack_emotion - 50) / 50, -0.35, 0.35)

    rating_conexion = st.slider("Rating conexión producto (1-10)", 1, 10, 7, key="sim_rating")
    sentiment_score = st.select_slider("Sentimiento del producto (-1/0/1)", options=[-1, 0, 1], value=1, key="sim_sentiment")

    base_conexion = (rating_conexion / 10) * 70 + sentiment_score * 15 + 5
    conexion_score = clip(base_conexion * (1 + uplift), 0, 100)

    # Mostrar scores intermedios
    s1, s2, s3 = st.columns(3)
    s1.metric("Claims Score", f"{cscore:.1f}/100")
    s2.metric("Emotion Pack Score", f"{pack_emotion:.1f}/100")
    s3.metric("Conexión final", f"{conexion_score:.1f}/100")

    # =========================
    # Entrada al modelo
    # =========================
    entrada = pd.DataFrame([{
        "precio": float(precio),
        "competencia": float(competencia),
        "demanda": float(demanda),
        "tendencia": float(tendencia),
        "margen_pct": float(margen_pct),
        "conexion_score": float(conexion_score),
        "rating_conexion": float(rating_conexion),
        "sentiment_score": float(sentiment_score),
        "marca": str(marca).lower(),
        "canal": str(canal).lower(),
    }])

    # =========================
    # Botón simular (GUARDA en session_state)
    # =========================
    if st.button("🚀 Simular", key="sim_btn"):
        prob = float(success_model.predict_proba(entrada)[0][1])
        ventas_u = float(sales_model.predict(entrada)[0])
        ventas_u = max(0.0, ventas_u)

        precio_u = float(precio)
        margen_pct_u = float(margen_pct)

        ingresos = ventas_u * precio_u
        utilidad_bruta = ventas_u * (precio_u * (margen_pct_u / 100.0))
        margen_unitario = precio_u * (margen_pct_u / 100.0)

        # ✅ Persistir
        st.session_state.last_sim = {
            "prob": prob,
            "ventas_u": ventas_u,
            "precio": precio_u,
            "margen_pct": margen_pct_u,
            "ingresos": ingresos,
            "utilidad_bruta": utilidad_bruta,
            "margen_unitario": margen_unitario,
            "entrada": entrada.copy(),
        }

    # =========================
    # Render persistente: NO se borra al mover ROI
    # =========================
    st.divider()

    if st.session_state.last_sim is None:
        st.info("Da click en **Simular** para generar resultados. Después podrás ajustar ROI sin que se borre.")
    else:
        sim = st.session_state.last_sim

        st.markdown("## 🎯 Resultado simulación")
        r1, r2, r3 = st.columns(3)
        r1.metric("Prob. Éxito", f"{sim['prob']*100:.2f}%")
        r2.metric("Ventas predichas", f"{sim['ventas_u']:,.0f} u.")
        r3.metric("Precio", f"${sim['precio']:,.0f}")

        st.markdown("### 💰 Unit economics (Simulación)")
        u1, u2, u3, u4 = st.columns(4)
        u1.metric("Ingresos ($)", f"${sim['ingresos']:,.0f}")
        u2.metric("Utilidad bruta ($)", f"${sim['utilidad_bruta']:,.0f}")
        u3.metric("Margen unitario ($/u)", f"${sim['margen_unitario']:.2f}")
        u4.metric("Margen %", f"{sim['margen_pct']:.1f}%")

        st.markdown("### 🎯 ROI (Financiero + Unidades)")
        rr1, rr2, rr3, rr4 = st.columns(4)

        inversion = rr1.number_input("Inversión ($) (opcional)", 0.0, 1e12, 0.0, step=1000.0, key="roi_inv")
        meta_unidades = rr2.number_input("Meta unidades (opcional)", 0.0, 1e12, 0.0, step=100.0, key="roi_meta_u")
        baseline_unidades = rr3.number_input(
            "Baseline unidades (opcional)",
            0.0, 1e12,
            float(np.median(df["ventas_unidades"])) if "ventas_unidades" in df.columns else 0.0,
            step=100.0,
            key="roi_base_u"
        )

        # Cumplimiento
        if meta_unidades > 0:
            cumplimiento = sim["ventas_u"] / meta_unidades
            rr4.metric("Cumplimiento vs meta", f"{cumplimiento*100:.1f}%")
        else:
            rr4.metric("Cumplimiento vs meta", "—")

        # Uplift unidades
        if baseline_unidades > 0:
            uplift = (sim["ventas_u"] - baseline_unidades) / baseline_unidades
            st.metric("Uplift vs baseline", f"{uplift*100:.1f}%")
        else:
            st.metric("Uplift vs baseline", "—")

        # ROI financiero
        if inversion > 0:
            roi_fin = (sim["utilidad_bruta"] - inversion) / inversion
            payback_units = inversion / max(sim["margen_unitario"], 1e-6)
            st.metric("ROI financiero", f"{roi_fin*100:.1f}%")
            st.caption(f"Payback aprox.: **{payback_units:,.0f} u.**")
        else:
            st.metric("ROI financiero", "—")

        st.markdown("### 📌 Inputs usados para predicción")
        st.dataframe(sim["entrada"], use_container_width=True)
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
# 📈 MARKET INTELLIGENCE TAB
# ============================================================
with tab_market:

    st.subheader("📈 Market Intelligence")

    if market_df is None:
        st.info("Sube market_intel.csv en sidebar")
        st.stop()

    st.markdown("### Benchmark precios")

    bench = (
        market_df
        .groupby(["marca","canal"])["precio"]
        .agg(p25=lambda x: np.percentile(x,25),
             p50=lambda x: np.percentile(x,50),
             p75=lambda x: np.percentile(x,75))
        .round(1)
        .reset_index()
    )

    st.dataframe(bench, use_container_width=True)

    st.markdown("### Competencia vs demanda")

    comp = (
        market_df
        .groupby("marca")[["competencia_skus","demanda_idx","tendencia_idx"]]
        .mean()
        .round(1)
    )

    st.dataframe(comp, use_container_width=True)
    st.bar_chart(comp[["demanda_idx"]])

    st.markdown("### White spaces (oportunidades)")

    m = market_df.copy()

    m["score"] = (
        0.4*m["demanda_idx"]/100 +
        0.3*m["tendencia_idx"]/100 +
        0.3*(1 - m["competencia_skus"]/m["competencia_skus"].max())
    )*100

    opp = (
        m.groupby(["categoria","canal"])["score"]
        .mean()
        .sort_values(ascending=False)
        .round(1)
        .to_frame("opportunity_score")
    )

    st.dataframe(opp, use_container_width=True)

    st.download_button(
        "📥 Descargar oportunidades",
        opp.to_csv().encode(),
        "oportunidades.csv"
    )
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