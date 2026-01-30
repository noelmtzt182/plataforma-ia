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
# BLOQUE 1 — CORE
# imports + helpers + loaders + claims + modelos
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

def safe_percent(x: float) -> str:
    return f"{x * 100:.2f}%"

def _clean_str_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()

def df_to_csv_bytes(df_: pd.DataFrame) -> bytes:
    return df_.to_csv(index=False).encode("utf-8")

def bar_df_from_value_counts(vc: pd.Series) -> pd.DataFrame:
    """
    Convierte value_counts() en DF con columnas estables para evitar
    SchemaValidationError / Altair issues.
    """
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

    for c in ["marca", "categoria", "canal", "estacionalidad", "comentario"]:
        if c in df.columns:
            df[c] = _clean_str_series(df[c])

    missing = sorted(list(REQUIRED_BASE - set(df.columns)))
    if missing:
        raise ValueError(f"Faltan columnas base en el CSV: {missing}")

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

    df = df.dropna(subset=[
        "marca", "categoria", "canal",
        "precio", "competencia", "demanda", "tendencia",
        "margen_pct", "conexion_score", "rating_conexion",
        "sentiment_score", "exito"
    ])

    df["exito"] = df["exito"].astype(int)
    return df


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

    mdf = mdf.dropna(subset=["categoria", "marca", "canal", "precio", "competencia_skus", "demanda_idx", "tendencia_idx"])
    return mdf


# ============================================================
# CLAIMS ENGINE (ROBUSTO)
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
        "Integral": 1.02,
    },
    "marketplace": {
        "Alto en proteína": 1.05,
        "Sin colorantes artificiales": 1.04,
        "Ingredientes seleccionados": 1.03,
    },
}

def _normalize_claim_items(items):
    """
    Acepta:
      - [("claim", 0.8), ...]
      - ["claim", ...]
      - [{"claim": "...", "score": 0.8}, ...]
    y devuelve: [("claim", float_score), ...]
    """
    norm = []
    if items is None:
        return norm

    for it in items:
        if isinstance(it, (tuple, list)) and len(it) >= 2:
            claim = str(it[0]).strip()
            try:
                score = float(it[1])
            except Exception:
                score = 0.70
            norm.append((claim, score))
            continue

        if isinstance(it, dict):
            claim = str(it.get("claim", "")).strip()
            sc = it.get("score", it.get("base", 0.70))
            try:
                score = float(sc)
            except Exception:
                score = 0.70
            if claim:
                norm.append((claim, score))
            continue

        if isinstance(it, str):
            norm.append((it.strip(), 0.70))
            continue

    return norm

def recommend_claims(segment: str, canal: str, max_claims: int = 6):
    seg = str(segment).lower().strip()
    can = str(canal).lower().strip()

    raw_items = CLAIMS_LIBRARY.get(seg, [])
    items = _normalize_claim_items(raw_items)

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

    boosts = []
    for c in selected_claims:
        c = str(c).strip()
        boosts.append(float(CANAL_CLAIM_BOOST.get(can, {}).get(c, 1.0)))

    base = float(np.mean(boosts)) if boosts else 1.0
    n = len(selected_claims)
    clarity_penalty = 1.0 if n <= 3 else max(0.65, 1.0 - 0.12 * (n - 3))

    score = 75.0 * base * clarity_penalty
    return float(np.clip(score, 0, 100))


# ============================================================
# PACK EMOTION (proxy ligero)
# ============================================================

def pack_emotion_score(pack_legibility: float, pack_pop: float, pack_clarity: float, claims_score_val: float, copy_tone: int):
    visual = 0.40 * (pack_pop / 100.0) + 0.30 * (pack_clarity / 100.0) + 0.15 * (pack_legibility / 100.0)
    claims = 0.15 * (claims_score_val / 100.0)
    tone_boost = 0.06 if copy_tone > 0 else (-0.06 if copy_tone < 0 else 0.0)
    score = (visual + claims + tone_boost) * 100.0
    return float(np.clip(score, 0, 100))


# ============================================================
# MODELOS — Éxito + Ventas (corregido)
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

    # Clasificación (éxito)
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

    # Regresión (ventas)
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

    # ✅ entrenar con yr_train (evita check_consistent_length)
    reg.fit(Xr_train, yr_train)
    yhat = reg.predict(Xr_test)
    MAE = mean_absolute_error(yr_test, yhat)

    return clf, reg, ACC, AUC, CM, MAE

# ============================================================
# BLOQUE 2 — SIDEBAR + CARGA + ENTRENAMIENTO
# ============================================================

st.sidebar.title("⚙️ Control")

# Limpiar cache (muy útil en Streamlit Cloud)
if st.sidebar.button("🔄 Limpiar cache"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.sidebar.success("Cache limpio ✅ (recarga la app)")

st.sidebar.divider()

# -------- Dataset principal
st.sidebar.subheader("📂 Dataset principal (ventas + éxito)")
uploaded_csv = st.sidebar.file_uploader(
    "Sube CSV principal",
    type=["csv"],
    key="uploader_main_csv"
)

try:
    if uploaded_csv is not None:
        df = load_data(uploaded_csv)
        st.sidebar.success("Dataset principal cargado ✅")
    else:
        if Path(DATA_PATH_DEFAULT).exists():
            df = load_data(DATA_PATH_DEFAULT)
            st.sidebar.info(f"Usando {DATA_PATH_DEFAULT}")
        else:
            st.sidebar.error(f"No encontré {DATA_PATH_DEFAULT}. Sube el CSV.")
            st.stop()
except Exception as e:
    st.sidebar.error(f"Error cargando dataset: {e}")
    st.stop()

# -------- Market intelligence
st.sidebar.subheader("📈 Market Intelligence")
market_file = st.sidebar.file_uploader(
    "Sube market_intel.csv",
    type=["csv"],
    key="uploader_market_csv"
)

market_df = None
try:
    if market_file is not None:
        market_df = load_market_intel(market_file)
        st.sidebar.success("Market Intel cargado ✅")
    else:
        if Path(MARKET_PATH_DEFAULT).exists():
            market_df = load_market_intel(MARKET_PATH_DEFAULT)
            st.sidebar.info(f"Usando {MARKET_PATH_DEFAULT}")
        else:
            st.sidebar.warning("Market Intel opcional: no cargado")
except Exception as e:
    st.sidebar.error(f"Error Market Intel: {e}")
    market_df = None

st.sidebar.divider()

# Entrenar modelos
try:
    success_model, sales_model, ACC, AUC, CM, MAE = train_models(df)
except Exception as e:
    st.error(f"Error entrenando modelos: {e}")
    st.stop()

# Header KPIs
st.title("🧠 Plataforma IA: Producto + Empaque + Claims + Market")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Registros", f"{len(df):,}")
k2.metric("Precisión", f"{ACC*100:.2f}%")
k3.metric("AUC", f"{AUC:.3f}")
k4.metric("Éxito base", f"{df['exito'].mean()*100:.1f}%")
k5.metric("MAE ventas", f"{MAE:,.0f} u.")

st.divider()

# ============================================================
# BLOQUE 3 — UI (Tabs) + Simulación (persistente)
# ============================================================

tab_sim, tab_ins, tab_market, tab_data, tab_diag = st.tabs([
    "🧪 Simulador",
    "📊 Insights",
    "📈 Market Intelligence",
    "📂 Datos",
    "🧠 Diagnóstico"
])

# ============================================================
# 🧪 SIMULADOR (no se borra con ROI)
# ============================================================
with tab_sim:
    st.subheader("🧪 Simulador What-If (éxito + ventas + pack + claims + conexión)")
    st.caption("Da click en **Simular** y luego ajusta ROI sin perder el resultado.")

    if "last_sim" not in st.session_state:
        st.session_state.last_sim = None

    marcas = sorted(df["marca"].unique().tolist())
    canales = sorted(df["canal"].unique().tolist())

    c1, c2, c3 = st.columns(3)
    marca = c1.selectbox("Marca", marcas, 0, key="sim_marca")
    canal = c2.selectbox("Canal", canales, 0, key="sim_canal")
    segmento = c3.selectbox("Segmento objetivo", ["fit", "kids", "premium", "value"], 0, key="sim_segmento")

    canal_norm = str(canal).lower().strip()

    st.markdown("### Variables de negocio")
    b1, b2, b3, b4, b5 = st.columns(5)
    precio = b1.number_input("Precio", 1.0, 99999.0, float(df["precio"].median()), step=1.0, key="sim_precio")
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
    recs = recommend_claims(segmento, canal_norm, max_claims=6)
    claim_options = [c for c, _ in recs]

    selected_claims_raw = st.multiselect(
        "Selecciona claims (ideal 2-3)",
        claim_options,
        default=claim_options[:2],
        key="sim_claims",
    )

    # ✅ Normalizar selected_claims (evita error en claims_score)
    selected_claims = []
    for x in (selected_claims_raw or []):
        if isinstance(x, (tuple, list)) and len(x) > 0:
            selected_claims.append(str(x[0]).strip())
        else:
            selected_claims.append(str(x).strip())

    cscore = claims_score(selected_claims, canal_norm)

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

    s1, s2, s3 = st.columns(3)
    s1.metric("Claims Score", f"{cscore:.1f}/100")
    s2.metric("Emotion Pack Score", f"{pack_emotion:.1f}/100")
    s3.metric("Conexión final", f"{conexion_score:.1f}/100")

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
        "canal": canal_norm,
    }])

    if st.button("🚀 Simular", key="sim_btn"):
        prob = float(success_model.predict_proba(entrada)[0][1])
        ventas_u = float(sales_model.predict(entrada)[0])
        ventas_u = max(0.0, ventas_u)

        precio_u = float(precio)
        margen_pct_u = float(margen_pct)
        ingresos = ventas_u * precio_u
        utilidad_bruta = ventas_u * (precio_u * (margen_pct_u / 100.0))
        margen_unitario = precio_u * (margen_pct_u / 100.0)

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

    st.divider()

    if st.session_state.last_sim is None:
        st.info("Da click en **Simular** para ver resultados. Luego puedes ajustar ROI sin que se borre.")
    else:
        sim = st.session_state.last_sim

        st.markdown("## 🎯 Resultado simulación")
        r1, r2, r3 = st.columns(3)
        r1.metric("Prob. Éxito", f"{sim['prob']*100:.2f}%")
        r2.metric("Ventas predichas", f"{sim['ventas_u']:,.0f} u.")
        r3.metric("Precio", f"${sim['precio']:,.0f}")

        st.markdown("### 💰 Unit economics")
        u1, u2, u3, u4 = st.columns(4)
        u1.metric("Ingresos ($)", f"${sim['ingresos']:,.0f}")
        u2.metric("Utilidad bruta ($)", f"${sim['utilidad_bruta']:,.0f}")
        u3.metric("Margen unitario ($/u)", f"${sim['margen_unitario']:.2f}")
        u4.metric("Margen %", f"{sim['margen_pct']:.1f}%")

        st.markdown("### 🎯 ROI (Financiero + Unidades)")
        rr1, rr2, rr3, rr4 = st.columns(4)

        inversion = rr1.number_input("Inversión ($) (opcional)", 0.0, 1e12, 0.0, step=1000.0, key="roi_inv")
        meta_u = rr2.number_input("Meta unidades (opcional)", 0.0, 1e12, 0.0, step=100.0, key="roi_meta_u")
        base_u = rr3.number_input(
            "Baseline unidades (opcional)",
            0.0, 1e12,
            float(np.median(df["ventas_unidades"])) if "ventas_unidades" in df.columns else 0.0,
            step=100.0,
            key="roi_base_u"
        )

        if meta_u > 0:
            rr4.metric("Cumplimiento vs meta", f"{(sim['ventas_u']/meta_u)*100:.1f}%")
        else:
            rr4.metric("Cumplimiento vs meta", "—")

        if base_u > 0:
            uplift = (sim["ventas_u"] - base_u) / base_u
            st.metric("Uplift vs baseline", f"{uplift*100:.1f}%")
        else:
            st.metric("Uplift vs baseline", "—")

        if inversion > 0:
            roi_fin = (sim["utilidad_bruta"] - inversion) / inversion
            payback_u = inversion / max(sim["margen_unitario"], 1e-6)
            st.metric("ROI financiero", f"{roi_fin*100:.1f}%")
            st.caption(f"Payback ≈ {payback_u:,.0f} u.")
        else:
            st.metric("ROI financiero", "—")

        st.markdown("### 📌 Inputs usados")
        st.dataframe(sim["entrada"], use_container_width=True)


# ============================================================
# 📊 INSIGHTS (con charts estables)
# ============================================================
with tab_ins:
    st.subheader("📊 Insights")
    left, right = st.columns(2)

    with left:
        st.markdown("**Ranking por marca (Conexión promedio)**")
        ins_marca = df.groupby("marca")[["conexion_score"]].mean().sort_values("conexion_score", ascending=False).round(2)
        st.dataframe(ins_marca, use_container_width=True)

        st.markdown("**Ranking por marca (Éxito %)**")
        ex_marca = df.groupby("marca")[["exito"]].mean().sort_values("exito", ascending=False).round(3)
        ex_marca["exito_%"] = (ex_marca["exito"] * 100).round(1)
        st.dataframe(ex_marca[["exito_%"]], use_container_width=True)

    with right:
        st.markdown("**Distribución Conexión (bucket)**")
        bins = pd.cut(df["conexion_score"], bins=[0, 20, 40, 60, 80, 100], include_lowest=True)
        vc = bins.value_counts().sort_index()
        chart_df = bar_df_from_value_counts(vc)
        st.bar_chart(chart_df.set_index("bucket"), use_container_width=True)

        st.markdown("**Distribución Ventas (bucket)**")
        b2 = pd.cut(df["ventas_unidades"].clip(0, 40000), bins=[0, 2000, 5000, 10000, 20000, 40000], include_lowest=True)
        vc2 = b2.value_counts().sort_index()
        chart_df2 = bar_df_from_value_counts(vc2)
        st.bar_chart(chart_df2.set_index("bucket"), use_container_width=True)


# ============================================================
# 📈 MARKET INTELLIGENCE
# ============================================================
with tab_market:
    st.subheader("📈 Market Intelligence")
    if market_df is None:
        st.info("Sube market_intel.csv (sidebar) para ver benchmarks y oportunidades.")
    else:
        st.markdown("### Benchmark de precio por marca/canal")
        bench = (
            market_df.groupby(["marca", "canal"])["precio"]
            .agg(p25=lambda x: np.percentile(x, 25),
                 p50=lambda x: np.percentile(x, 50),
                 p75=lambda x: np.percentile(x, 75))
            .round(1)
            .reset_index()
        )
        st.dataframe(bench, use_container_width=True)

        st.markdown("### Oportunidades (white spaces)")
        m = market_df.copy()
        max_comp = max(m["competencia_skus"].max(), 1e-6)
        m["opportunity_score"] = (
            0.4 * (m["demanda_idx"] / 100.0) +
            0.3 * (m["tendencia_idx"] / 100.0) +
            0.3 * (1.0 - (m["competencia_skus"] / max_comp))
        ) * 100.0

        opp = (
            m.groupby(["categoria", "canal"])["opportunity_score"]
            .mean()
            .sort_values(ascending=False)
            .round(1)
            .to_frame()
            .reset_index()
        )
        st.dataframe(opp, use_container_width=True)

        st.download_button(
            "📥 Descargar oportunidades (CSV)",
            data=df_to_csv_bytes(opp),
            file_name="oportunidades_market_intel.csv",
            mime="text/csv"
        )


# ============================================================
# 📂 DATOS
# ============================================================
with tab_data:
    st.subheader("📂 Dataset")
    st.download_button(
        "📥 Descargar dataset (CSV)",
        data=df_to_csv_bytes(df),
        file_name="dataset_con_ventas.csv",
        mime="text/csv"
    )
    st.dataframe(df.head(300), use_container_width=True)


# ============================================================
# 🧠 DIAGNÓSTICO
# ============================================================
with tab_diag:
    st.subheader("🧠 Diagnóstico de modelo")
    st.write("Matriz de confusión (Éxito):")
    st.dataframe(pd.DataFrame(CM, index=["Real 0", "Real 1"], columns=["Pred 0", "Pred 1"]), use_container_width=True)
    st.write(f"MAE ventas: **{MAE:,.0f}** unidades.")

