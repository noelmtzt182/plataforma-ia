import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from datetime import datetime
import io
import re

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="Plataforma IA | Producto + Empaque + Claims", layout="wide")
DATA_PATH_DEFAULT = "mercado_cereales_5000_con_ventas.csv"

REQUIRED_BASE = {
    "marca", "categoria", "canal", "precio", "costo", "margen", "margen_pct",
    "competencia", "demanda", "tendencia", "estacionalidad",
    "rating_conexion", "comentario", "sentiment_score",
    "conexion_score", "conexion_alta", "score_latente", "exito"
}
REQUIRED_SALES = {"ventas_unidades", "ventas_ingresos", "utilidad"}


# ============================================================
# HELPERS
# ============================================================
def _clean_str_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()

def clip(v, a, b):
    return float(max(a, min(b, v)))

def safe_percent(x):
    return f"{x*100:.2f}%"

def bar_plot_from_value_counts(vc: pd.Series, title: str):
    """Chart estable sin Altair (Matplotlib)."""
    dfp = vc.reset_index()
    if dfp.shape[1] >= 2:
        dfp = dfp.iloc[:, :2].copy()
        dfp.columns = ["bucket", "count"]
    else:
        dfp = pd.DataFrame({"bucket": ["(sin buckets)"], "count": [0]})
    dfp["bucket"] = dfp["bucket"].astype(str)
    dfp["count"] = pd.to_numeric(dfp["count"], errors="coerce").fillna(0)

    st.markdown(f"**{title}**")
    fig, ax = plt.subplots()
    ax.bar(dfp["bucket"], dfp["count"])
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", labelrotation=30)
    st.pyplot(fig, use_container_width=True)


# ============================================================
# PACK VISION (sin OCR)
# ============================================================
def image_metrics(img: Image.Image) -> dict:
    im = img.convert("RGB")
    arr = np.asarray(im).astype(np.float32)

    gray = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2])
    brightness = float(np.mean(gray) / 255.0)
    contrast = float(np.std(gray) / 255.0)

    rg = arr[..., 0] - arr[..., 1]
    yb = 0.5 * (arr[..., 0] + arr[..., 1]) - arr[..., 2]
    colorfulness = float((np.std(rg) + 0.3 * np.std(yb)) / 255.0)

    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    mag = np.sqrt(gx ** 2 + gy ** 2)
    thresh = np.percentile(mag, 85)
    edges = (mag > thresh).astype(np.float32)
    edge_density = float(np.mean(edges))

    pop_score = clip(0.55 * contrast + 0.45 * colorfulness, 0, 1)

    return {
        "brightness": brightness,
        "contrast": contrast,
        "colorfulness": colorfulness,
        "edge_density": edge_density,
        "pop_score": pop_score,
    }

def pack_scores_from_metrics(m: dict) -> dict:
    legibility = 70 * m["contrast"] + 30 * (1 - abs(m["edge_density"] - 0.18) / 0.18)
    legibility = clip(legibility, 0, 1) * 100

    target_brightness = 0.55
    brightness_fit = 1 - abs(m["brightness"] - target_brightness) / target_brightness
    shelf_pop = clip(0.75 * m["pop_score"] + 0.25 * clip(brightness_fit, 0, 1), 0, 1) * 100

    clarity = clip(0.6 * m["contrast"] + 0.4 * (1 - clip(m["edge_density"] / 0.35, 0, 1)), 0, 1) * 100

    # extras (Vision+)
    brand_contrast = clip(0.70 * m["contrast"] + 0.30 * (1 - abs(m["edge_density"] - 0.16) / 0.16), 0, 1) * 100
    text_overload = clip((m["edge_density"] / 0.35), 0, 1) * 100

    return {
        "pack_legibility_score": round(legibility, 1),
        "pack_shelf_pop_score": round(shelf_pop, 1),
        "pack_clarity_score": round(clarity, 1),
        "pack_brand_contrast_score": round(brand_contrast, 1),
        "pack_text_overload_score": round(text_overload, 1),
    }


# ============================================================
# CLAIMS ENGINE (BASE + DATA-DRIVEN)
# ============================================================

# Biblioteca base (curada, comercial; no legal)
CLAIMS_LIBRARY = {
    "fit": [
        ("alto en proteína", 0.92),
        ("sin azúcar añadida", 0.90),
        ("alto en fibra", 0.88),
        ("integral", 0.82),
        ("con granos enteros", 0.80),
        ("bajo en calorías", 0.76),
        ("sin colorantes artificiales", 0.74),
        ("sin conservadores", 0.72),
        ("clean label", 0.70),
        ("alto en hierro", 0.62),
    ],
    "kids": [
        ("con vitaminas y minerales", 0.88),
        ("sabor chocolate", 0.82),
        ("con granos", 0.76),
        ("sin conservadores", 0.74),
        ("sin colorantes artificiales", 0.72),
        ("fuente de energía", 0.70),
        ("con calcio", 0.68),
        ("con fibra", 0.66),
    ],
    "premium": [
        ("ingredientes seleccionados", 0.84),
        ("sabor intenso", 0.78),
        ("hecho con avena real", 0.78),
        ("calidad premium", 0.72),
        ("sin sabores artificiales", 0.70),
        ("origen/ingrediente real", 0.68),
        ("receta artesanal", 0.66),
    ],
    "value": [
        ("rinde más", 0.80),
        ("ideal para la familia", 0.76),
        ("gran sabor a mejor precio", 0.74),
        ("económico y práctico", 0.70),
        ("paquete ahorro", 0.68),
        ("más por menos", 0.66),
    ],
}

CANAL_CLAIM_BOOST = {
    "retail": {
        "sin azúcar añadida": 1.04,
        "alto en fibra": 1.03,
        "integral": 1.02,
        "paquete ahorro": 1.03,
        "ideal para la familia": 1.03,
        "sin conservadores": 1.02,
        "sin colorantes artificiales": 1.02,
    },
    "marketplace": {
        "alto en proteína": 1.05,
        "clean label": 1.04,
        "ingredientes seleccionados": 1.04,
        "alto en fibra": 1.04,
        "sin colorantes artificiales": 1.04,
    },
}

STOP_ES = set("""
de la que el en y a los del se las por un para con no una su al lo como más pero sus le ya o este sí porque esta entre cuando muy sin sobre también me hasta hay donde quien desde todo nos durante todos uno les ni contra otros ese eso ante ellos e esto mí antes algunos qué unos yo otro otras otra él tanto esa estos mucho quienes nada muchos cual poco ella estar estas algunas algo nosotros mi mis tú te ti tu tus ellas nosotras vosotros vosotras os mío mía míos mías tuyo tuya tuyos tuyas suyo suya suyos suyas nuestro nuestra nuestros nuestras vuestro vuestra vuestros vuestras esos esas estoy estás está estamos están
""".split())

def normalize_text(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"[^a-záéíóúñü0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

@st.cache_data
def build_dynamic_claims(text_series: pd.Series, top_k: int = 60) -> list[tuple[str, float]]:
    """
    Extrae 'claims' (keywords/bigramas) de comentarios reales.
    Si no hay labels (ventas/exito), rankea por relevancia (TF-IDF mean).
    Retorna lista [(term, score_0_1), ...]
    """
    s = text_series.fillna("").astype(str).apply(normalize_text)
    if (s.str.len() < 5).mean() > 0.85:
        return []

    vec = TfidfVectorizer(
        stop_words=list(STOP_ES),
        ngram_range=(1, 2),
        min_df=5,
        max_features=3500
    )
    X = vec.fit_transform(s)
    terms = np.array(vec.get_feature_names_out())
    tfidf_mean = np.asarray(X.mean(axis=0)).ravel()

    # Normaliza a 0..1
    score = (tfidf_mean - tfidf_mean.min()) / (tfidf_mean.max() - tfidf_mean.min() + 1e-9)

    good = []
    for t, sc in zip(terms, score):
        if len(t) < 4:
            continue
        if t in STOP_ES:
            continue
        if any(x in t for x in ["http", "www", ".com"]):
            continue
        good.append((t, float(sc)))

    good.sort(key=lambda x: x[1], reverse=True)
    return good[:top_k]

def recommend_claims(segment: str, canal: str, max_claims: int = 10, text_corpus: pd.Series | None = None):
    """
    Mezcla:
      - Biblioteca base (curada por segmento)
      - Biblioteca dinámica (aprendida de comentarios reales)
    Regresa lista: (claim, score, source)
    """
    seg = str(segment).lower().strip()
    canal = str(canal).lower().strip()

    scored = []

    # Base
    for claim, base in CLAIMS_LIBRARY.get(seg, [])[:]:
        boost = CANAL_CLAIM_BOOST.get(canal, {}).get(claim, 1.0)
        scored.append((claim, base * boost, "base"))

    # Data-driven (comentarios reales)
    if text_corpus is not None and len(text_corpus) > 0:
        dyn = build_dynamic_claims(text_corpus, top_k=50)
        for term, s in dyn:
            # map 0..1 -> 0.55..0.85 para convivir con base
            base = 0.55 + 0.30 * float(s)
            boost = CANAL_CLAIM_BOOST.get(canal, {}).get(term, 1.0)
            scored.append((term, base * boost, "data"))

    # Dedup (conserva el mayor score)
    best = {}
    for claim, sc, src in scored:
        k = str(claim).lower().strip()
        if k not in best or sc > best[k][0]:
            best[k] = (float(sc), src)

    merged = [(k, v[0], v[1]) for k, v in best.items()]
    merged.sort(key=lambda x: x[1], reverse=True)
    return merged[:max_claims]

def claims_score(selected_claims, canal: str) -> float:
    if not selected_claims:
        return 0.0
    canal = str(canal).lower().strip()
    boosts = [CANAL_CLAIM_BOOST.get(canal, {}).get(str(c).lower().strip(), 1.0) for c in selected_claims]
    base = float(np.mean(boosts))
    n = len(selected_claims)
    clarity_penalty = 1.0 if n <= 3 else max(0.65, 1.0 - 0.12 * (n - 3))
    score = 75 * base * clarity_penalty
    return float(np.clip(score, 0, 100))

def pack_emotion_score(pack_legibility, pack_pop, pack_clarity, claims_score_val, copy_tone: int):
    visual = 0.40 * (pack_pop / 100) + 0.30 * (pack_clarity / 100) + 0.15 * (pack_legibility / 100)
    claims = 0.15 * (claims_score_val / 100)
    tone_boost = 0.06 if copy_tone > 0 else (-0.06 if copy_tone < 0 else 0.0)
    score = (visual + claims + tone_boost) * 100
    return float(np.clip(score, 0, 100))


# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data
def load_data(path_or_file) -> pd.DataFrame:
    df = pd.read_csv(path_or_file).copy()

    for c in ["marca", "categoria", "canal", "estacionalidad", "comentario"]:
        if c in df.columns:
            df[c] = _clean_str_series(df[c])

    missing = sorted(list(REQUIRED_BASE - set(df.columns)))
    if missing:
        raise ValueError(f"Faltan columnas base en el CSV: {missing}")

    num_cols = [
        "precio", "costo", "margen", "margen_pct", "competencia", "demanda", "tendencia",
        "rating_conexion", "sentiment_score", "conexion_score", "conexion_alta",
        "score_latente", "exito"
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ["ventas_unidades", "ventas_ingresos", "utilidad"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=[
        "marca", "canal", "precio", "competencia", "demanda", "tendencia",
        "margen_pct", "conexion_score", "rating_conexion", "sentiment_score", "exito"
    ]).copy()

    df["exito"] = df["exito"].astype(int)

    # ventas opcional
    for c in REQUIRED_SALES:
        if c not in df.columns:
            df[c] = np.nan

    return df

@st.cache_data
def load_reviews(path_or_file) -> pd.DataFrame:
    """
    Carga comentarios reales (reviews) de consumidores.
    Formato recomendado: texto, rating (opcional), marca/canal/categoria (opcionales), fecha (opcional), fuente (opcional)
    """
    rv = pd.read_csv(path_or_file).copy()
    # normaliza nombres típicos
    rename_map = {}
    for col in rv.columns:
        c = col.strip().lower()
        if c in ["comentario", "review", "reseña", "resena", "texto_review"]:
            rename_map[col] = "texto"
        if c in ["rating", "estrellas", "stars", "calificacion", "calificación"]:
            rename_map[col] = "rating"
    if rename_map:
        rv = rv.rename(columns=rename_map)

    if "texto" not in rv.columns:
        raise ValueError("El reviews CSV debe incluir columna 'texto' (comentario del consumidor).")

    rv["texto"] = rv["texto"].astype(str)
    for c in ["marca", "canal", "categoria", "fuente"]:
        if c in rv.columns:
            rv[c] = _clean_str_series(rv[c])

    return rv


# ============================================================
# MODELS
# ============================================================
@st.cache_resource
def train_success_model(df: pd.DataFrame):
    features = [
        "precio", "competencia", "demanda", "tendencia", "margen_pct",
        "conexion_score", "rating_conexion", "sentiment_score",
        "marca", "canal"
    ]
    X = df[features]
    y = df["exito"].astype(int)

    num_cols = [
        "precio", "competencia", "demanda", "tendencia", "margen_pct",
        "conexion_score", "rating_conexion", "sentiment_score"
    ]
    cat_cols = ["marca", "canal"]

    pre = ColumnTransformer(transformers=[
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])

    clf = Pipeline(steps=[
        ("preprocessor", pre),
        ("model", RandomForestClassifier(
            n_estimators=350,
            random_state=42,
            class_weight="balanced_subsample"
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    proba = clf.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, proba)
    cm = confusion_matrix(y_test, pred)
    return clf, acc, auc, cm

@st.cache_resource
def train_sales_model(df: pd.DataFrame):
    df2 = df.dropna(subset=["ventas_unidades"]).copy()
    if len(df2) < 50:
        raise ValueError("Muy pocos registros con ventas_unidades para entrenar ventas.")

    features = [
        "precio", "competencia", "demanda", "tendencia", "margen_pct",
        "conexion_score", "rating_conexion", "sentiment_score",
        "marca", "canal"
    ]
    X = df2[features]
    y = df2["ventas_unidades"].astype(float)

    num_cols = [
        "precio", "competencia", "demanda", "tendencia", "margen_pct",
        "conexion_score", "rating_conexion", "sentiment_score"
    ]
    cat_cols = ["marca", "canal"]

    pre = ColumnTransformer(transformers=[
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])

    reg = Pipeline(steps=[
        ("preprocessor", pre),
        ("model", RandomForestRegressor(n_estimators=350, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    reg.fit(X_train, y_train)

    pred = reg.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    return reg, mae


# ============================================================
# SIDEBAR: DATA + REVIEWS
# ============================================================
st.sidebar.title("⚙️ Datos")
uploaded = st.sidebar.file_uploader("Sube tu CSV principal (con ventas)", type=["csv"], key="uploader_csv_main")

if uploaded is not None:
    df = load_data(uploaded)
else:
    if Path(DATA_PATH_DEFAULT).exists():
        df = load_data(DATA_PATH_DEFAULT)
    else:
        st.warning(f"❗No encontré '{DATA_PATH_DEFAULT}' en el repo. Sube el CSV para arrancar.")
        st.stop()

st.sidebar.divider()
st.sidebar.subheader("🗣️ Comentarios reales (opcional)")
st.sidebar.caption("Sube un CSV con columna 'texto'. Recomendado: texto,rating,marca,canal,categoria,fecha,fuente.")
uploaded_reviews = st.sidebar.file_uploader("Sube reviews_consumidores.csv", type=["csv"], key="uploader_reviews")

reviews_df = None
if uploaded_reviews is not None:
    try:
        reviews_df = load_reviews(uploaded_reviews)
        st.sidebar.success(f"Reviews cargadas: {len(reviews_df):,}")
    except Exception as e:
        st.sidebar.error(f"Error cargando reviews: {e}")
        reviews_df = None

# Corpus de texto para claims data-driven:
# Si hay reviews reales, usamos eso primero. Si no, usamos df['comentario'].
if reviews_df is not None:
    text_corpus = reviews_df["texto"].astype(str)
else:
    text_corpus = df["comentario"].astype(str) if "comentario" in df.columns else pd.Series([], dtype=str)

# Train models
success_model, acc, auc, cm = train_success_model(df)

sales_model = None
mae = None
try:
    if df["ventas_unidades"].notna().any():
        sales_model, mae = train_sales_model(df)
except Exception as e:
    st.sidebar.warning(f"Modelo de ventas desactivado: {e}")


# ============================================================
# HEADER
# ============================================================
st.title("🧠 Plataforma IA: Producto + Empaque + Claims (v1 + claims reales)")
st.caption("Éxito + Ventas estimadas + Insights + Pack Lab + Claims Lab + Experimentos + Inversionista + Recomendador + Reporte")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Registros", f"{len(df):,}")
k2.metric("Precisión", f"{acc*100:.2f}%")
k3.metric("AUC", f"{auc:.3f}")
k4.metric("Éxito base", f"{df['exito'].mean()*100:.1f}%")
k5.metric("MAE ventas", f"{mae:,.0f} u." if mae is not None else "N/A")

st.divider()

# ============================================================
# TABS
# ============================================================
tab_sim, tab_ins, tab_pack, tab_claims, tab_exp, tab_inv, tab_rec, tab_report, tab_data, tab_model = st.tabs(
    [
        "🧪 Simulador",
        "📊 Insights",
        "📦 Pack Lab",
        "🏷️ Claims Lab",
        "🧪 Experimentos",
        "💼 Inversionista",
        "🧠 Recomendador",
        "📄 Reporte",
        "📂 Datos",
        "🧠 Modelo",
    ]
)

# ============================================================
# STATE: Reporte
# ============================================================
if "last_run" not in st.session_state:
    st.session_state.last_run = {}
if "last_pack" not in st.session_state:
    st.session_state.last_pack = {}
if "last_rec" not in st.session_state:
    st.session_state.last_rec = {}

# ============================================================
# 🧪 SIMULADOR
# ============================================================
with tab_sim:
    st.subheader("🧪 Simulador What-If (incluye empaque + claims)")

    marcas = sorted(df["marca"].unique().tolist())
    canales = sorted(df["canal"].unique().tolist())

    c1, c2, c3 = st.columns(3)
    marca = c1.selectbox("Marca", marcas, 0, key="sim_marca")
    canal = c2.selectbox("Canal", canales, 0, key="sim_canal")
    segmento = c3.selectbox("Segmento objetivo", ["fit", "kids", "premium", "value"], 0, key="sim_segmento")

    st.markdown("### Variables de negocio")
    b1, b2, b3, b4, b5 = st.columns(5)
    precio = b1.number_input("Precio", value=float(df["precio"].median()), step=1.0, key="sim_precio")
    competencia = b2.slider("Competencia (1-10)", 1, 10, int(df["competencia"].median()), key="sim_competencia")
    demanda = b3.slider("Demanda (10-100)", 10, 100, int(df["demanda"].median()), key="sim_demanda")
    tendencia = b4.slider("Tendencia (20-100)", 20, 100, int(df["tendencia"].median()), key="sim_tendencia")
    margen_pct = b5.slider("Margen %", 0, 90, int(np.clip(df["margen_pct"].median(), 0, 90)), key="sim_margen_pct")

    st.markdown("### Empaque + Claims")
    p1, p2, p3 = st.columns(3)
    pack_legibility_score = p1.slider("Pack legibilidad (0-100)", 0, 100, 65, key="sim_pack_legibility")
    pack_shelf_pop_score = p2.slider("Pack shelf pop (0-100)", 0, 100, 70, key="sim_pack_pop")
    pack_clarity_score = p3.slider("Pack claridad (0-100)", 0, 100, 65, key="sim_pack_clarity")

    recs = recommend_claims(segmento, canal, max_claims=12, text_corpus=text_corpus)
    claim_options = [c for c, _, _ in recs]

    selected_claims = st.multiselect(
        "Selecciona claims (ideal 2-3)",
        claim_options,
        default=claim_options[:2],
        key="sim_claims"
    )

    copy = st.text_input("Copy corto (opcional)", value="energía y nutrición para tu día", key="sim_copy")
    pos_kw = ["energía", "nutrición", "saludable", "delicioso", "me encanta", "premium", "calidad", "proteína", "fibra"]
    neg_kw = ["caro", "no", "malo", "rechazo", "no me gusta", "pésimo", "horrible"]
    t = copy.lower()
    tone = 0
    if any(k in t for k in pos_kw): tone += 1
    if any(k in t for k in neg_kw): tone -= 1
    copy_tone = 1 if tone > 0 else (-1 if tone < 0 else 0)

    cscore = claims_score(selected_claims, canal)
    pack_emotion = pack_emotion_score(pack_legibility_score, pack_shelf_pop_score, pack_clarity_score, cscore, copy_tone)
    uplift = clip((pack_emotion - 50) / 50, -0.35, 0.35)

    rating_conexion = st.slider("Rating conexión producto (1-10)", 1, 10, 7, key="sim_rating")
    sentiment_score = st.select_slider("Sentimiento del producto (-1/0/1)", options=[-1, 0, 1], value=1, key="sim_sentiment")

    base_conexion = (rating_conexion / 10) * 70 + sentiment_score * 15 + 5
    conexion_score = clip(base_conexion * (1 + uplift), 0, 100)

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Claims Score", f"{cscore:.1f}/100")
    s2.metric("Pack Emotion Score", f"{pack_emotion:.1f}/100")
    s3.metric("Uplift conexión (±35%)", f"{uplift*100:+.1f}%")
    s4.metric("Conexión final", f"{conexion_score:.1f}/100")

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
        "canal": str(canal).lower()
    }])

    if st.button("🚀 Simular", key="sim_btn"):
        p = float(success_model.predict_proba(entrada)[0][1])
        pred = int(success_model.predict(entrada)[0])

        ventas = None
        ingresos = None
        utilidad = None

        if sales_model is not None:
            ventas = max(0, round(float(sales_model.predict(entrada)[0])))
            ingresos = ventas * float(precio)
            utilidad = ventas * (float(precio) * (float(margen_pct) / 100.0))

        st.session_state.last_run = {
            "timestamp": datetime.utcnow().isoformat(),
            "marca": marca,
            "canal": canal,
            "segmento": segmento,
            "precio": float(precio),
            "competencia": float(competencia),
            "demanda": float(demanda),
            "tendencia": float(tendencia),
            "margen_pct": float(margen_pct),
            "claims": selected_claims,
            "claims_score": float(cscore),
            "pack_legibility": float(pack_legibility_score),
            "pack_pop": float(pack_shelf_pop_score),
            "pack_clarity": float(pack_clarity_score),
            "pack_emotion": float(pack_emotion),
            "conexion_score": float(conexion_score),
            "prob_exito": float(p),
            "pred_exito": int(pred),
            "ventas_unidades": None if ventas is None else int(ventas),
            "ingresos": None if ingresos is None else float(ingresos),
            "utilidad": None if utilidad is None else float(utilidad),
        }

        r1, r2, r3 = st.columns(3)
        r1.metric("Prob. éxito", safe_percent(p))
        r2.metric("Predicción", "✅ Éxito" if pred else "⚠️ Riesgo")

        if ventas is not None:
            r3.metric("Ventas (unidades)", f"{ventas:,.0f}")
            rr1, rr2 = st.columns(2)
            rr1.metric("Ingresos ($)", f"${ingresos:,.0f}")
            rr2.metric("Utilidad ($)", f"${utilidad:,.0f}")
        else:
            r3.info("Modelo de ventas no disponible")

        st.dataframe(entrada, use_container_width=True)

# ============================================================
# 📊 INSIGHTS
# ============================================================
with tab_ins:
    st.subheader("📊 Insights")

    left, right = st.columns(2)

    with left:
        st.markdown("**Ranking por marca (Conexión promedio)**")
        ins_marca = df.groupby("marca")[["conexion_score"]].mean().sort_values("conexion_score", ascending=False).round(2)
        st.dataframe(ins_marca.reset_index(), use_container_width=True)

        st.markdown("**Ranking por marca (Éxito %)**")
        ex_marca = df.groupby("marca")[["exito"]].mean().sort_values("exito", ascending=False).round(3)
        ex_marca["exito_%"] = (ex_marca["exito"] * 100).round(1)
        st.dataframe(ex_marca[["exito_%"]].reset_index(), use_container_width=True)

        st.markdown("**Ranking por marca (Ventas promedio)**")
        if df["ventas_unidades"].notna().any():
            v_marca = df.groupby("marca")[["ventas_unidades"]].mean().sort_values("ventas_unidades", ascending=False).round(0)
            st.dataframe(v_marca.reset_index(), use_container_width=True)
        else:
            st.info("No hay ventas_unidades en el dataset.")

    with right:
        st.markdown("**Marca + Canal (Conexión promedio)**")
        ins_mc = df.groupby(["marca", "canal"])[["conexion_score"]].mean().sort_values("conexion_score", ascending=False).round(2)
        st.dataframe(ins_mc.head(25).reset_index(), use_container_width=True)

        st.markdown("**Marca + Canal (Éxito %)**")
        ex_mc = df.groupby(["marca", "canal"])[["exito"]].mean().sort_values("exito", ascending=False).round(3)
        ex_mc["exito_%"] = (ex_mc["exito"] * 100).round(1)
        st.dataframe(ex_mc.head(25)[["exito_%"]].reset_index(), use_container_width=True)

        st.markdown("**Marca + Canal (Ventas promedio)**")
        if df["ventas_unidades"].notna().any():
            v_mc = df.groupby(["marca", "canal"])[["ventas_unidades"]].mean().sort_values("ventas_unidades", ascending=False).round(0)
            st.dataframe(v_mc.head(25).reset_index(), use_container_width=True)
        else:
            st.info("No hay ventas_unidades en el dataset.")

    st.divider()
    d1, d2 = st.columns(2)
    with d1:
        bins = pd.cut(df["conexion_score"], bins=[0, 20, 40, 60, 80, 100], include_lowest=True)
        dist = bins.value_counts().sort_index()
        bar_plot_from_value_counts(dist, "Distribución: Conexión emocional (bucket)")

    with d2:
        if df["ventas_unidades"].notna().any():
            bins2 = pd.cut(df["ventas_unidades"].fillna(0).clip(0, 40000), bins=[0, 2000, 5000, 10000, 20000, 40000], include_lowest=True)
            dist2 = bins2.value_counts().sort_index()
            bar_plot_from_value_counts(dist2, "Distribución: Ventas unidades (bucket)")
        else:
            st.info("No hay ventas_unidades para graficar.")

# ============================================================
# 📦 PACK LAB
# ============================================================
with tab_pack:
    st.subheader("📦 Pack Lab / Pack Vision+ (sube tu empaque)")

    img_file = st.file_uploader("Sube imagen del empaque (PNG/JPG)", type=["png", "jpg", "jpeg"], key="pack_uploader")
    if img_file is None:
        st.info("Sube una imagen para generar análisis del empaque (Vision+).")
    else:
        img = Image.open(img_file)
        st.image(img, caption="Empaque cargado", use_container_width=True)

        m = image_metrics(img)
        scores = pack_scores_from_metrics(m)

        st.session_state.last_pack = {
            "brightness": m["brightness"],
            "contrast": m["contrast"],
            "colorfulness": m["colorfulness"],
            "edge_density": m["edge_density"],
            "pack_legibility_score": scores["pack_legibility_score"],
            "pack_shelf_pop_score": scores["pack_shelf_pop_score"],
            "pack_clarity_score": scores["pack_clarity_score"],
            "pack_brand_contrast_score": scores["pack_brand_contrast_score"],
            "pack_text_overload_score": scores["pack_text_overload_score"],
        }

        a1, a2, a3, a4, a5 = st.columns(5)
        a1.metric("Brillo", f"{m['brightness']:.2f}")
        a2.metric("Contraste", f"{m['contrast']:.2f}")
        a3.metric("Colorfulness", f"{m['colorfulness']:.2f}")
        a4.metric("Edge density", f"{m['edge_density']:.3f}")
        a5.metric("Brand contrast+", f"{scores['pack_brand_contrast_score']:.1f}/100")

        b1, b2, b3, b4, b5 = st.columns(5)
        b1.metric("Legibilidad", f"{scores['pack_legibility_score']}/100")
        b2.metric("Shelf Pop", f"{scores['pack_shelf_pop_score']}/100")
        b3.metric("Claridad", f"{scores['pack_clarity_score']}/100")
        b4.metric("Text overload", f"{scores['pack_text_overload_score']:.1f}/100")
        b5.metric("Visibilidad (proxy)", f"{(0.5*scores['pack_shelf_pop_score']+0.5*scores['pack_legibility_score']):.1f}/100")

        st.markdown("### Recomendaciones rápidas (Vision+)")
        recs_list = []
        if scores["pack_legibility_score"] < 60:
            recs_list.append("• Sube legibilidad: más contraste texto/fondo, tipografía más gruesa, menos elementos alrededor del claim principal.")
        if scores["pack_clarity_score"] < 60:
            recs_list.append("• Mejora claridad: reduce ruido visual, deja aire y limita a 2–3 claims.")
        if scores["pack_shelf_pop_score"] < 60:
            recs_list.append("• Sube shelf pop: usa color acento y evita pack muy oscuro o demasiado lavado.")
        if scores["pack_text_overload_score"] > 70:
            recs_list.append("• Exceso de texto/ruido: elimina microtextos y patrones, simplifica fondos.")
        if scores["pack_brand_contrast_score"] < 55:
            recs_list.append("• Marca poco visible: aumenta contraste del logo/nombre y evita colocarlo sobre zonas con ruido.")
        if not recs_list:
            recs_list.append("• Visualmente va bien. Ajusta jerarquía: Marca → beneficio → variedad → credencial.")
        st.write("\n".join(recs_list))

# ============================================================
# 🏷️ CLAIMS LAB
# ============================================================
with tab_claims:
    st.subheader("🏷️ Claims Lab (base + data-driven de consumidores)")

    c1, c2 = st.columns(2)
    segmento = c1.selectbox("Segmento", ["fit", "kids", "premium", "value"], 0, key="claims_segmento")
    canal_c = c2.selectbox("Canal", ["retail", "marketplace"], 0, key="claims_canal")

    recs = recommend_claims(segmento, canal_c, max_claims=20, text_corpus=text_corpus)
    rec_df = pd.DataFrame(recs, columns=["claim", "score", "source"])
    rec_df["score"] = (rec_df["score"] * 100).round(1)
    st.dataframe(rec_df, use_container_width=True)

    options = rec_df["claim"].tolist()
    selected = st.multiselect("Selecciona 2-3 claims", options, default=options[:2], key="claims_selected")
    cscore = claims_score(selected, canal_c)
    st.metric("Claims Score", f"{cscore:.1f}/100")

    st.warning("Nota: esto es recomendación comercial (no legal/regulatoria).")

    if reviews_df is not None:
        st.success("✅ Usando comentarios reales de consumidores para sugerir claims (source=data).")
    else:
        st.info("Tip: sube reviews_consumidores.csv para que el motor aprenda de consumidores reales.")

# ============================================================
# 🧪 EXPERIMENTOS
# ============================================================
with tab_exp:
    st.subheader("🧪 Experimentos (A/B)")

    if "experiments" not in st.session_state:
        st.session_state.experiments = []

    c1, c2, c3 = st.columns(3)
    exp_name = c1.text_input("Nombre experimento", value="test pack v1 vs v2", key="exp_name")
    variant = c2.selectbox("Variante", ["A", "B"], 0, key="exp_variant")
    metric = c3.selectbox("Métrica", ["intencion_compra", "conexion_pack", "ventas_piloto"], 0, key="exp_metric")

    v1, v2, v3 = st.columns(3)
    marca_e = v1.selectbox("Marca", sorted(df["marca"].unique().tolist()), 0, key="exp_marca")
    canal_e = v2.selectbox("Canal", sorted(df["canal"].unique().tolist()), 0, key="exp_canal")
    value = v3.number_input("Valor observado", value=7.0, step=0.1, key="exp_value")

    if st.button("➕ Guardar medición", key="exp_save"):
        st.session_state.experiments.append({
            "experimento": exp_name,
            "variante": variant,
            "marca": marca_e,
            "canal": canal_e,
            "metrica": metric,
            "valor": float(value),
        })
        st.success("Guardado.")

    if st.session_state.experiments:
        exp_df = pd.DataFrame(st.session_state.experiments)
        st.dataframe(exp_df, use_container_width=True)

        try:
            pivot = exp_df.pivot_table(index=["experimento", "metrica"], columns="variante", values="valor", aggfunc="mean")
            pivot["lift_B_vs_A"] = (pivot.get("B") - pivot.get("A"))
            st.dataframe(pivot.reset_index().round(3), use_container_width=True)
        except Exception:
            st.info("Necesitas registros en A y B para calcular lift.")
    else:
        st.info("Aún no hay experimentos guardados.")

# ============================================================
# 💼 INVERSIONISTA (resumen simple)
# ============================================================
with tab_inv:
    st.subheader("💼 Vista Inversionista (simple)")

    a1, a2, a3, a4 = st.columns(4)
    tam_mx = a1.number_input("TAM MX (mdd/año)", value=1800.0, step=50.0, key="inv_tam")
    som_pct = a2.slider("SOM capturable (%)", 1, 30, 6, key="inv_som") / 100.0
    asp = a3.number_input("ASP promedio ($/unidad)", value=float(df["precio"].median()), step=1.0, key="inv_asp")
    gross_margin = a4.slider("Margen bruto (%)", 10, 80, int(np.clip(df["margen_pct"].median(), 10, 80)), key="inv_gm") / 100.0

    som_value = tam_mx * som_pct
    base_sales = df["ventas_unidades"].dropna().mean() if df["ventas_unidades"].notna().any() else 8000.0

    k1, k2, k3 = st.columns(3)
    k1.metric("SOM estimado (mdd/año)", f"{som_value:,.1f}")
    k2.metric("Ventas base (u.)", f"{base_sales:,.0f}")
    k3.metric("GM (%)", f"{gross_margin*100:.1f}%")

    uplift = st.slider("Uplift esperado por IA (%)", 0, 60, 15, key="inv_upl") / 100.0
    sales_opt = base_sales * (1 + uplift)
    revenue_opt = sales_opt * asp
    profit_opt = revenue_opt * gross_margin

    st.metric("Upside utilidad bruta ($)", f"${profit_opt - (base_sales*asp*gross_margin):,.0f}")

# ============================================================
# 🧠 RECOMENDADOR (simple)
# ============================================================
with tab_rec:
    st.subheader("🧠 Recomendador (simple)")

    st.info("Este bloque lo puedes dejar simple o lo reemplazo por el motor what-if avanzado (si lo quieres).")

    segmento_r = st.selectbox("Segmento", ["fit", "kids", "premium", "value"], 0, key="rec_segmento")
    canal_r = st.selectbox("Canal", sorted(df["canal"].unique().tolist()), 0, key="rec_canal")
    recs = recommend_claims(segmento_r, canal_r, max_claims=15, text_corpus=text_corpus)
    st.dataframe(pd.DataFrame(recs, columns=["claim", "score", "source"]), use_container_width=True)

# ============================================================
# 📄 REPORTE (TXT + CSV)
# ============================================================
with tab_report:
    st.subheader("📄 Reporte Ejecutivo (descargable)")

    last_run = st.session_state.last_run
    last_pack = st.session_state.last_pack
    last_rec = st.session_state.last_rec

    lines = []
    lines.append("PLATAFORMA IA PRODUCTO - REPORTE EJECUTIVO")
    lines.append(f"Fecha (UTC): {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Reviews reales cargadas: {'SI' if reviews_df is not None else 'NO'}")
    lines.append("")

    if last_run:
        lines.append("== Última Simulación ==")
        for k, v in last_run.items():
            lines.append(f"{k}: {v}")
        lines.append("")

    if last_pack:
        lines.append("== Pack Vision+ ==")
        for k, v in last_pack.items():
            lines.append(f"{k}: {v}")
        lines.append("")

    report_txt = "\n".join(lines)

    st.download_button(
        "📥 Descargar reporte (TXT)",
        data=report_txt.encode("utf-8"),
        file_name="reporte_plataforma_ia_producto.txt",
        mime="text/plain",
        key="dl_report_txt",
    )

    st.text_area("Preview reporte", report_txt, height=280)

# ============================================================
# 📂 DATOS
# ============================================================
with tab_data:
    st.subheader("📂 Datos + Descarga")
    st.download_button(
        label="📥 Descargar dataset (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="dataset_con_ventas.csv",
        mime="text/csv",
        key="download_csv",
    )
    st.dataframe(df.head(300), use_container_width=True)

    st.divider()
    st.subheader("🗣️ Preview comentarios reales")
    if reviews_df is not None:
        st.dataframe(reviews_df.head(200), use_container_width=True)
    else:
        st.info("Sube reviews_consumidores.csv para activar esta sección.")

# ============================================================
# 🧠 MODELO
# ============================================================
with tab_model:
    st.subheader("🧠 Diagnóstico")
    st.dataframe(pd.DataFrame(cm, index=["Real 0", "Real 1"], columns=["Pred 0", "Pred 1"]), use_container_width=True)
    st.write(f"Accuracy: **{acc*100:.2f}%** | AUC: **{auc:.3f}**")
    st.caption("Nota: el modelo predice probabilidad de éxito. Ventas depende de tener ventas_unidades suficientes.")