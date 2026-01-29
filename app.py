import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, mean_absolute_error

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

def safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def bar_plot_from_value_counts(vc: pd.Series, title: str):
    """Chart estable (sin Altair) para vc=value_counts()."""
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
# PACK (IMAGEN) METRICS (ligero, sin OCR)
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

    return {
        "pack_legibility_score": round(legibility, 1),
        "pack_shelf_pop_score": round(shelf_pop, 1),
        "pack_clarity_score": round(clarity, 1),
    }


# ============================================================
# CLAIMS ENGINE
# ============================================================
CLAIMS_LIBRARY = {
    "fit": [
        ("alto en proteína", 0.90),
        ("sin azúcar añadida", 0.88),
        ("alto en fibra", 0.86),
        ("integral", 0.80),
        ("bajo en calorías", 0.78),
        ("sin colorantes artificiales", 0.72),
    ],
    "kids": [
        ("con vitaminas y minerales", 0.86),
        ("sabor chocolate", 0.82),
        ("energía para su día", 0.78),
        ("hecho con granos", 0.74),
        ("sin conservadores", 0.70),
    ],
    "premium": [
        ("ingredientes seleccionados", 0.82),
        ("sabor intenso", 0.78),
        ("hecho con avena real", 0.76),
        ("calidad premium", 0.70),
        ("receta artesanal", 0.64),
    ],
    "value": [
        ("rinde más", 0.78),
        ("gran sabor a mejor precio", 0.74),
        ("ideal para la familia", 0.72),
        ("económico y práctico", 0.66),
    ],
}

CANAL_CLAIM_BOOST = {
    "retail": {"sin azúcar añadida": 1.04, "alto en fibra": 1.03, "integral": 1.02},
    "marketplace": {"alto en proteína": 1.05, "sin colorantes artificiales": 1.04, "ingredientes seleccionados": 1.03},
}

def recommend_claims(segment: str, canal: str, max_claims: int = 6):
    seg = str(segment).lower().strip()
    canal = str(canal).lower().strip()
    items = CLAIMS_LIBRARY.get(seg, [])[:]
    scored = []
    for claim, base in items:
        boost = CANAL_CLAIM_BOOST.get(canal, {}).get(claim, 1.0)
        scored.append((claim, base * boost))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:max_claims]

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


# ============================================================
# CONEXIÓN EMOCIONAL DEL EMPAQUE (heurística)
# ============================================================
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

    # estandariza strings
    for c in ["marca", "categoria", "canal", "estacionalidad", "comentario"]:
        if c in df.columns:
            df[c] = _clean_str_series(df[c])

    # valida columnas
    missing = sorted(list(REQUIRED_BASE - set(df.columns)))
    if missing:
        raise ValueError(f"Faltan columnas base en el CSV: {missing}")

    # fuerza numéricas
    num_cols = [
        "precio", "costo", "margen", "margen_pct", "competencia", "demanda", "tendencia",
        "rating_conexion", "sentiment_score", "conexion_score", "conexion_alta",
        "score_latente", "exito"
    ]
    for c in num_cols:
        df[c] = safe_num(df[c])

    for c in ["ventas_unidades", "ventas_ingresos", "utilidad"]:
        if c in df.columns:
            df[c] = safe_num(df[c])

    # limpia NaNs críticos
    df = df.dropna(subset=[
        "marca", "canal", "precio", "competencia", "demanda", "tendencia",
        "margen_pct", "conexion_score", "rating_conexion", "sentiment_score", "exito"
    ]).copy()

    df["exito"] = df["exito"].astype(int)

    # si no existen ventas, crea placeholders (pero el modelo de ventas requerirá ventas reales)
    if not REQUIRED_SALES.issubset(df.columns):
        for c in ["ventas_unidades", "ventas_ingresos", "utilidad"]:
            if c not in df.columns:
                df[c] = np.nan

    return df


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

    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

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
    if not REQUIRED_SALES.issubset(df.columns):
        raise ValueError("Tu CSV no contiene columnas de ventas. Usa un CSV con ventas_* para entrenar ventas.")

    df2 = df.dropna(subset=["ventas_unidades"]).copy()
    if len(df2) < 50:
        raise ValueError("Muy pocos registros con ventas_unidades para entrenar el modelo de ventas (mínimo ~50).")

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

    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

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
# SIDEBAR DATA
# ============================================================
st.sidebar.title("⚙️ Datos")
uploaded = st.sidebar.file_uploader("Sube tu CSV (con ventas)", type=["csv"], key="uploader_csv")

if uploaded is not None:
    df = load_data(uploaded)
else:
    if Path(DATA_PATH_DEFAULT).exists():
        df = load_data(DATA_PATH_DEFAULT)
    else:
        st.warning(f"❗No encontré '{DATA_PATH_DEFAULT}' en el repo. Sube el CSV para arrancar.")
        st.stop()

# Entrena modelos (si ventas fallan, no truena toda la app: solo desactiva predicción de ventas)
try:
    success_model, acc, auc, cm = train_success_model(df)
except Exception as e:
    st.error(f"Error entrenando modelo de éxito: {e}")
    st.stop()

sales_model = None
mae = None
try:
    sales_model, mae = train_sales_model(df)
except Exception as e:
    st.sidebar.warning(f"Modelo de ventas desactivado: {e}")

# ============================================================
# HEADER
# ============================================================
st.title("🧠 Plataforma IA: Producto + Empaque + Claims")
st.caption("Éxito + (Ventas si hay datos) + Insights + Pack Lab + Claims Lab + Experimentos")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Registros", f"{len(df):,}")
k2.metric("Precisión", f"{acc*100:.2f}%")
k3.metric("AUC", f"{auc:.3f}")
k4.metric("Éxito base", f"{df['exito'].mean()*100:.1f}%")
k5.metric("MAE ventas", f"{mae:,.0f} u." if mae is not None else "N/A")

st.divider()

tab_sim, tab_ins, tab_pack, tab_claims, tab_exp, tab_data, tab_model = st.tabs(
    ["🧪 Simulador", "📊 Insights", "📦 Pack Lab", "🏷️ Claims Lab", "🧪 Experimentos", "📂 Datos", "🧠 Modelo"]
)

# ============================================================
# TAB: SIMULADOR
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

    recs = recommend_claims(segmento, canal, max_claims=8)
    claim_options = [c for c, _ in recs]
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
    pack_emotion = pack_emotion_score(
        pack_legibility_score, pack_shelf_pop_score, pack_clarity_score,
        cscore, copy_tone
    )
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

        st.metric("Prob. éxito", safe_percent(p))
        st.metric("Predicción", "✅ Éxito" if pred else "⚠️ Riesgo")

        if sales_model is not None:
            ventas = max(0, round(float(sales_model.predict(entrada)[0])))
            ingresos = ventas * float(precio)
            utilidad = ventas * (float(precio) * (float(margen_pct) / 100.0))
            cA, cB, cC = st.columns(3)
            cA.metric("Ventas (unidades)", f"{ventas:,.0f}")
            cB.metric("Ingresos ($)", f"${ingresos:,.0f}")
            cC.metric("Utilidad ($)", f"${utilidad:,.0f}")
        else:
            st.info("Modelo de ventas no disponible (tu CSV no tiene ventas suficientes).")

        st.dataframe(entrada, use_container_width=True)


# ============================================================
# TAB: INSIGHTS
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
        if "ventas_unidades" in df.columns and df["ventas_unidades"].notna().any():
            v_marca = df.groupby("marca")[["ventas_unidades"]].mean().sort_values("ventas_unidades", ascending=False).round(0)
            st.dataframe(v_marca.reset_index(), use_container_width=True)
        else:
            st.info("No hay ventas_unidades disponibles en el dataset.")

    with right:
        st.markdown("**Marca + Canal (Conexión promedio)**")
        ins_mc = df.groupby(["marca", "canal"])[["conexion_score"]].mean().sort_values("conexion_score", ascending=False).round(2)
        st.dataframe(ins_mc.head(25).reset_index(), use_container_width=True)

        st.markdown("**Marca + Canal (Éxito %)**")
        ex_mc = df.groupby(["marca", "canal"])[["exito"]].mean().sort_values("exito", ascending=False).round(3)
        ex_mc["exito_%"] = (ex_mc["exito"] * 100).round(1)
        st.dataframe(ex_mc.head(25)[["exito_%"]].reset_index(), use_container_width=True)

        st.markdown("**Marca + Canal (Ventas promedio)**")
        if "ventas_unidades" in df.columns and df["ventas_unidades"].notna().any():
            v_mc = df.groupby(["marca", "canal"])[["ventas_unidades"]].mean().sort_values("ventas_unidades", ascending=False).round(0)
            st.dataframe(v_mc.head(25).reset_index(), use_container_width=True)
        else:
            st.info("No hay ventas_unidades disponibles en el dataset.")

    st.divider()
    d1, d2 = st.columns(2)
    with d1:
        bins = pd.cut(df["conexion_score"], bins=[0, 20, 40, 60, 80, 100], include_lowest=True)
        dist = bins.value_counts().sort_index()
        bar_plot_from_value_counts(dist, "Distribución: Conexión emocional (bucket)")
    with d2:
        if "ventas_unidades" in df.columns and df["ventas_unidades"].notna().any():
            bins2 = pd.cut(df["ventas_unidades"].fillna(0).clip(0, 40000), bins=[0, 2000, 5000, 10000, 20000, 40000], include_lowest=True)
            dist2 = bins2.value_counts().sort_index()
            bar_plot_from_value_counts(dist2, "Distribución: Ventas unidades (bucket)")
        else:
            st.info("No hay ventas_unidades para graficar.")


# ============================================================
# TAB: PACK LAB
# ============================================================
with tab_pack:
    st.subheader("📦 Pack Lab (sube tu empaque y te doy recomendaciones)")

    img_file = st.file_uploader("Sube imagen del empaque (PNG/JPG)", type=["png", "jpg", "jpeg"], key="pack_uploader")
    if img_file is None:
        st.info("Sube una imagen para generar análisis del empaque.")
    else:
        img = Image.open(img_file)
        st.image(img, caption="Empaque cargado", use_container_width=True)

        m = image_metrics(img)
        scores = pack_scores_from_metrics(m)

        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Brillo", f"{m['brightness']:.2f}")
        a2.metric("Contraste", f"{m['contrast']:.2f}")
        a3.metric("Colorfulness", f"{m['colorfulness']:.2f}")
        a4.metric("Edge density", f"{m['edge_density']:.3f}")

        b1, b2, b3 = st.columns(3)
        b1.metric("Legibilidad", f"{scores['pack_legibility_score']}/100")
        b2.metric("Shelf Pop", f"{scores['pack_shelf_pop_score']}/100")
        b3.metric("Claridad", f"{scores['pack_clarity_score']}/100")

        st.markdown("### Recomendaciones rápidas")
        recs_list = []
        if scores["pack_legibility_score"] < 60:
            recs_list.append("• Sube legibilidad: más contraste texto/fondo, tipografía más gruesa y menos ruido alrededor del claim principal.")
        if scores["pack_clarity_score"] < 60:
            recs_list.append("• Mejora claridad: reduce elementos, deja aire y limita a 2–3 claims.")
        if scores["pack_shelf_pop_score"] < 60:
            recs_list.append("• Sube shelf pop: usa color acento y evita pack muy oscuro o demasiado lavado.")
        if m["edge_density"] > 0.28:
            recs_list.append("• Saturación visual alta: simplifica fondos, patrones y microtextos.")
        if not recs_list:
            recs_list.append("• Visualmente va bien. Ajusta jerarquía: Marca → beneficio → variedad → credencial.")
        st.write("\n".join(recs_list))


# ============================================================
# TAB: CLAIMS LAB
# ============================================================
with tab_claims:
    st.subheader("🏷️ Claims Lab (claims ganadores)")

    c1, c2 = st.columns(2)
    segmento = c1.selectbox("Segmento", ["fit", "kids", "premium", "value"], 0, key="claims_segmento")
    canal_c = c2.selectbox("Canal", ["retail", "marketplace"], 0, key="claims_canal")

    recs = recommend_claims(segmento, canal_c, max_claims=10)
    rec_df = pd.DataFrame(recs, columns=["claim", "score_base"])
    rec_df["score_base"] = (rec_df["score_base"] * 100).round(1)
    st.dataframe(rec_df, use_container_width=True)

    options = rec_df["claim"].tolist()
    selected = st.multiselect("Selecciona 2-3 claims", options, default=options[:2], key="claims_selected")
    cscore = claims_score(selected, canal_c)
    st.metric("Claims Score", f"{cscore:.1f}/100")

    st.warning("Nota: esto es recomendación comercial (no legal/regulatoria).")


# ============================================================
# TAB: EXPERIMENTOS
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
            pivot = exp_df.pivot_table(
                index=["experimento", "metrica"],
                columns="variante",
                values="valor",
                aggfunc="mean"
            )
            pivot["lift_B_vs_A"] = (pivot.get("B") - pivot.get("A"))
            st.dataframe(pivot.reset_index().round(3), use_container_width=True)
        except Exception:
            st.info("Necesitas registros en A y B para calcular lift.")
    else:
        st.info("Aún no hay experimentos guardados.")


# ============================================================
# TAB: DATOS
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


# ============================================================
# TAB: MODELO
# ============================================================
with tab_model:
    st.subheader("🧠 Diagnóstico del modelo de Éxito")
    st.dataframe(
        pd.DataFrame(cm, index=["Real 0", "Real 1"], columns=["Pred 0", "Pred 1"]),
        use_container_width=True
    )
    st.write(f"AUC: **{auc:.3f}** | Accuracy: **{acc*100:.2f}%**")
    if mae is not None:
        st.write(f"Error absoluto medio (ventas): **{mae:,.0f}** unidades.")
    else:
        st.write("Modelo de ventas: **No entrenado / no disponible** (faltan ventas).")