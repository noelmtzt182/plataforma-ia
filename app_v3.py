# app_v3.py
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


import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, mean_absolute_error

# ----------------------------
# App Config
# ----------------------------
st.set_page_config(
    page_title="Plataforma IA | Producto + Empaque + Claims + Shelf (v2.3)",
    layout="wide"
)

DATA_PATH_DEFAULT = "mercado_cereales_5000_con_ventas.csv"

REQUIRED_BASE = {
    "marca", "categoria", "canal", "precio", "costo", "margen", "margen_pct",
    "competencia", "demanda", "tendencia", "estacionalidad",
    "rating_conexion", "comentario", "sentiment_score",
    "conexion_score", "conexion_alta", "score_latente", "exito"
}
REQUIRED_SALES = {"ventas_unidades", "ventas_ingresos", "utilidad"}

# ----------------------------
# Small Helpers
# ----------------------------
def _clean_str_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()

def clip(v, a, b):
    return float(max(a, min(b, v)))

def safe_percent(x):
    return f"{x*100:.1f}%"

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def bar_df_from_value_counts(vc: pd.Series) -> pd.DataFrame:
    # Convierte vc (Series) a DataFrame con columnas estándar -> evita Altair schema errors
    out = vc.reset_index()
    out = out.iloc[:, :2].copy()
    out.columns = ["bucket", "count"]
    out["bucket"] = out["bucket"].astype(str)
    out["count"] = pd.to_numeric(out["count"], errors="coerce").fillna(0)
    return out

def softmax(x):
    x = np.array(x, dtype=float)
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)

# Session defaults
if "learning_log" not in st.session_state:
    st.session_state.learning_log = []
if "last_sim" not in st.session_state:
    st.session_state.last_sim = None
if "last_shelf" not in st.session_state:
    st.session_state.last_shelf = None
if "last_new" not in st.session_state:
    st.session_state.last_new = None
if "last_invest" not in st.session_state:
    st.session_state.last_invest = None

# ============================================================
# BLOQUE 2 — Claims + Pack Vision+ + Shelf & Emotion (3s)
# ============================================================

# ----------------------------
# Claims library ampliada
# (heurística -> luego lo conectas a research real / reviews / social)
# ----------------------------
CLAIMS_LIBRARY = {
    "fit": [
        ("alto en proteína", 0.90),
        ("sin azúcar añadida", 0.89),
        ("alto en fibra", 0.87),
        ("integral", 0.82),
        ("bajo en calorías", 0.78),
        ("sin colorantes artificiales", 0.76),
        ("con avena", 0.74),
        ("sin jarabe de maíz", 0.72),
    ],
    "kids": [
        ("con vitaminas y minerales", 0.86),
        ("sabor chocolate", 0.84),
        ("energía para su día", 0.80),
        ("con calcio", 0.74),
        ("sin conservadores", 0.72),
        ("con hierro", 0.70),
        ("divertido y crujiente", 0.68),
    ],
    "premium": [
        ("ingredientes seleccionados", 0.84),
        ("hecho con avena real", 0.80),
        ("sabor intenso", 0.78),
        ("calidad premium", 0.74),
        ("receta artesanal", 0.66),
        ("sin sabores artificiales", 0.68),
        ("granos enteros", 0.70),
    ],
    "value": [
        ("rinde más", 0.80),
        ("ideal para la familia", 0.76),
        ("gran sabor a mejor precio", 0.74),
        ("económico y práctico", 0.70),
        ("bolsa resellable", 0.62),
        ("más por menos", 0.64),
    ],
}

CANAL_CLAIM_BOOST = {
    "retail": {
        "sin azúcar añadida": 1.05,
        "alto en fibra": 1.04,
        "ideal para la familia": 1.03,
        "con vitaminas y minerales": 1.03,
    },
    "marketplace": {
        "alto en proteína": 1.06,
        "ingredientes seleccionados": 1.04,
        "sin colorantes artificiales": 1.05,
        "hecho con avena real": 1.03,
    }
}

def recommend_claims(segment: str, canal: str, max_claims: int = 8):
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
    boosts = []
    for c in selected_claims:
        c = str(c).lower().strip()
        boosts.append(CANAL_CLAIM_BOOST.get(can, {}).get(c, 1.0))
    base = float(np.mean(boosts)) if boosts else 1.0
    n = len(selected_claims)
    clarity_penalty = 1.0 if n <= 3 else max(0.65, 1.0 - 0.12*(n-3))
    score = 75 * base * clarity_penalty
    return float(np.clip(score, 0, 100))

# ----------------------------
# Pack Vision: métricas rápidas (sin OCR pesado)
# ----------------------------
def image_metrics(img: Image.Image) -> dict:
    im = img.convert("RGB")
    arr = np.asarray(im).astype(np.float32)

    gray = (0.299*arr[...,0] + 0.587*arr[...,1] + 0.114*arr[...,2])
    brightness = float(np.mean(gray) / 255.0)
    contrast = float(np.std(gray) / 255.0)

    rg = arr[...,0] - arr[...,1]
    yb = 0.5*(arr[...,0] + arr[...,1]) - arr[...,2]
    colorfulness = float((np.std(rg) + 0.3*np.std(yb)) / 255.0)

    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:,1:-1] = gray[:,2:] - gray[:,:-2]
    gy[1:-1,:] = gray[2:,:] - gray[:-2,:]
    mag = np.sqrt(gx**2 + gy**2)
    thresh = np.percentile(mag, 85)
    edges = (mag > thresh).astype(np.float32)
    edge_density = float(np.mean(edges))

    pop_score = clip(0.55*contrast + 0.45*colorfulness, 0, 1)

    return {
        "brightness": brightness,
        "contrast": contrast,
        "colorfulness": colorfulness,
        "edge_density": edge_density,
        "pop_score": pop_score
    }

def pack_scores_from_metrics(m: dict) -> dict:
    # legibility: contraste + densidad de borde “moderada”
    legibility = 0.7*m["contrast"] + 0.3*(1 - abs(m["edge_density"] - 0.18)/0.18)
    legibility = clip(legibility, 0, 1) * 100

    target_brightness = 0.55
    brightness_fit = 1 - abs(m["brightness"] - target_brightness)/target_brightness
    shelf_pop = clip(0.75*m["pop_score"] + 0.25*clip(brightness_fit, 0, 1), 0, 1) * 100

    clarity = clip(0.6*m["contrast"] + 0.4*(1 - clip(m["edge_density"]/0.35, 0, 1)), 0, 1) * 100

    return {
        "pack_legibility_score": round(float(legibility), 1),
        "pack_shelf_pop_score": round(float(shelf_pop), 1),
        "pack_clarity_score": round(float(clarity), 1),
    }

def pack_quick_wins(scores: dict, m: dict) -> list[str]:
    out = []
    if scores["pack_legibility_score"] < 60:
        out.append("Sube contraste texto/fondo y simplifica tipografía (menos microtextos).")
    if scores["pack_clarity_score"] < 60:
        out.append("Reduce ruido visual: menos elementos, más aire; prioriza 1 beneficio principal.")
    if scores["pack_shelf_pop_score"] < 60:
        out.append("Aumenta shelf pop: color acento y jerarquía (marca→beneficio→variedad).")
    if m["edge_density"] > 0.28:
        out.append("Saturación alta: elimina patrones, fondos complejos y exceso de sellos.")
    if not out:
        out.append("Se ve sólido. Ajusta jerarquía y prueba 2–3 claims máximo.")
    return out

def pack_heatmap_image_from_edges(img: Image.Image) -> Image.Image:
    # “heatmap” liviano: edges -> overlay rojo
    im = img.convert("RGB")
    arr = np.asarray(im).astype(np.float32)
    gray = (0.299*arr[...,0] + 0.587*arr[...,1] + 0.114*arr[...,2])

    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:,1:-1] = gray[:,2:] - gray[:,:-2]
    gy[1:-1,:] = gray[2:,:] - gray[:-2,:]
    mag = np.sqrt(gx**2 + gy**2)
    mag = mag / (np.max(mag) + 1e-6)

    # create red overlay
    overlay = arr.copy()
    overlay[...,0] = np.clip(overlay[...,0] + 120*mag, 0, 255)
    overlay[...,1] = np.clip(overlay[...,1] * (1 - 0.35*mag), 0, 255)
    overlay[...,2] = np.clip(overlay[...,2] * (1 - 0.35*mag), 0, 255)

    return Image.fromarray(overlay.astype(np.uint8))

# ----------------------------
# Emotion proxy (simple + útil)
# ----------------------------
def pack_emotion_score(pack_legibility, pack_pop, pack_clarity, claims_score_val, copy_tone: int):
    visual = 0.40*(pack_pop/100) + 0.30*(pack_clarity/100) + 0.15*(pack_legibility/100)
    claims = 0.15*(claims_score_val/100)
    tone_boost = 0.06 if copy_tone > 0 else (-0.06 if copy_tone < 0 else 0.0)
    score = (visual + claims + tone_boost) * 100
    return float(np.clip(score, 0, 100))

# ----------------------------
# Shelf & Emotion Predictor (3-second)
# - "choice_3s": score 0-100
# - MNL multinomial: probabilidad de elección vs competidores
# ----------------------------
def pack_3sec_choice_score(leg, pop, clarity, emotion):
    # combinación simple
    s = 0.35*(pop/100) + 0.25*(clarity/100) + 0.20*(leg/100) + 0.20*(emotion/100)
    return float(np.clip(s*100, 0, 100))

def shelf_rank_from_pack_scores(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows).copy()
    df["utility"] = df["choice_3s"] / 10.0  # escala suave
    probs = softmax(df["utility"].values)
    df["mnl_prob"] = probs
    df = df.sort_values("mnl_prob", ascending=False).reset_index(drop=True)
    df["mnl_prob_%"] = (df["mnl_prob"]*100).round(1)
    return df[["pack", "choice_3s", "mnl_prob_%"]]

def crop_image(img: Image.Image, x1, y1, x2, y2) -> Image.Image:
    w,h = img.size
    x1 = int(np.clip(x1, 0, w-1)); x2 = int(np.clip(x2, 1, w))
    y1 = int(np.clip(y1, 0, h-1)); y2 = int(np.clip(y2, 1, h))
    if x2 <= x1+5: x2 = min(w, x1+50)
    if y2 <= y1+5: y2 = min(h, y1+50)
    return img.crop((x1,y1,x2,y2))

def draw_rois(img: Image.Image, rois: list[tuple], labels: list[str]) -> Image.Image:
    im = img.convert("RGB").copy()
    draw = ImageDraw.Draw(im)
    for (x1,y1,x2,y2), lab in zip(rois, labels):
        draw.rectangle([x1,y1,x2,y2], outline=(255,0,0), width=3)
        draw.text((x1+4, y1+4), lab, fill=(255,0,0))
    return im

# ============================================================
# BLOQUE 3 — Load Data + Train Models + Market Loader
# ============================================================

@st.cache_data
def load_data(path_or_file) -> pd.DataFrame:
    df = pd.read_csv(path_or_file).copy()

    # normaliza strings
    for c in ["marca","categoria","canal","estacionalidad","comentario"]:
        if c in df.columns:
            df[c] = _clean_str_series(df[c])

    missing = sorted(list(REQUIRED_BASE - set(df.columns)))
    if missing:
        raise ValueError(f"Faltan columnas base en el CSV: {missing}")

    # num cols
    num_cols = [
        "precio","costo","margen","margen_pct",
        "competencia","demanda","tendencia",
        "rating_conexion","sentiment_score",
        "conexion_score","conexion_alta",
        "score_latente","exito"
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ["ventas_unidades","ventas_ingresos","utilidad"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # drop NA mínimos
    df = df.dropna(subset=[
        "marca","categoria","canal","precio","competencia",
        "demanda","tendencia","margen_pct",
        "conexion_score","rating_conexion","sentiment_score","exito"
    ])

    df["exito"] = df["exito"].astype(int)

    # sanity
    df["ventas_unidades"] = pd.to_numeric(df.get("ventas_unidades", np.nan), errors="coerce")
    df["ventas_unidades"] = df["ventas_unidades"].fillna(df["ventas_unidades"].median() if "ventas_unidades" in df.columns else 0)

    return df

@st.cache_data
def load_market_intel(path_or_file):
    try:
        mdf = pd.read_csv(path_or_file).copy()
        # columnas sugeridas: fecha, fuente, categoria, marca, canal, precio_prom, share, tendencia, comentario
        for c in ["fuente","categoria","marca","canal","comentario"]:
            if c in mdf.columns:
                mdf[c] = _clean_str_series(mdf[c])
        return mdf
    except Exception:
        return None

@st.cache_resource
def train_models(df: pd.DataFrame):
    features = [
        "precio","competencia","demanda","tendencia","margen_pct",
        "conexion_score","rating_conexion","sentiment_score",
        "marca","canal"
    ]
    X = df[features].copy()
    y = df["exito"].astype(int)

    num_cols = [
        "precio","competencia","demanda","tendencia","margen_pct",
        "conexion_score","rating_conexion","sentiment_score"
    ]
    cat_cols = ["marca","canal"]

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
    proba = clf.predict_proba(X_test)[:,1]
    ACC = accuracy_score(y_test, pred)
    AUC = roc_auc_score(y_test, proba)
    CM = confusion_matrix(y_test, pred)

    # Regresión (ventas)
    if not REQUIRED_SALES.issubset(set(df.columns)):
        # si no trae ventas, crea proxy para no tronar
        df = df.copy()
        df["ventas_unidades"] = (
            1200
            + 35*df["demanda"]
            + 18*df["tendencia"]
            - 6*df["precio"]
            + 20*df["margen_pct"]
            + 12*df["conexion_score"]
        ).clip(0)

    yv = df["ventas_unidades"].astype(float)
    X2 = df[features].copy()

    reg = Pipeline(steps=[
        ("preprocessor", pre),
        ("model", RandomForestRegressor(n_estimators=350, random_state=42))
    ])

    X_train2, X_test2, y_train2, y_test2 = train_test_split(
        X2, yv, test_size=0.2, random_state=42
    )
    reg.fit(X_train2, y_train2)
    predv = reg.predict(X_test2)
    MAE = mean_absolute_error(y_test2, predv)

    return clf, reg, ACC, AUC, CM, MAE

# ----------------------------
# Sidebar: load data
# ----------------------------
st.sidebar.title("⚙️ Datos")

uploaded = st.sidebar.file_uploader("Sube tu dataset (CSV con ventas)", type=["csv"], key="uploader_dataset")
if uploaded is not None:
    df = load_data(uploaded)
else:
    if Path(DATA_PATH_DEFAULT).exists():
        df = load_data(DATA_PATH_DEFAULT)
    else:
        st.sidebar.warning(f"No encontré '{DATA_PATH_DEFAULT}'. Sube tu CSV.")
        st.stop()

# Market intel
st.sidebar.subheader("📈 Market Intelligence")
market_up = st.sidebar.file_uploader("Sube market_intel.csv (opcional)", type=["csv"], key="uploader_market")
market_df = load_market_intel(market_up) if market_up else None

# Train models
try:
    success_model, sales_model, ACC, AUC, CM, MAE = train_models(df)
except Exception as e:
    st.error(f"Error entrenando modelos: {e}")
    st.stop()

# ----------------------------
# Header metrics
# ----------------------------
st.title("🧠 Plataforma IA: Producto + Empaque + Claims + Shelf (v2.3)")
st.caption("Éxito + Ventas + Insights + Pack Vision+ + Shelf 3-Second + Producto Nuevo + Inversionista + Reportes")

k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("Registros", f"{len(df):,}")
k2.metric("Precisión", f"{ACC*100:.2f}%")
k3.metric("AUC", f"{AUC:.3f}")
k4.metric("Éxito base", f"{df['exito'].mean()*100:.1f}%")
k5.metric("MAE ventas", f"{MAE:,.0f} u.")

st.divider()

# ============================================================
# BLOQUE 4 — Tabs + Simulador + Insights
# ============================================================

tab_sim, tab_ins, tab_claims, tab_pack, tab_shelf, tab_new, tab_invest, tab_report, tab_market, tab_data, tab_diag = st.tabs([
    "🧪 Simulador",
    "📊 Insights",
    "🏷️ Claims Lab",
    "📦 Pack Vision+",
    "🧲 Shelf & Emotion (3s)",
    "🧊 Producto Nuevo",
    "💼 Inversionista",
    "📄 Reporte Ejecutivo",
    "📈 Market Intelligence",
    "📂 Datos",
    "🧠 Diagnóstico",
])

# ============================================================
# 🧪 SIMULADOR
# ============================================================
with tab_sim:
    st.subheader("🧪 Simulador What-If (éxito + ventas + pack + claims + conexión + ROI)")

    marcas = sorted(df["marca"].unique().tolist())
    canales = sorted(df["canal"].unique().tolist())

    c1,c2,c3 = st.columns(3)
    marca = c1.selectbox("Marca", marcas, key="sim_marca")
    canal = c2.selectbox("Canal", canales, key="sim_canal")
    segmento = c3.selectbox("Segmento", ["fit","kids","premium","value"], key="sim_segmento")

    canal_norm = str(canal).lower().strip()

    st.markdown("### Variables de negocio")
    b1,b2,b3,b4,b5 = st.columns(5)
    precio = b1.number_input("Precio", value=float(df["precio"].median()), key="sim_precio")
    competencia = b2.slider("Competencia (1-10)", 1, 10, int(df["competencia"].median()), key="sim_competencia")
    demanda = b3.slider("Demanda (10-100)", 10, 100, int(df["demanda"].median()), key="sim_demanda")
    tendencia = b4.slider("Tendencia (20-100)", 20, 100, int(df["tendencia"].median()), key="sim_tendencia")
    margen_pct = b5.slider("Margen %", 0, 90, int(np.clip(df["margen_pct"].median(), 0, 90)), key="sim_margen_pct")

    st.markdown("### Claims")
    recs = recommend_claims(segmento, canal_norm, 8)
    claim_opts = [c for c,_ in recs]
    selected_claims = st.multiselect("Selecciona claims (ideal 2-3)", claim_opts, default=claim_opts[:2], key="sim_claims")
    cscore = claims_score(selected_claims, canal_norm)

    st.markdown("### Empaque (manual sliders)")
    p1,p2,p3 = st.columns(3)
    pack_leg = p1.slider("Pack legibilidad (0-100)", 0, 100, 65, key="sim_pack_leg")
    pack_pop = p2.slider("Pack shelf pop (0-100)", 0, 100, 70, key="sim_pack_pop")
    pack_cla = p3.slider("Pack claridad (0-100)", 0, 100, 65, key="sim_pack_cla")

    pack_emotion = pack_emotion_score(pack_leg, pack_pop, pack_cla, cscore, 0)

    # conexión proxy (para simulación)
    conexion_score = clip(0.45*demanda + 0.35*pack_pop + 0.20*cscore, 0, 100)

    entrada = pd.DataFrame([{
        "precio": float(precio),
        "competencia": float(competencia),
        "demanda": float(demanda),
        "tendencia": float(tendencia),
        "margen_pct": float(margen_pct),
        "conexion_score": float(conexion_score),
        "rating_conexion": float(7),
        "sentiment_score": float(1),
        "marca": str(marca).lower(),
        "canal": str(canal_norm).lower(),
    }])

    s1,s2,s3,s4 = st.columns(4)
    s1.metric("Claims Score", f"{cscore:.1f}/100")
    s2.metric("Emotion Pack Score", f"{pack_emotion:.1f}/100")
    s3.metric("Conexión final", f"{conexion_score:.1f}/100")
    s4.metric("Pack pop", f"{pack_pop:.0f}/100")

    # ---------- ROI inputs (no se borra porque tiene keys únicas)
    st.divider()
    st.subheader("🎯 ROI (Financiero + Unidades)")

    r1,r2,r3 = st.columns(3)
    inversion = r1.number_input("Inversión ($) (opcional)", 0.0, key="sim_roi_inv")
    meta_u = r2.number_input("Meta unidades (opcional)", 0.0, key="sim_roi_goal_u")
    base_u = r3.number_input("Baseline unidades (opcional)", 0.0, key="sim_roi_base_u")

    if st.button("🚀 Simular", key="sim_btn"):
        prob = float(success_model.predict_proba(entrada)[0][1])
        ventas = max(0.0, float(sales_model.predict(entrada)[0]))

        ingresos = ventas * float(precio)
        utilidad = ventas * (float(precio) * (float(margen_pct)/100.0))

        st.session_state.last_sim = {
            "marca": marca, "canal": canal_norm, "segmento": segmento,
            "precio": float(precio), "competencia": float(competencia),
            "demanda": float(demanda), "tendencia": float(tendencia),
            "margen_pct": float(margen_pct),
            "claims": selected_claims, "claims_score": float(cscore),
            "pack_leg": float(pack_leg), "pack_pop": float(pack_pop), "pack_cla": float(pack_cla),
            "pack_emotion": float(pack_emotion),
            "conexion_score": float(conexion_score),
            "prob_exito": float(prob),
            "ventas_unidades": float(ventas),
            "ingresos": float(ingresos),
            "utilidad": float(utilidad),
            "roi_inversion": float(inversion),
            "roi_meta_u": float(meta_u),
            "roi_base_u": float(base_u),
        }

        o1,o2,o3,o4 = st.columns(4)
        o1.metric("Prob. éxito", f"{prob*100:.1f}%")
        o2.metric("Ventas predichas", f"{ventas:,.0f} u.")
        o3.metric("Ingresos", f"${ingresos:,.0f}")
        o4.metric("Utilidad", f"${utilidad:,.0f}")

        # ROI outputs
        st.markdown("### ROI")
        if inversion > 0:
            roi_fin = (utilidad - inversion) / max(inversion, 1e-9)
            st.metric("ROI financiero", f"{roi_fin*100:.1f}%")
        else:
            st.metric("ROI financiero", "—")

        if meta_u > 0:
            cumplimiento = ventas / max(meta_u, 1e-9)
            st.metric("Cumplimiento vs meta", f"{cumplimiento*100:.1f}%")
        else:
            st.metric("Cumplimiento vs meta", "—")

        if base_u > 0:
            uplift = (ventas - base_u) / max(base_u, 1e-9)
            st.metric("Uplift vs baseline", f"{uplift*100:.1f}%")
        else:
            st.metric("Uplift vs baseline", "—")

        st.dataframe(entrada, use_container_width=True)

# ============================================================
# 📊 INSIGHTS
# ============================================================
with tab_ins:
    st.subheader("📊 Insights (rankings + distribuciones)")

    left,right = st.columns(2)
    with left:
        st.markdown("**Ranking por marca (Conexión promedio)**")
        st.dataframe(
            df.groupby("marca")[["conexion_score"]].mean().sort_values("conexion_score", ascending=False).round(2),
            use_container_width=True
        )

        st.markdown("**Ranking por marca (Éxito %)**")
        ex_m = df.groupby("marca")[["exito"]].mean().sort_values("exito", ascending=False)
        ex_m["exito_%"] = (ex_m["exito"]*100).round(1)
        st.dataframe(ex_m[["exito_%"]], use_container_width=True)

    with right:
        st.markdown("**Ranking por marca (Ventas promedio)**")
        st.dataframe(
            df.groupby("marca")[["ventas_unidades"]].mean().sort_values("ventas_unidades", ascending=False).round(0),
            use_container_width=True
        )

        st.markdown("**Marca + Canal (Ventas promedio)**")
        st.dataframe(
            df.groupby(["marca","canal"])[["ventas_unidades"]].mean().sort_values("ventas_unidades", ascending=False).head(25).round(0),
            use_container_width=True
        )

    st.divider()
    d1,d2 = st.columns(2)

    with d1:
        bins = pd.cut(df["conexion_score"], bins=[0,20,40,60,80,100], include_lowest=True)
        vc = bins.value_counts().sort_index()
        bar = bar_df_from_value_counts(vc)
        st.bar_chart(bar.set_index("bucket"), use_container_width=True)

    with d2:
        bins2 = pd.cut(df["ventas_unidades"].clip(0, 40000), bins=[0,2000,5000,10000,20000,40000], include_lowest=True)
        vc2 = bins2.value_counts().sort_index()
        bar2 = bar_df_from_value_counts(vc2)
        st.bar_chart(bar2.set_index("bucket"), use_container_width=True)

# ============================================================
# BLOQUE 5 — Claims Lab + Pack Vision+ + Shelf & Emotion (3s)
# ============================================================

# ============================================================
# 🏷️ CLAIMS LAB
# ============================================================
with tab_claims:
    st.subheader("🏷️ Claims Lab (recomendaciones + score)")

    c1,c2 = st.columns(2)
    seg = c1.selectbox("Segmento", ["fit","kids","premium","value"], key="claims_seg")
    can = c2.selectbox("Canal", ["retail","marketplace"], key="claims_can")

    recs = recommend_claims(seg, can, 12)
    rec_df = pd.DataFrame(recs, columns=["claim","score"])
    rec_df["score"] = (rec_df["score"]*100).round(1)

    st.dataframe(rec_df, use_container_width=True)

    selected = st.multiselect("Selecciona 2-3 claims", rec_df["claim"].tolist(), default=rec_df["claim"].tolist()[:2], key="claims_selected")
    cscore = claims_score(selected, can)
    st.metric("Claims Score", f"{cscore:.1f}/100")

    st.info("Nota: Recomendación comercial. Valida compliance/regulatorio antes de imprimir en empaque.")

# ============================================================
# 📦 PACK VISION+
# ============================================================
with tab_pack:
    st.subheader("📦 Pack Vision+ (imagen -> métricas -> heatmap -> quick wins)")

    img_file = st.file_uploader("Sube imagen del empaque (PNG/JPG)", type=["png","jpg","jpeg"], key="pack_uploader")
    if img_file is None:
        st.info("Sube tu empaque para análisis visual.")
    else:
        img = Image.open(img_file)
        st.image(img, caption="Empaque cargado", use_container_width=True)

        m = image_metrics(img)
        sc = pack_scores_from_metrics(m)
        heat = pack_heatmap_image_from_edges(img)

        a1,a2,a3,a4 = st.columns(4)
        a1.metric("Brillo", f"{m['brightness']:.2f}")
        a2.metric("Contraste", f"{m['contrast']:.2f}")
        a3.metric("Colorfulness", f"{m['colorfulness']:.2f}")
        a4.metric("Edge density", f"{m['edge_density']:.3f}")

        b1,b2,b3 = st.columns(3)
        b1.metric("Legibilidad", f"{sc['pack_legibility_score']}/100")
        b2.metric("Shelf Pop", f"{sc['pack_shelf_pop_score']}/100")
        b3.metric("Claridad", f"{sc['pack_clarity_score']}/100")

        st.image(heat, caption="Heatmap (proxy de atención visual)", use_container_width=True)

        st.markdown("### Quick wins (pack)")
        for w in pack_quick_wins(sc, m):
            st.write("•", w)

# ============================================================
# 🧲 SHELF & EMOTION (3s)
# - pack suelto vs competidores
# - foto de anaquel + recortes ROI
# - ranking + MNL
# - learning log descargable
# ============================================================
with tab_shelf:
    st.subheader("🧲 Shelf & Emotion Predictor (3-Second Test)")

    st.caption("Sube tu pack y competidores (o foto de anaquel + recortes) para estimar atención/elección y simular MNL.")

    # ----------------------------
    # MODO A: Packs sueltos
    # ----------------------------
    st.markdown("## A) Packs sueltos (tu pack vs competidores)")
    p0 = st.file_uploader("Tu pack", type=["png","jpg","jpeg"], key="shelf_pack_0")
    p1 = st.file_uploader("Competidor 1", type=["png","jpg","jpeg"], key="shelf_pack_1")
    p2 = st.file_uploader("Competidor 2", type=["png","jpg","jpeg"], key="shelf_pack_2")
    p3 = st.file_uploader("Competidor 3 (opcional)", type=["png","jpg","jpeg"], key="shelf_pack_3")

    packs = [p0,p1,p2,p3]
    rows = []
    if any(packs):
        for i,f in enumerate(packs):
            if f is None:
                continue
            im = Image.open(f)
            m = image_metrics(im)
            sc = pack_scores_from_metrics(m)

            # emotion proxy (sin claims aquí -> usa neutral 60)
            emotion = 60.0
            choice = pack_3sec_choice_score(
                sc["pack_legibility_score"],
                sc["pack_shelf_pop_score"],
                sc["pack_clarity_score"],
                emotion
            )

            rows.append({"pack": f"pack_{i}", "choice_3s": round(choice,1)})

        if rows:
            rank_df = shelf_rank_from_pack_scores(rows)
            st.dataframe(rank_df, use_container_width=True)

    # ----------------------------
    # MODO B: Foto de anaquel + ROI recortes
    # ----------------------------
    st.divider()
    st.markdown("## B) Foto de anaquel + recortes ROI (tu pack + hasta 3 competidores)")
    shelf_img_file = st.file_uploader("Sube foto de anaquel", type=["png","jpg","jpeg"], key="shelf_photo")

    if shelf_img_file:
        shelf_img = Image.open(shelf_img_file)
        st.image(shelf_img, caption="Anaquel (original)", use_container_width=True)

        w,h = shelf_img.size
        st.markdown("### Define recortes (ROI) con sliders (x1,y1,x2,y2)")
        st.caption("Tip: empieza con 4 ROIs: tu pack + 3 competidores. Ajusta a ojo.")

        def roi_controls(idx, label):
            st.markdown(f"**ROI {idx}: {label}**")
            c1,c2,c3,c4 = st.columns(4)
            x1 = c1.slider("x1", 0, w-1, int(w*0.05), key=f"roi_{idx}_x1")
            y1 = c2.slider("y1", 0, h-1, int(h*0.10), key=f"roi_{idx}_y1")
            x2 = c3.slider("x2", 1, w, int(w*0.25), key=f"roi_{idx}_x2")
            y2 = c4.slider("y2", 1, h, int(h*0.40), key=f"roi_{idx}_y2")
            return (x1,y1,x2,y2)

        labels = [
            st.text_input("Etiqueta ROI 0 (tu pack)", "tu_pack", key="roi_lab_0"),
            st.text_input("Etiqueta ROI 1", "comp_1", key="roi_lab_1"),
            st.text_input("Etiqueta ROI 2", "comp_2", key="roi_lab_2"),
            st.text_input("Etiqueta ROI 3", "comp_3", key="roi_lab_3"),
        ]

        rois = [
            roi_controls(0, labels[0]),
            roi_controls(1, labels[1]),
            roi_controls(2, labels[2]),
            roi_controls(3, labels[3]),
        ]

        preview = draw_rois(shelf_img, rois, labels)
        st.image(preview, caption="Preview ROIs", use_container_width=True)

        # Compute
        if st.button("🧲 Calcular Shelf 3s (anaquel)", key="btn_shelf_calc"):
            rows2 = []
            crops_show = st.columns(4)

            for idx,(roi,lab) in enumerate(zip(rois, labels)):
                crop = crop_image(shelf_img, *roi)
                m = image_metrics(crop)
                sc = pack_scores_from_metrics(m)

                emotion = 60.0
                choice = pack_3sec_choice_score(
                    sc["pack_legibility_score"],
                    sc["pack_shelf_pop_score"],
                    sc["pack_clarity_score"],
                    emotion
                )
                rows2.append({"pack": lab, "choice_3s": round(choice,1)})

                crops_show[idx].image(crop, caption=f"{lab} ({choice:.1f})", use_container_width=True)

            rank2 = shelf_rank_from_pack_scores(rows2)
            st.dataframe(rank2, use_container_width=True)

            # save to session + learning log
            st.session_state.last_shelf = {
                "mode": "shelf_photo",
                "labels": labels,
                "rank": rank2.to_dict(orient="records"),
            }

            st.session_state.learning_log.append({
                "timestamp": pd.Timestamp.now().isoformat(),
                "type": "shelf_3s",
                "labels": labels,
                "rank": rank2.to_dict(orient="records"),
            })

            st.success("Guardado en learning log.")

    # ----------------------------
    # Learning log download
    # ----------------------------
    st.divider()
    st.markdown("## 📥 Learning log (descargable)")
    if st.session_state.learning_log:
        log_df = pd.DataFrame(st.session_state.learning_log)
        st.dataframe(log_df.tail(50), use_container_width=True)

        st.download_button(
            "Descargar learning_log.csv",
            df_to_csv_bytes(log_df),
            file_name="learning_log.csv",
            mime="text/csv",
            key="dl_learning_log"
        )
    else:
        st.info("Aún no hay learning log. Corre Shelf 3s (packs o anaquel) para registrar resultados.")

# ============================================================
# BLOQUE 6 — Producto Nuevo + What-if + Inversionista + Market + Reporte + Datos + Diagnóstico
# ============================================================

def coldstart_recommendations(success_model, sales_model, base_row: dict, n_keep: int = 12):
    """
    Explora escenarios simples (precio/margen + proxies pack/claims) y recomienda quick wins.
    No requiere histórico del producto (cold start).
    """
    base_df = pd.DataFrame([base_row])
    base_prob = float(success_model.predict_proba(base_df)[0][1])
    base_sales = float(sales_model.predict(base_df)[0])

    # rangos
    precio0 = float(base_row["precio"])
    margen0 = float(base_row["margen_pct"])
    conn0 = float(base_row["conexion_score"])

    price_grid = [precio0*0.9, precio0*0.95, precio0, precio0*1.05, precio0*1.10]
    margin_grid = [max(0,margen0-8), max(0,margen0-4), margen0, min(90,margen0+4), min(90,margen0+8)]
    claims_proxy = [-8, -4, 0, 4, 8]   # proxy delta conexión por claims
    pack_proxy = [-10, -5, 0, 5, 10]   # proxy delta conexión por pack

    out = []
    for pr in price_grid:
        for mg in margin_grid:
            for dc in claims_proxy:
                for dp in pack_proxy:
                    row = dict(base_row)
                    row["precio"] = float(pr)
                    row["margen_pct"] = float(mg)
                    row["conexion_score"] = float(np.clip(conn0 + 0.55*dp + 0.45*dc, 0, 100))
                    df1 = pd.DataFrame([row])
                    prob = float(success_model.predict_proba(df1)[0][1])
                    sales = float(sales_model.predict(df1)[0])

                    out.append({
                        **row,
                        "prob_exito": prob,
                        "ventas_unidades": sales,
                        "uplift_prob_pp": (prob - base_prob)*100,
                        "uplift_sales": (sales - base_sales),
                        "delta_claims_proxy": dc,
                        "delta_pack_proxy": dp,
                    })

    out_df = pd.DataFrame(out).sort_values(["prob_exito","ventas_unidades"], ascending=False).head(n_keep).copy()

    best_prob = float(out_df.iloc[0]["prob_exito"])
    best_sales = float(out_df.iloc[0]["ventas_unidades"])

    recs = []
    top = out_df.iloc[0]
    if top["precio"] < precio0:
        recs.append("Bajar ligeramente precio (5–10%) mejora probabilidad de elección en anaquel.")
    if top["margen_pct"] > margen0:
        recs.append("Subir margen sin disparar precio aumenta utilidad y no necesariamente baja ventas si el pack/claims compensan.")
    if top["delta_pack_proxy"] > 0:
        recs.append("Mejora pack (jerarquía, contraste, simplificación) para subir atención + claridad en 3 segundos.")
    if top["delta_claims_proxy"] > 0:
        recs.append("Optimiza claims: reduce a 2–3 y alinea con segmento/canal (fit/kids/premium/value).")

    summary = {
        "base_prob_%": base_prob*100,
        "best_prob_%": best_prob*100,
        "uplift_prob_pp": (best_prob - base_prob)*100,
        "base_sales": base_sales,
        "best_sales": best_sales,
        "uplift_sales": best_sales - base_sales
    }

    return out_df, recs, summary
# ============================================================
# 🧊 PRODUCTO NUEVO — Cold Start + What-If Recomendaciones (FIX)
# ============================================================
with tab_new:
    st.subheader("🧊 Producto Nuevo — Cold Start (sin histórico)")
    st.caption("Predice éxito y ventas con atributos + proxies (claims/pack) aunque la marca no exista en el dataset.")

    # -----------------------------
    # Init states (persistencia)
    # -----------------------------
    if "last_new" not in st.session_state:
        st.session_state.last_new = None

    if "new_recos_out" not in st.session_state:
        st.session_state.new_recos_out = None
    if "new_recos_txt" not in st.session_state:
        st.session_state.new_recos_txt = None
    if "new_recos_sum" not in st.session_state:
        st.session_state.new_recos_sum = None

    # -----------------------------
    # Inputs base
    # -----------------------------
    c1, c2, c3 = st.columns(3)
    categorias = sorted(df["categoria"].unique().tolist()) if "categoria" in df.columns else ["cereales"]
    canales = sorted(df["canal"].unique().tolist())

    categoria = c1.selectbox("Categoría comparable", categorias, key="new_cat_fix")
    canal = c2.selectbox("Canal", canales, key="new_canal_fix")
    segmento = c3.selectbox("Segmento objetivo", ["fit", "kids", "premium", "value"], key="new_seg_fix")

    canal_norm = str(canal).lower().strip()

    b1, b2, b3, b4, b5 = st.columns(5)
    precio = b1.number_input("Precio", 1.0, 99999.0, float(df["precio"].median()), step=1.0, key="new_precio_fix")
    margen = b2.slider("Margen %", 0, 90, int(np.clip(df["margen_pct"].median(), 0, 90)), key="new_margen_fix")
    competencia = b3.slider("Competencia (1-10)", 1, 10, int(df["competencia"].median()), key="new_comp_fix")
    demanda = b4.slider("Demanda (10-100)", 10, 100, int(df["demanda"].median()), key="new_dem_fix")
    tendencia = b5.slider("Tendencia (20-100)", 20, 100, int(df["tendencia"].median()), key="new_tend_fix")

    # -----------------------------
    # Claims (Cold Start)
    # -----------------------------
    st.markdown("### 🏷️ Claims (Cold Start)")
    try:
        recs = recommend_claims(segmento, canal_norm, 10)
        claim_opts = [c for c, _ in recs] if recs else []
    except Exception:
        claim_opts = []

    if not claim_opts:
        st.warning("No hay claims recomendados (revisa que recommend_claims exista y regrese lista).")
        claims_sel = []
    else:
        claims_sel = st.multiselect(
            "Selecciona 2-3 claims",
            claim_opts,
            default=claim_opts[:2],
            key="new_claims_fix"
        )

    # claims_score debe existir
    try:
        cscore = float(claims_score(claims_sel, canal_norm))
    except Exception:
        cscore = 0.0

    st.metric("Claims Score", f"{cscore:.1f}/100")

    # -----------------------------
    # Pack (opcional)
    # -----------------------------
    st.markdown("### 📦 Empaque (opcional, recomendado)")
    img = st.file_uploader("Sube empaque (PNG/JPG)", type=["png", "jpg", "jpeg"], key="new_pack_fix")

    pack_choice = 60.0   # proxy si no hay imagen
    pack_emotion = 60.0  # proxy si no hay imagen
    pack_label = "neutral"
    pack_quickwins = []

    if img is not None:
        try:
            im = Image.open(img)
            st.image(im, caption="Empaque cargado", use_container_width=True)

            m = image_metrics(im)
            sc = pack_scores_from_metrics(m)

            # Si tienes funciones avanzadas, úsala. Si no, usa proxies estables.
            # (Esto evita que truene si aún no integras shelf/emotion full)
            # pack_choice proxy: combina pop + claridad
            pack_choice = float(clip(0.6 * sc["pack_shelf_pop_score"] + 0.4 * sc["pack_clarity_score"], 0, 100))
            # emoción proxy: combina pop + legibilidad + claims
            pack_emotion = float(clip(0.5 * sc["pack_shelf_pop_score"] + 0.3 * sc["pack_legibility_score"] + 0.2 * cscore, 0, 100))

            # etiqueta simple por buckets
            if pack_emotion >= 75:
                pack_label = "exciting"
            elif pack_emotion >= 60:
                pack_label = "positive"
            elif pack_emotion >= 45:
                pack_label = "neutral"
            else:
                pack_label = "confusing"

            # Quick wins básicos
            if sc["pack_legibility_score"] < 60:
                pack_quickwins.append("Sube legibilidad: más contraste texto/fondo y tipografía más gruesa.")
            if sc["pack_clarity_score"] < 60:
                pack_quickwins.append("Reduce ruido: menos elementos y 2–3 claims máximo.")
            if sc["pack_shelf_pop_score"] < 60:
                pack_quickwins.append("Aumenta shelf pop: color acento + jerarquía clara (Marca→beneficio→variedad).")
            if not pack_quickwins:
                pack_quickwins.append("Visualmente va bien: afina jerarquía y consistencia de beneficios.")

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Elección (3s) proxy", f"{pack_choice:.1f}/100")
            k2.metric("Emoción dominante", pack_label.upper())
            k3.metric("Legibilidad", f"{sc['pack_legibility_score']}/100")
            k4.metric("Claridad", f"{sc['pack_clarity_score']}/100")

            st.markdown("**Quick wins (pack)**")
            for w in pack_quickwins:
                st.write("•", w)

        except Exception as e:
            st.warning(f"No pude procesar la imagen del pack: {e}")

    # -----------------------------
    # Conexión proxy (Cold Start)
    # -----------------------------
    # Combina: demanda + pack_choice + claims_score (0-100)
    conexion_score = float(clip(0.45 * float(demanda) + 0.35 * float(pack_choice) + 0.20 * float(cscore), 0, 100))

    # -----------------------------
    # Entrada al modelo (marca nueva)
    # -----------------------------
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
        "canal": canal_norm,
    }])

    # -----------------------------
    # Predicciones base
    # -----------------------------
    prob = float(success_model.predict_proba(entrada)[0][1])
    ventas_point = float(sales_model.predict(entrada)[0])

    # -----------------------------
    # Comparables: rango p25/p50/p75
    # -----------------------------
    comp = df.copy()
    if "categoria" in df.columns:
        comp = df[df["categoria"] == str(categoria).lower()].copy()
        if comp.empty:
            comp = df.copy()

    # Distancia simple para top comparables
    comp["dist"] = (
        (comp["precio"] - float(precio)).abs() / max(float(precio), 1e-6) +
        (comp["margen_pct"] - float(margen)).abs() / 100.0 +
        (comp["demanda"] - float(demanda)).abs() / 100.0 +
        (comp["tendencia"] - float(tendencia)).abs() / 100.0
    )
    top = comp.sort_values("dist").head(20)

    p25 = float(np.percentile(top["ventas_unidades"], 25))
    p50 = float(np.percentile(top["ventas_unidades"], 50))
    p75 = float(np.percentile(top["ventas_unidades"], 75))

    launch_score = float(
        0.45 * (prob * 100.0) +
        0.25 * float(pack_choice) +
        0.15 * float(cscore) +
        0.15 * float(pack_emotion)
    )

    # -----------------------------
    # Output base
    # -----------------------------
    st.markdown("## 🎯 Resultado (Producto Nuevo)")
    o1, o2, o3, o4 = st.columns(4)
    o1.metric("Prob. éxito", f"{prob*100:.1f}%")
    o2.metric("Ventas (punto)", f"{ventas_point:,.0f} u.")
    o3.metric("Rango comparables (p25–p75)", f"{p25:,.0f} — {p75:,.0f} u.")
    o4.metric("Launch Score", f"{launch_score:.1f}/100")

    if launch_score >= 75:
        st.success("✅ GO — Alto potencial")
    elif launch_score >= 60:
        st.warning("🟡 AJUSTAR — Optimiza pack/claims/precio para subir score")
    else:
        st.error("🔴 NO-GO — Riesgo alto (necesita rediseño o cambiar estrategia)")

    st.markdown("### 🔍 Top comparables usados (20)")
    show_cols = [c for c in ["marca","precio","margen_pct","demanda","tendencia","ventas_unidades","exito"] if c in top.columns]
    st.dataframe(top[show_cols].copy(), use_container_width=True)

    # Guardar escenario base (para reporte / otros tabs)
    st.session_state.last_new = {
        "categoria": categoria,
        "canal": canal_norm,
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
        "pack_emotion_label": str(pack_label),
        "conexion_score_proxy": float(conexion_score),
        "prob_exito": float(prob),
        "ventas_point": float(ventas_point),
        "ventas_p25": float(p25),
        "ventas_p50": float(p50),
        "ventas_p75": float(p75),
        "launch_score": float(launch_score),
    }
# ============================================================
# ✅ WHAT-IF RECOMENDACIONES (DEBUG + PERSISTENTE)
# ============================================================
st.divider()
st.markdown("## 🧠 Recomendaciones What-If (Producto Nuevo)")
st.caption("Si no aparece nada, aquí te muestro el motivo (debug visible).")

# states
if "new_recos_out" not in st.session_state:
    st.session_state.new_recos_out = None
if "new_recos_txt" not in st.session_state:
    st.session_state.new_recos_txt = None
if "new_recos_sum" not in st.session_state:
    st.session_state.new_recos_sum = None
if "new_recos_clicked" not in st.session_state:
    st.session_state.new_recos_clicked = 0
if "new_recos_error" not in st.session_state:
    st.session_state.new_recos_error = ""

# base_row SIEMPRE armado con floats
base_row = {
    "precio": float(precio),
    "competencia": float(competencia),
    "demanda": float(demanda),
    "tendencia": float(tendencia),
    "margen_pct": float(margen),
    "conexion_score": float(conexion_score),
    "rating_conexion": 7.0,
    "sentiment_score": 1.0,
    "marca": "nueva",
    "canal": str(canal).lower().strip(),
}

# Debug: confirma que el bloque existe y que el modelo existe
st.write("🔎 **Debug base_row**:", base_row)
st.write("🔎 **Debug modelos**:",
         {"success_model": type(success_model).__name__, "sales_model": type(sales_model).__name__})
st.write("🔎 **Debug función coldstart_recommendations existe?**:",
         "coldstart_recommendations" in globals())

# Botón con contador (para confirmar que se está clickeando)
clicked = st.button("🚀 Generar recomendaciones (what-if)", key="btn_new_recos_debug")

if clicked:
    st.session_state.new_recos_clicked += 1
    st.session_state.new_recos_error = ""
    st.info(f"✅ Click detectado #{st.session_state.new_recos_clicked}. Generando escenarios...")

    # Validación dura: si no existe la función, te lo dice
    if "coldstart_recommendations" not in globals():
        st.session_state.new_recos_error = "❌ No existe la función coldstart_recommendations() en el archivo. Pégala en el BLOQUE 2."
    else:
        try:
            out_df, recs_txt, summary = coldstart_recommendations(success_model, sales_model, base_row)

            st.session_state.new_recos_out = out_df
            st.session_state.new_recos_txt = recs_txt
            st.session_state.new_recos_sum = summary

            st.success("✅ Recomendaciones generadas y guardadas en session_state.")

        except Exception as e:
            st.session_state.new_recos_out = None
            st.session_state.new_recos_txt = None
            st.session_state.new_recos_sum = None
            st.session_state.new_recos_error = f"❌ Error ejecutando coldstart_recommendations: {repr(e)}"

# Mostrar error si lo hay
if st.session_state.new_recos_error:
    st.error(st.session_state.new_recos_error)

# Mostrar resultados si existen
if st.session_state.new_recos_out is not None:
    out_df = st.session_state.new_recos_out
    recs_txt = st.session_state.new_recos_txt or []
    summary = st.session_state.new_recos_sum or {}

    st.markdown("### 📌 Resumen")
    c1, c2, c3 = st.columns(3)
    c1.metric("Prob base", f"{summary.get('base_prob_%', 0):.1f}%")
    c2.metric("Mejor prob", f"{summary.get('best_prob_%', 0):.1f}%")
    c3.metric("Uplift", f"+{summary.get('uplift_prob_pp', 0):.1f} pp")

    d1, d2, d3 = st.columns(3)
    d1.metric("Ventas base", f"{summary.get('base_sales', 0):,.0f} u.")
    d2.metric("Mejor ventas", f"{summary.get('best_sales', 0):,.0f} u.")
    d3.metric("Uplift ventas", f"+{summary.get('uplift_sales', 0):,.0f} u.")

    st.markdown("### ✅ Quick wins")
    if not recs_txt:
        st.write("• (Sin quick wins) — revisa escenarios o función.")
    else:
        for r in recs_txt:
            st.write("•", r)

    st.markdown("### 🧪 Escenarios")
    if isinstance(out_df, pd.DataFrame) and not out_df.empty:
        show = out_df.copy()
        if "prob_exito" in show.columns:
            show["prob_exito_%"] = (show["prob_exito"] * 100).round(1)
        if "ventas_unidades" in show.columns:
            show["ventas_unidades"] = show["ventas_unidades"].round(0).astype(int)
        st.dataframe(show, use_container_width=True)
    else:
        st.warning("out_df no es DataFrame o viene vacío. La función está devolviendo vacío.")
else:
    st.info("Aún no hay recomendaciones guardadas. Presiona el botón.")

# ============================================================
# 💼 INVERSIONISTA
# ============================================================
with tab_invest:
    st.subheader("💼 Vista Inversionista (TAM + escenarios + unit economics + launch score)")

    st.caption("Modo narrativo/financiero para pitch: TAM, escenarios y upside.")

    a1,a2,a3 = st.columns(3)
    tam = a1.number_input("TAM anual (MXN)", value=5_000_000_000.0, step=100_000_000.0, key="inv_tam")
    som = a2.slider("SOM % (capturable)", 0.0, 10.0, 1.0, step=0.1, key="inv_som")
    share = a3.slider("Share objetivo %", 0.0, 5.0, 0.3, step=0.1, key="inv_share")

    b1,b2,b3 = st.columns(3)
    asp = b1.number_input("ASP ($/unidad)", value=55.0, key="inv_asp")
    cogs = b2.number_input("COGS ($/unidad)", value=32.0, key="inv_cogs")
    mkt = b3.number_input("Marketing mensual (MXN)", value=500_000.0, key="inv_mkt")

    gross = (asp - cogs)
    gross_pct = gross / max(asp, 1e-9)

    st.metric("Gross margin / unidad", f"${gross:.1f} ({gross_pct*100:.1f}%)")

    # usa últimos resultados si existen
    base_prob = None
    if st.session_state.last_new:
        base_prob = st.session_state.last_new.get("prob_exito", None)
    elif st.session_state.last_sim:
        base_prob = st.session_state.last_sim.get("prob_exito", None)

    if base_prob is None:
        st.info("Corre primero Simulador o Producto Nuevo para alimentar Launch Score.")
        base_prob = 0.55

    launch_score = (base_prob*100)

    # escenarios
    st.markdown("### Escenarios")
    s1,s2,s3 = st.columns(3)
    units_low = s1.number_input("Unidades / mes (Low)", value=50_000.0, key="inv_u_low")
    units_mid = s2.number_input("Unidades / mes (Mid)", value=120_000.0, key="inv_u_mid")
    units_high = s3.number_input("Unidades / mes (High)", value=250_000.0, key="inv_u_high")

    def scenario(units):
        rev = units * asp
        gp = units * gross
        op = gp - mkt
        return rev, gp, op

    rows = []
    for name, u in [("Low", units_low), ("Mid", units_mid), ("High", units_high)]:
        rev,gp,op = scenario(u)
        rows.append({"escenario": name, "unidades_mes": u, "ingresos_mes": rev, "gross_profit_mes": gp, "operating_profit_mes": op})

    inv_df = pd.DataFrame(rows)
    st.dataframe(inv_df.style.format({
        "unidades_mes":"{:.0f}",
        "ingresos_mes":"${:,.0f}",
        "gross_profit_mes":"${:,.0f}",
        "operating_profit_mes":"${:,.0f}",
    }), use_container_width=True)

    st.metric("Launch Score (base)", f"{launch_score:.1f}/100")

    st.session_state.last_invest = {
        "tam": float(tam), "som_pct": float(som), "share_pct": float(share),
        "asp": float(asp), "cogs": float(cogs), "gross": float(gross), "gross_pct": float(gross_pct),
        "mkt_mensual": float(mkt),
        "escenarios": rows,
        "launch_score_base": float(launch_score),
    }

# ============================================================
# 📈 MARKET INTELLIGENCE
# ============================================================
with tab_market:
    st.subheader("📈 Market Intelligence (tipo Atlantia-lite)")
    st.caption("Carga research/market_intel.csv o integra fuentes más adelante (APIs/scraping).")

    if market_df is None:
        st.info("No hay market_intel.csv cargado. Sube uno desde la barra lateral.")
    else:
        st.dataframe(market_df.head(300), use_container_width=True)

        # insights básicos si hay columnas
        cols = set(market_df.columns)
        if {"categoria","marca"}.issubset(cols):
            st.markdown("### Top marcas por categoría (conteo)")
            topm = market_df.groupby(["categoria","marca"]).size().reset_index(name="menciones").sort_values("menciones", ascending=False)
            st.dataframe(topm.head(50), use_container_width=True)

# ============================================================
# 📄 REPORTE EJECUTIVO
# ============================================================
with tab_report:
    st.subheader("📄 Reporte Ejecutivo (TXT + CSV inputs)")

    def build_report_txt():
        lines = []
        lines.append("PRODUCT LAB IA — REPORTE EJECUTIVO")
        lines.append("="*45)

        if st.session_state.last_sim:
            s = st.session_state.last_sim
            lines.append("\n[SIMULADOR]")
            lines.append(f"Marca/Canal/Segmento: {s['marca']} / {s['canal']} / {s['segmento']}")
            lines.append(f"Precio: {s['precio']:.2f} | Margen%: {s['margen_pct']:.1f}")
            lines.append(f"Prob éxito: {s['prob_exito']*100:.1f}% | Ventas: {s['ventas_unidades']:.0f} u.")
            lines.append(f"Ingresos: ${s['ingresos']:.0f} | Utilidad: ${s['utilidad']:.0f}")
            lines.append(f"Claims: {', '.join(s['claims'])} (score {s['claims_score']:.1f}/100)")
            lines.append(f"Pack: leg {s['pack_leg']:.0f} pop {s['pack_pop']:.0f} cla {s['pack_cla']:.0f} emotion {s['pack_emotion']:.1f}")

        if st.session_state.last_new:
            n = st.session_state.last_new
            lines.append("\n[PRODUCTO NUEVO]")
            lines.append(f"Categoría/Canal/Segmento: {n['categoria']} / {n['canal']} / {n['segmento']}")
            lines.append(f"Precio: {n['precio']:.2f} | Margen%: {n['margen_pct']:.1f}")
            lines.append(f"Prob éxito: {n['prob_exito']*100:.1f}% | Ventas punto: {n['ventas_point']:.0f} u.")
            lines.append(f"Comparables p25-p75: {n['ventas_p25']:.0f} — {n['ventas_p75']:.0f} u.")
            lines.append(f"Launch Score: {n['launch_score']:.1f}/100")
            lines.append(f"Claims: {', '.join(n['claims'])} (score {n['claims_score']:.1f}/100)")

        if st.session_state.last_shelf:
            sh = st.session_state.last_shelf
            lines.append("\n[SHELF 3-SECOND]")
            lines.append(f"Modo: {sh.get('mode','-')}")
            rank = sh.get("rank", [])
            if rank:
                lines.append("Ranking (MNL prob %):")
                for r in rank:
                    lines.append(f" - {r.get('pack','')}: {r.get('mnl_prob_%','')}% (choice {r.get('choice_3s','')})")

        if st.session_state.last_invest:
            inv = st.session_state.last_invest
            lines.append("\n[INVERSIONISTA]")
            lines.append(f"TAM: ${inv['tam']:.0f} | SOM%: {inv['som_pct']:.2f} | Share%: {inv['share_pct']:.2f}")
            lines.append(f"ASP: {inv['asp']:.2f} | COGS: {inv['cogs']:.2f} | GM%: {inv['gross_pct']*100:.1f}")
            lines.append(f"Launch Score base: {inv['launch_score_base']:.1f}/100")

        return "\n".join(lines)

    report_txt = build_report_txt()
    st.download_button("⬇️ Descargar Reporte TXT", report_txt, file_name="reporte_ejecutivo.txt", key="dl_report_txt")

    # CSV inputs (últimos)
    rows = []
    if st.session_state.last_sim:
        rows.append({"tipo":"simulador", **st.session_state.last_sim})
    if st.session_state.last_new:
        rows.append({"tipo":"producto_nuevo", **st.session_state.last_new})
    if st.session_state.last_invest:
        rows.append({"tipo":"inversionista", **st.session_state.last_invest})

    if rows:
        inputs_df = pd.json_normalize(rows)
        st.download_button("⬇️ Descargar Inputs CSV", df_to_csv_bytes(inputs_df), file_name="inputs_reporte.csv", key="dl_report_csv")
        st.dataframe(inputs_df.head(50), use_container_width=True)
    else:
        st.info("Corre Simulador / Producto Nuevo / Inversionista para generar inputs descargables.")

# ============================================================
# 📂 DATOS
# ============================================================
with tab_data:
    st.subheader("📂 Datos (preview + descarga)")

    st.download_button(
        label="📥 Descargar dataset actual",
        data=df_to_csv_bytes(df),
        file_name="dataset_actual.csv",
        mime="text/csv",
        key="dl_dataset_actual"
    )
    st.dataframe(df.head(300), use_container_width=True)

# ============================================================
# 🧠 DIAGNÓSTICO
# ============================================================
with tab_diag:
    st.subheader("🧠 Diagnóstico de modelo")

    st.markdown("### Matriz de confusión (éxito)")
    st.dataframe(pd.DataFrame(CM, index=["Real 0","Real 1"], columns=["Pred 0","Pred 1"]), use_container_width=True)

    st.markdown("### Métricas")
    st.write(f"Precisión: **{ACC*100:.2f}%**")
    st.write(f"AUC: **{AUC:.3f}**")
    st.write(f"MAE ventas: **{MAE:,.0f}** unidades")

