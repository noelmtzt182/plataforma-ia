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
# BLOQUE 1 — CORE (Completo)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps

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
    page_title="Plataforma IA | Producto + Empaque + Claims + Shelf + Market",
    layout="wide"
)

DATA_PATH_DEFAULT = "mercado_cereales_5000_con_ventas.csv"
MARKET_PATH_DEFAULT = "market_intel.csv"


# ----------------------------
# Helpers base
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
    Convierte value_counts() a DF estable para st.bar_chart
    (evita SchemaValidationError / Altair issues).
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

def now_ts_str():
    # timestamp simple (sin pytz)
    return pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


# ----------------------------
# Columnas requeridas
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
# Loaders
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

    # opcionales
    for c in ["share_proxy", "rating_promedio", "sentiment_promedio"]:
        if c in mdf.columns:
            mdf[c] = pd.to_numeric(mdf[c], errors="coerce")

    mdf = mdf.dropna(subset=["categoria","marca","canal","precio","competencia_skus","demanda_idx","tendencia_idx"])
    return mdf


# ============================================================
# Claims Engine (ROBUSTO)
# ============================================================
CLAIMS_LIBRARY = {
    "fit": [
        ("Alto en proteína", 0.90),
        ("Sin azúcar añadida", 0.88),
        ("Alto en fibra", 0.86),
        ("Integral", 0.80),
        ("Sin colorantes artificiales", 0.72),
        ("Sin jarabe de maíz", 0.68),
        ("Con ingredientes reales", 0.66),
    ],
    "kids": [
        ("Con vitaminas y minerales", 0.86),
        ("Sabor chocolate", 0.82),
        ("Energía para su día", 0.78),
        ("Hecho con granos", 0.74),
        ("Sin conservadores", 0.70),
        ("Diversión en cada bocado", 0.68),
    ],
    "premium": [
        ("Ingredientes seleccionados", 0.82),
        ("Hecho con avena real", 0.76),
        ("Calidad premium", 0.70),
        ("Receta artesanal", 0.64),
        ("Sabor intenso", 0.62),
    ],
    "value": [
        ("Rinde más", 0.78),
        ("Ideal para la familia", 0.72),
        ("Gran sabor a mejor precio", 0.74),
        ("Económico y práctico", 0.66),
        ("Tamaño familiar", 0.68),
    ],
}

CANAL_CLAIM_BOOST = {
    "retail": {
        "Sin azúcar añadida": 1.04,
        "Alto en fibra": 1.03,
        "Integral": 1.02,
        "Tamaño familiar": 1.02,
    },
    "marketplace": {
        "Alto en proteína": 1.05,
        "Sin colorantes artificiales": 1.04,
        "Ingredientes seleccionados": 1.03,
        "Con ingredientes reales": 1.03,
    },
}

def _normalize_claim_items(items):
    """
    Acepta:
      - [("claim", 0.8), ...]
      - ["claim", ...]
      - [{"claim":"...", "score":0.8}, ...]
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

def recommend_claims(segment: str, canal: str, max_claims: int = 8):
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

    # penaliza demasiados claims
    n = len(selected_claims)
    clarity_penalty = 1.0 if n <= 3 else max(0.65, 1.0 - 0.12 * (n - 3))

    score = 75.0 * base * clarity_penalty
    return float(np.clip(score, 0, 100))


# ============================================================
# Pack Vision+ (sin libs pesadas)
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

    # edge-ish (gradiente simple)
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
        "edge_map": mag,  # para heatmap ligero
    }

def pack_scores_from_metrics(m: dict) -> dict:
    # legibilidad ~ contraste y ruido moderado
    legibility = 0.70 * clip(m["contrast"] / 0.22, 0, 1) + 0.30 * (1 - clip(abs(m["edge_density"] - 0.18) / 0.18, 0, 1))
    legibility = clip(legibility, 0, 1) * 100

    # shelf pop ~ pop_score y brillo objetivo
    target_brightness = 0.55
    brightness_fit = 1 - abs(m["brightness"] - target_brightness) / target_brightness
    shelf_pop = clip(0.75 * m["pop_score"] + 0.25 * clip(brightness_fit, 0, 1), 0, 1) * 100

    clarity = clip(0.60 * clip(m["contrast"] / 0.22, 0, 1) + 0.40 * (1 - clip(m["edge_density"] / 0.35, 0, 1)), 0, 1) * 100

    return {
        "pack_legibility_score": round(float(legibility), 1),
        "pack_shelf_pop_score": round(float(shelf_pop), 1),
        "pack_clarity_score": round(float(clarity), 1),
    }

def pack_heatmap_image(m: dict, max_size=700) -> Image.Image:
    """
    Heatmap ligero: normaliza edge_map y lo convierte a imagen en escala.
    (sin matplotlib/altair)
    """
    mag = m["edge_map"].copy()
    if mag.size == 0:
        return Image.new("L", (256, 256), 0)

    mag = mag - mag.min()
    denom = (mag.max() + 1e-6)
    mag = (mag / denom) * 255.0
    hm = Image.fromarray(mag.astype(np.uint8), mode="L")

    # resize
    w, h = hm.size
    scale = min(max_size / max(w, h), 1.0)
    if scale < 1.0:
        hm = hm.resize((int(w * scale), int(h * scale)))
    return hm

def pack_3sec_choice_score(pack_legibility, pack_pop, pack_clarity, claims_score_val):
    """
    Score 0-100 (elección 3s) proxy:
    - pop pesa más, claridad y legibilidad, claims ayuda.
    """
    score = (
        0.45 * (pack_pop / 100.0) +
        0.25 * (pack_clarity / 100.0) +
        0.20 * (pack_legibility / 100.0) +
        0.10 * (claims_score_val / 100.0)
    ) * 100.0
    return float(np.clip(score, 0, 100))

def pack_quick_wins(scores: dict, m: dict):
    out = []
    if scores["pack_legibility_score"] < 60:
        out.append("Sube legibilidad: más contraste texto/fondo y tipografía más gruesa en beneficio principal.")
    if scores["pack_clarity_score"] < 60:
        out.append("Reduce ruido: limita a 2–3 claims y deja más aire alrededor del título/beneficio.")
    if scores["pack_shelf_pop_score"] < 60:
        out.append("Mejora shelf-pop: agrega un color acento y evita packs muy lavados u oscuros.")
    if m["edge_density"] > 0.28:
        out.append("Saturación visual alta: simplifica fondos/patrones y reduce microtextos.")
    if not out:
        out.append("Vas bien: optimiza jerarquía (Marca → beneficio → variedad → credencial).")
    return out


# ============================================================
# Conexión / emoción (proxy)
# ============================================================
def pack_emotion_score(pack_legibility: float, pack_pop: float, pack_clarity: float, claims_score_val: float, copy_tone: int):
    visual = 0.40 * (pack_pop / 100.0) + 0.30 * (pack_clarity / 100.0) + 0.15 * (pack_legibility / 100.0)
    claims = 0.15 * (claims_score_val / 100.0)
    tone_boost = 0.06 if copy_tone > 0 else (-0.06 if copy_tone < 0 else 0.0)
    score = (visual + claims + tone_boost) * 100.0
    return float(np.clip(score, 0, 100))


# ============================================================
# Shelf & Emotion (3s) — MNL + recortes manuales
# ============================================================
def crop_image(img: Image.Image, x, y, w, h) -> Image.Image:
    W, H = img.size
    x = int(clip(x, 0, W-1))
    y = int(clip(y, 0, H-1))
    w = int(clip(w, 1, W-x))
    h = int(clip(h, 1, H-y))
    return img.crop((x, y, x+w, y+h))

def mnl_prob(utilities: np.ndarray) -> np.ndarray:
    """
    Multinomial logit (softmax) estable.
    """
    u = utilities.astype(float)
    u = u - np.max(u)
    e = np.exp(u)
    return e / np.sum(e)

def shelf_rank_from_pack_scores(pack_rows: list) -> pd.DataFrame:
    """
    pack_rows: list of dicts con keys: name, choice_3s (0-100), emotion (0-100)
    utility simple: 0.7*choice + 0.3*emotion
    """
    dfp = pd.DataFrame(pack_rows).copy()
    if dfp.empty:
        return dfp
    dfp["utility"] = 0.70 * (dfp["choice_3s"]/100.0) + 0.30 * (dfp["emotion"]/100.0)
    probs = mnl_prob(dfp["utility"].values)
    dfp["mnl_prob"] = probs
    dfp = dfp.sort_values("mnl_prob", ascending=False).reset_index(drop=True)
    return dfp


# ============================================================
# Producto Nuevo — Recomendaciones What-If (coldstart)
# ============================================================
def coldstart_recommendations(success_model, sales_model, base_row: dict, grid_size: int = 24):
    """
    Grid search ligero:
    - precio: -10, -5, 0, +5, +10%
    - margen: -5, 0, +5 pp
    - demanda: 0, +5
    - tendencia: 0, +5
    - conexion_score: +0, +5, +10 (proxy de pack/claims)
    """
    base = base_row.copy()

    base_df = pd.DataFrame([base])
    base_prob = float(success_model.predict_proba(base_df)[0][1])
    base_sales = float(sales_model.predict(base_df)[0])

    price0 = float(base["precio"])
    margin0 = float(base["margen_pct"])
    dem0 = float(base["demanda"])
    tend0 = float(base["tendencia"])
    conn0 = float(base["conexion_score"])

    price_mults = [0.90, 0.95, 1.00, 1.05, 1.10]
    margin_deltas = [-5, 0, +5]
    dem_deltas = [0, +5]
    tend_deltas = [0, +5]
    conn_deltas = [0, +5, +10]

    rows = []
    for pm in price_mults:
        for md in margin_deltas:
            for dd in dem_deltas:
                for td in tend_deltas:
                    for cd in conn_deltas:
                        r = base.copy()
                        r["precio"] = float(price0 * pm)
                        r["margen_pct"] = float(np.clip(margin0 + md, 0, 90))
                        r["demanda"] = float(np.clip(dem0 + dd, 10, 100))
                        r["tendencia"] = float(np.clip(tend0 + td, 20, 100))
                        r["conexion_score"] = float(np.clip(conn0 + cd, 0, 100))

                        df1 = pd.DataFrame([r])
                        prob = float(success_model.predict_proba(df1)[0][1])
                        sales = float(sales_model.predict(df1)[0])
                        rows.append({
                            **r,
                            "prob_exito": prob,
                            "ventas_unidades": max(0.0, sales),
                            "uplift_prob_pp": (prob - base_prob) * 100.0,
                            "uplift_sales": (sales - base_sales),
                            "delta_conn": cd,
                            "delta_margin": md,
                            "delta_price_pct": (pm - 1.0) * 100.0,
                        })

    out = pd.DataFrame(rows).copy()
    out = out.sort_values(["prob_exito", "ventas_unidades"], ascending=False).head(grid_size).reset_index(drop=True)

    # Recos texto
    best = out.iloc[0].to_dict()
    recs_txt = []
    if abs(best["delta_price_pct"]) >= 5:
        direction = "bajar" if best["delta_price_pct"] < 0 else "subir"
        recs_txt.append(f"Ajusta precio: {direction} ~{abs(best['delta_price_pct']):.0f}% (mejoró prob/ventas en escenarios top).")
    if best["delta_margin"] != 0:
        recs_txt.append(f"Ajusta margen: {best['delta_margin']:+.0f} pp (balance unit economics + prob).")
    if best["delta_conn"] > 0:
        recs_txt.append("Sube conexión (pack/claims/copy): +5 a +10 pts mueve fuerte la probabilidad en cold start.")
    if not recs_txt:
        recs_txt.append("Tu configuración ya está cerca del óptimo en este grid. Itera con pack/claims para mover conexión.")

    summary = {
        "base_prob_%": base_prob * 100.0,
        "best_prob_%": float(best["prob_exito"]) * 100.0,
        "uplift_prob_pp": (float(best["prob_exito"]) - base_prob) * 100.0,
        "base_sales": base_sales,
        "best_sales": float(best["ventas_unidades"]),
        "uplift_sales": float(best["ventas_unidades"]) - base_sales,
    }
    return out, recs_txt, summary


# ============================================================
# Modelos (Éxito + Ventas) — sin errores de longitud
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

    # Ventas
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

    reg.fit(Xr_train, yr_train)
    yhat = reg.predict(Xr_test)
    MAE = mean_absolute_error(yr_test, yhat)

    return clf, reg, ACC, AUC, CM, MAE

# ============================================================
# BLOQUE 2 — SIDEBAR + CARGA + ENTRENAMIENTO + ESTADO
# ============================================================

st.sidebar.title("⚙️ Control")

if st.sidebar.button("🔄 Limpiar cache"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.sidebar.success("Cache limpio ✅ (recarga la app)")

st.sidebar.divider()

# Dataset principal
st.sidebar.subheader("📂 Dataset principal (ventas + éxito)")
uploaded_csv = st.sidebar.file_uploader("Sube CSV principal", type=["csv"], key="uploader_main_csv")

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

# Market Intel
st.sidebar.subheader("📈 Market Intelligence (opcional)")
market_file = st.sidebar.file_uploader("Sube market_intel.csv", type=["csv"], key="uploader_market_csv")

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
            st.sidebar.warning("Market Intel no cargado (opcional).")
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

# Estado global para reportes / logs
if "last_sim" not in st.session_state:
    st.session_state.last_sim = None

if "last_pack" not in st.session_state:
    st.session_state.last_pack = None

if "last_shelf" not in st.session_state:
    st.session_state.last_shelf = None

if "last_new" not in st.session_state:
    st.session_state.last_new = None

if "learning_log" not in st.session_state:
    st.session_state.learning_log = []  # lista de dicts


# Header KPIs
st.title("🧠 Plataforma IA: Producto + Empaque + Claims + Shelf + Market")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Registros", f"{len(df):,}")
k2.metric("Precisión", f"{ACC*100:.2f}%")
k3.metric("AUC", f"{AUC:.3f}")
k4.metric("Éxito base", f"{df['exito'].mean()*100:.1f}%")
k5.metric("MAE ventas", f"{MAE:,.0f} u.")

st.divider()

# ============================================================
# BLOQUE 3A — Tabs + Simulador + Insights
# ============================================================

tab_sim, tab_ins, tab_claims, tab_pack, tab_shelf, tab_new, tab_report, tab_market, tab_data, tab_diag = st.tabs([
    "🧪 Simulador",
    "📊 Insights",
    "🏷️ Claims Lab",
    "📦 Pack Vision+",
    "🧲 Shelf & Emotion (3s)",
    "🧊 Producto Nuevo",
    "📄 Reporte Ejecutivo",
    "📈 Market Intelligence",
    "📂 Datos",
    "🧠 Diagnóstico",
])

# ============================================================
# 🧪 SIMULADOR
# ============================================================
with tab_sim:

    st.subheader("🧪 Simulador What-If")

    marcas = sorted(df["marca"].unique().tolist())
    canales = sorted(df["canal"].unique().tolist())

    c1,c2,c3 = st.columns(3)
    marca = c1.selectbox("Marca", marcas, key="sim_marca")
    canal = c2.selectbox("Canal", canales, key="sim_canal")
    segmento = c3.selectbox("Segmento", ["fit","kids","premium","value"], key="sim_segmento")

    canal_norm = str(canal).lower().strip()

    b1,b2,b3,b4,b5 = st.columns(5)
    precio = b1.number_input("Precio", value=float(df["precio"].median()), key="sim_precio")
    competencia = b2.slider("Competencia",1,10,int(df["competencia"].median()), key="sim_competencia")
    demanda = b3.slider("Demanda",10,100,int(df["demanda"].median()), key="sim_demanda")
    tendencia = b4.slider("Tendencia",20,100,int(df["tendencia"].median()), key="sim_tendencia")
    margen_pct = b5.slider("Margen %",0,90,int(df["margen_pct"].median()), key="sim_margen_pct")

    # ---------- Claims ----------
    recs = recommend_claims(segmento, canal_norm, 8)
    claim_opts = [c for c,_ in recs]
    claims = st.multiselect("Claims", claim_opts, claim_opts[:2], key="sim_claims")
    cscore = claims_score(claims, canal_norm)

    # ---------- Pack ----------
    p1,p2,p3 = st.columns(3)
    pack_leg = p1.slider("Pack legibilidad",0,100,65, key="sim_pack_leg")
    pack_pop = p2.slider("Pack pop",0,100,70, key="sim_pack_pop")
    pack_cla = p3.slider("Pack claridad",0,100,65, key="sim_pack_cla")

    pack_emotion = pack_emotion_score(pack_leg, pack_pop, pack_cla, cscore, 0)

    conexion_score = clip(
        0.5*demanda + 0.3*pack_pop + 0.2*cscore,
        0,100
    )

    entrada = pd.DataFrame([{
        "precio":precio,
        "competencia":competencia,
        "demanda":demanda,
        "tendencia":tendencia,
        "margen_pct":margen_pct,
        "conexion_score":conexion_score,
        "rating_conexion":7,
        "sentiment_score":1,
        "marca":marca,
        "canal":canal_norm
    }])

    if st.button("🚀 Simular", key="sim_btn"):

        prob = success_model.predict_proba(entrada)[0][1]
        ventas = float(sales_model.predict(entrada)[0])

        ingresos = ventas * precio
        utilidad = ventas * precio * (margen_pct/100)

        k1,k2,k3 = st.columns(3)
        k1.metric("Prob éxito", f"{prob*100:.1f}%")
        k2.metric("Ventas", f"{ventas:,.0f}")
        k3.metric("Ingresos", f"${ingresos:,.0f}")

        st.metric("Utilidad", f"${utilidad:,.0f}")

        # ---------- ROI ----------
        st.divider()
        st.subheader("🎯 ROI")

        inv = st.number_input("Inversión ($)", 0.0, key="sim_inversion")
        if inv > 0:
            roi = (utilidad - inv) / inv
            st.metric("ROI", f"{roi*100:.1f}%")

# ============================================================
# 📊 INSIGHTS
# ============================================================
with tab_ins:

    st.subheader("📊 Insights")

    st.dataframe(
        df.groupby("marca")[["ventas_unidades"]]
        .mean()
        .sort_values("ventas_unidades",ascending=False),
        use_container_width=True
    )

    bins = pd.cut(df["conexion_score"], [0,20,40,60,80,100])
    vc = bins.value_counts().sort_index()

    st.bar_chart(
        bar_df_from_value_counts(vc).set_index("bucket"),
        use_container_width=True
    )

# ============================================================
# BLOQUE 3B — Claims + Pack Vision + Shelf
# ============================================================

# ============================================================
# 🏷️ CLAIMS LAB
# ============================================================
with tab_claims:

    st.subheader("🏷️ Claims Lab")

    seg = st.selectbox("Segmento", ["fit","kids","premium","value"], key="claims_segmento")
    can = st.selectbox("Canal", ["retail","marketplace"], key="claims_canal")

    recs = recommend_claims(seg, can, 10)

    st.dataframe(
        pd.DataFrame(recs, columns=["claim","score"]),
        use_container_width=True
    )

    sel = st.multiselect("Selecciona claims", [c for c,_ in recs], key="claims_sel")

    st.metric("Claims Score", f"{claims_score(sel, can):.1f}/100")

# ============================================================
# 📦 PACK VISION+
# ============================================================
with tab_pack:

    st.subheader("📦 Pack Vision+")

    img_file = st.file_uploader("Sube pack", type=["png","jpg","jpeg"], key="pack_uploader")

    if img_file:
        img = Image.open(img_file)
        st.image(img, use_container_width=True)

        m = image_metrics(img)
        sc = pack_scores_from_metrics(m)

        st.metric("Legibilidad", sc["pack_legibility_score"])
        st.metric("Pop", sc["pack_shelf_pop_score"])
        st.metric("Claridad", sc["pack_clarity_score"])

        st.image(pack_heatmap_image(m))

# ============================================================
# 🧲 SHELF 3-SECOND
# ============================================================
with tab_shelf:

    st.subheader("🧲 Shelf & Emotion Predictor")

    files = [
        st.file_uploader("Tu pack", type=["png","jpg","jpeg"], key="sp0"),
        st.file_uploader("Comp 1", type=["png","jpg","jpeg"], key="sp1"),
        st.file_uploader("Comp 2", type=["png","jpg","jpeg"], key="sp2"),
    ]

    rows=[]

    for i,f in enumerate(files):
        if f:
            im = Image.open(f)
            m = image_metrics(im)
            sc = pack_scores_from_metrics(m)

            choice = pack_3sec_choice_score(
                sc["pack_legibility_score"],
                sc["pack_shelf_pop_score"],
                sc["pack_clarity_score"],
                60
            )

            rows.append({
                "pack": f"Pack_{i}",
                "choice_3s": choice
            })

    if rows:
        r = shelf_rank_from_pack_scores(rows)
        st.dataframe(r, use_container_width=True)

# ============================================================
# BLOQUE 3C — Producto Nuevo + Market + Reporte + Datos
# ============================================================

# ============================================================
# 🧊 PRODUCTO NUEVO
# ============================================================
with tab_new:

    st.subheader("🧊 Producto Nuevo — Cold Start")

    precio = st.number_input("Precio",100.0, key="new_precio")
    margen = st.slider("Margen",0,90,30, key="new_margen")

    entrada = pd.DataFrame([{
        "precio":precio,
        "competencia":5,
        "demanda":60,
        "tendencia":60,
        "margen_pct":margen,
        "conexion_score":60,
        "rating_conexion":7,
        "sentiment_score":1,
        "marca":"nueva",
        "canal":"retail"
    }])

    p = success_model.predict_proba(entrada)[0][1]
    v = sales_model.predict(entrada)[0]

    st.metric("Prob éxito", f"{p*100:.1f}%")
    st.metric("Ventas", f"{v:,.0f}")

# ============================================================
# 📈 MARKET
# ============================================================
with tab_market:

    st.subheader("📈 Market Intelligence")

    if market_df is not None:
        st.dataframe(market_df.head(100), use_container_width=True)
    else:
        st.info("Carga market_intel.csv")

# ============================================================
# 📄 REPORTE
# ============================================================
with tab_report:

    st.subheader("📄 Reporte Ejecutivo")

    txt = "Reporte generado Product Lab IA"
    st.download_button("Descargar TXT", txt, key="report_txt")

# ============================================================
# 📂 DATOS
# ============================================================
with tab_data:

    st.subheader("📂 Datos")

    st.download_button(
        "Descargar CSV",
        df_to_csv_bytes(df),
        "dataset.csv",
        key="download_dataset"
    )

# ============================================================
# 🧠 DIAGNÓSTICO
# ============================================================
with tab_diag:

    st.subheader("🧠 Diagnóstico Modelo")

    st.dataframe(pd.DataFrame(CM), use_container_width=True)
    st.write("MAE ventas:", MAE)