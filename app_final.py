# app_final.py
# ============================================================
# PRODUCT LAB IA — FINAL (Streamlit Cloud Ready)
# Incluye:
# ✅ Modelos: éxito (clasificación) + ventas (regresión) con RandomForest
# ✅ Claims Lab
# ✅ Pack Vision+ (imagen -> métricas -> heatmap proxy)
# ✅ Shelf 3s (packs sueltos + foto anaquel ROIs)
# ✅ Video anaquel (frame + ROIs + promedio)
# ✅ Producto Nuevo (Cold Start + comparables p25/p50/p75 + launch score + what-if)
# ✅ Market Intelligence (upload market_intel.csv + recomendaciones)
# ✅ Inversionista (TAM/SOM/Share + unit economics + escenarios)
# ✅ Reporte TXT + inputs CSV + learning log
# ✅ PRO mode (Deep Sentiment opcional con fallback automático)
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
    page_title="Product Lab IA — FINAL (Empaque + Shelf + Video + New + Market + Invest)",
    layout="wide"
)

DATA_PATH_DEFAULT = "mercado_cereales_5000_con_ventas.csv"
MARKET_DEFAULT = "market_intel.csv"

REQUIRED_BASE = {
    "marca", "categoria", "canal", "precio", "costo", "margen", "margen_pct",
    "competencia", "demanda", "tendencia", "estacionalidad",
    "rating_conexion", "comentario", "sentiment_score",
    "conexion_score", "conexion_alta", "score_latente", "exito"
}

# ----------------------------
# Helpers
# ----------------------------
def _clean_str_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()

def clip(v, a, b):
    return float(max(a, min(b, v)))

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def softmax(x):
    x = np.array(x, dtype=float)
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-9)

def bar_df_from_value_counts(vc: pd.Series) -> pd.DataFrame:
    out = vc.reset_index()
    out = out.iloc[:, :2].copy()
    out.columns = ["bucket", "count"]
    out["bucket"] = out["bucket"].astype(str)
    out["count"] = pd.to_numeric(out["count"], errors="coerce").fillna(0)
    return out

# ----------------------------
# Session State defaults
# ----------------------------
defaults = {
    "learning_log": [],
    "last_sim": None,
    "last_pack": None,
    "last_shelf": None,
    "last_new": None,
    "last_invest": None,
    "new_recos_out": None,
    "new_recos_txt": None,
    "new_recos_sum": None,
    "new_recos_error": "",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================
# CLAIMS ENGINE
# ============================================================

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

# ============================================================
# PACK VISION+ + KPIs + PRO (fallback)
# ============================================================

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

def pack_heatmap_image_from_edges(img: Image.Image) -> Image.Image:
    im = img.convert("RGB")
    arr = np.asarray(im).astype(np.float32)
    gray = (0.299*arr[...,0] + 0.587*arr[...,1] + 0.114*arr[...,2])

    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:,1:-1] = gray[:,2:] - gray[:,:-2]
    gy[1:-1,:] = gray[2:,:] - gray[:-2,:]
    mag = np.sqrt(gx**2 + gy**2)
    mag = mag / (np.max(mag) + 1e-6)

    overlay = arr.copy()
    overlay[...,0] = np.clip(overlay[...,0] + 120*mag, 0, 255)
    overlay[...,1] = np.clip(overlay[...,1] * (1 - 0.35*mag), 0, 255)
    overlay[...,2] = np.clip(overlay[...,2] * (1 - 0.35*mag), 0, 255)
    return Image.fromarray(overlay.astype(np.uint8))

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

# KPIs solicitados
def kpi_attention_visual(m: dict, sc: dict) -> float:
    return float(np.clip(55*m["edge_density"] + 0.45*sc["pack_shelf_pop_score"], 0, 100))

def kpi_visual_memory(sc: dict, m: dict) -> float:
    return float(np.clip(0.45*sc["pack_shelf_pop_score"] + 0.35*sc["pack_clarity_score"] + 20*m["colorfulness"], 0, 100))

def emotion_proxy(sc: dict, claims_score_val: float) -> float:
    return float(np.clip(0.45*sc["pack_shelf_pop_score"] + 0.25*sc["pack_legibility_score"] + 0.30*claims_score_val, 0, 100))

# PRO mode (Ruta B): fallback
@st.cache_resource
def _try_load_sentiment_pipeline():
    try:
        from transformers import pipeline
        return pipeline("sentiment-analysis")
    except Exception:
        return None

def deep_sentiment_from_text(text: str, enabled: bool) -> float:
    if not enabled:
        return 60.0
    pipe = _try_load_sentiment_pipeline()
    if pipe is None:
        t = str(text).lower()
        pos = sum(w in t for w in ["rico","me encanta","bueno","excelente","delicioso","top","wow","increíble","premium"])
        neg = sum(w in t for w in ["malo","feo","caro","horrible","no me gusta","decepcion","asco"])
        return float(np.clip(60 + 10*pos - 12*neg, 0, 100))
    try:
        res = pipe(str(text)[:512])[0]
        label = str(res.get("label","")).upper()
        conf = float(res.get("score", 0.5))
        if "POS" in label:
            return float(np.clip(50 + 50*conf, 0, 100))
        if "NEG" in label:
            return float(np.clip(50 - 50*conf, 0, 100))
        return 60.0
    except Exception:
        return 60.0

# ============================================================
# SHELF 3s + MNL + ROIs + VIDEO (imageio, sin OpenCV)
# ============================================================

def pack_3sec_choice_score(leg, pop, clarity, emotion):
    s = 0.35*(pop/100) + 0.25*(clarity/100) + 0.20*(leg/100) + 0.20*(emotion/100)
    return float(np.clip(s*100, 0, 100))

def shelf_rank_from_pack_scores(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows).copy()
    df["utility"] = df["choice_3s"] / 10.0
    df["mnl_prob"] = softmax(df["utility"].values)
    df = df.sort_values("mnl_prob", ascending=False).reset_index(drop=True)
    df["mnl_prob_%"] = (df["mnl_prob"] * 100).round(1)
    return df[["pack", "choice_3s", "mnl_prob_%"]]

def crop_image(img: Image.Image, x1, y1, x2, y2) -> Image.Image:
    w, h = img.size
    x1 = int(np.clip(x1, 0, w-1)); x2 = int(np.clip(x2, 1, w))
    y1 = int(np.clip(y1, 0, h-1)); y2 = int(np.clip(y2, 1, h))
    if x2 <= x1 + 5: x2 = min(w, x1 + 50)
    if y2 <= y1 + 5: y2 = min(h, y1 + 50)
    return img.crop((x1, y1, x2, y2))

def draw_rois(img: Image.Image, rois: list[tuple], labels: list[str]) -> Image.Image:
    im = img.convert("RGB").copy()
    draw = ImageDraw.Draw(im)
    for (x1, y1, x2, y2), lab in zip(rois, labels):
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        draw.text((x1 + 4, y1 + 4), lab, fill=(255, 0, 0))
    return im

def _read_video_frames_bytes(video_bytes: bytes, max_frames: int = 180):
    try:
        import imageio.v3 as iio
    except Exception:
        st.error("Falta imageio. Agrega 'imageio' y 'imageio-ffmpeg' a requirements.txt.")
        return [], None

    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    fps = None
    try:
        meta = iio.immeta(tmp_path, plugin="FFMPEG")
        fps = float(meta.get("fps", 0)) if meta else None
    except Exception:
        fps = None

    frames = []
    try:
        idx = 0
        for frame in iio.imiter(tmp_path, plugin="FFMPEG"):
            if idx >= max_frames:
                break
            frames.append(Image.fromarray(frame))
            idx += 1
    except Exception as e:
        st.error(f"No pude leer el video: {e}")
        return [], fps

    return frames, fps

# ============================================================
# DATA LOAD + MARKET LOAD + TRAIN
# ============================================================

@st.cache_data
def load_data(path_or_file) -> pd.DataFrame:
    df = pd.read_csv(path_or_file).copy()

    for c in ["marca","categoria","canal","estacionalidad","comentario"]:
        if c in df.columns:
            df[c] = _clean_str_series(df[c])

    missing = sorted(list(REQUIRED_BASE - set(df.columns)))
    if missing:
        raise ValueError(f"Faltan columnas base en el CSV: {missing}")

    num_cols = [
        "precio","costo","margen","margen_pct",
        "competencia","demanda","tendencia",
        "rating_conexion","sentiment_score",
        "conexion_score","conexion_alta","score_latente","exito"
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ["ventas_unidades","ventas_ingresos","utilidad"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=[
        "marca","categoria","canal","precio",
        "competencia","demanda","tendencia","margen_pct",
        "conexion_score","rating_conexion","sentiment_score","exito"
    ])
    df["exito"] = df["exito"].astype(int)

    if "ventas_unidades" not in df.columns or df["ventas_unidades"].isna().all():
        df["ventas_unidades"] = (
            1200 + 35*df["demanda"] + 18*df["tendencia"] - 6*df["precio"]
            + 20*df["margen_pct"] + 12*df["conexion_score"]
        ).clip(0)

    df["ventas_unidades"] = pd.to_numeric(df["ventas_unidades"], errors="coerce")
    df["ventas_unidades"] = df["ventas_unidades"].fillna(df["ventas_unidades"].median())

    return df

@st.cache_data
def load_market_intel(path_or_file):
    try:
        mdf = pd.read_csv(path_or_file).copy()
        mdf.columns = [str(c).strip().lower() for c in mdf.columns]
        for c in ["fuente","categoria","marca","canal","comentario","claim_top","tendencia_claim_top","tendencia_pack_top"]:
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

    clf = Pipeline(steps=[
        ("preprocessor", pre),
        ("model", RandomForestClassifier(
            n_estimators=300,
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

    reg = Pipeline(steps=[
        ("preprocessor", pre),
        ("model", RandomForestRegressor(n_estimators=300, random_state=42))
    ])
    yv = df["ventas_unidades"].astype(float)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X, yv, test_size=0.2, random_state=42)
    reg.fit(X_train2, y_train2)
    predv = reg.predict(X_test2)
    MAE = mean_absolute_error(y_test2, predv)

    return clf, reg, ACC, AUC, CM, MAE

# ============================================================
# SIDEBAR LOADERS
# ============================================================
st.sidebar.title("⚙️ Config")

uploaded = st.sidebar.file_uploader("Sube tu dataset principal (CSV)", type=["csv"], key="uploader_dataset")
if uploaded is not None:
    df = load_data(uploaded)
else:
    if Path(DATA_PATH_DEFAULT).exists():
        df = load_data(DATA_PATH_DEFAULT)
    else:
        st.sidebar.error(f"No encontré '{DATA_PATH_DEFAULT}'. Sube tu CSV.")
        st.stop()

st.sidebar.subheader("📈 Market Intelligence")
market_up = st.sidebar.file_uploader("Sube market_intel.csv (opcional)", type=["csv"], key="uploader_market")
if market_up is not None:
    market_df = load_market_intel(market_up)
else:
    market_df = load_market_intel(MARKET_DEFAULT) if Path(MARKET_DEFAULT).exists() else None

success_model, sales_model, ACC, AUC, CM, MAE = train_models(df)

st.title("🧠 Product Lab IA — FINAL")
st.caption("Empaque + Shelf + Video + Producto Nuevo + Market + Inversionista")

k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("Registros", f"{len(df):,}")
k2.metric("Precisión", f"{ACC*100:.2f}%")
k3.metric("AUC", f"{AUC:.3f}")
k4.metric("Éxito base", f"{df['exito'].mean()*100:.1f}%")
k5.metric("MAE ventas", f"{MAE:,.0f} u.")
st.divider()

# ============================================================
# PRODUCTO NUEVO — What-if engine
# ============================================================
def coldstart_recommendations(success_model, sales_model, base_row: dict, top_k: int = 12):
    def _sf(x, d=0.0):
        try: return float(x)
        except Exception: return float(d)

    precio_base = max(_sf(base_row.get("precio", 100)), 1e-6)
    margen_base = np.clip(_sf(base_row.get("margen_pct", 30)), 0, 90)
    conn_base = np.clip(_sf(base_row.get("conexion_score", 60)), 0, 100)

    precio_grid = np.array([0.85, 0.92, 1.00, 1.08, 1.15]) * precio_base
    margen_grid = np.clip(np.array([-6, -3, 0, 3, 6]) + margen_base, 0, 90)
    claims_proxy_grid = np.array([0, 6, 12, 18], dtype=float)
    pack_proxy_grid = np.array([0, 6, 12, 18], dtype=float)

    Xb = pd.DataFrame([base_row])
    base_prob = float(success_model.predict_proba(Xb)[0][1])
    base_sales = float(sales_model.predict(Xb)[0])

    rows = []
    for p in precio_grid:
        for m in margen_grid:
            for dc in claims_proxy_grid:
                for dp in pack_proxy_grid:
                    row = dict(base_row)
                    row["precio"] = float(p)
                    row["margen_pct"] = float(m)
                    row["conexion_score"] = float(np.clip(conn_base + 0.55*dc + 0.55*dp, 0, 100))

                    X = pd.DataFrame([row])
                    prob = float(success_model.predict_proba(X)[0][1])
                    sales = float(sales_model.predict(X)[0])

                    rows.append({
                        "prob_exito": prob,
                        "ventas_unidades": max(0.0, sales),
                        "precio": float(p),
                        "margen_pct": float(m),
                        "delta_claims_proxy": float(dc),
                        "delta_pack_proxy": float(dp),
                    })

    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame([{
            "prob_exito": base_prob,
            "ventas_unidades": base_sales,
            "precio": precio_base,
            "margen_pct": margen_base,
            "delta_claims_proxy": 0.0,
            "delta_pack_proxy": 0.0,
        }])

    out["uplift_prob_pp"] = (out["prob_exito"] - base_prob) * 100.0
    out["uplift_sales"] = out["ventas_unidades"] - base_sales
    out["rank_score"] = (out["prob_exito"]*100.0) + 0.0025*out["ventas_unidades"] + 0.6*out["uplift_prob_pp"]
    out = out.sort_values("rank_score", ascending=False).head(top_k).reset_index(drop=True)
    best = out.iloc[0].to_dict()

    recs_txt = []
    if best["delta_pack_proxy"] >= 12:
        recs_txt.append("Mejora empaque (claridad + shelf-pop): el escenario top depende fuerte del pack.")
    if best["delta_claims_proxy"] >= 12:
        recs_txt.append("Optimiza claims: 2–3 claims máximo y jerarquía clara (beneficio principal primero).")
    if best["precio"] < precio_base:
        recs_txt.append("Prueba precio ligeramente menor: el escenario top sube probabilidad.")
    if best["margen_pct"] > margen_base:
        recs_txt.append("Margen puede subir sin matar ventas (escenario top): revisa costo/gramaje/promo.")
    if not recs_txt:
        recs_txt.append("Itera micro-ajustes: precio ±8% y margen ±3–6 pp + refuerzo pack/claims.")

    summary = {
        "base_prob_%": base_prob*100.0,
        "best_prob_%": float(best["prob_exito"])*100.0,
        "uplift_prob_pp": float(best["uplift_prob_pp"]),
        "base_sales": float(base_sales),
        "best_sales": float(best["ventas_unidades"]),
        "uplift_sales": float(best["uplift_sales"]),
    }
    return out, recs_txt, summary

# ============================================================
# UI TABS (incluye Inversionista)
# ============================================================
tab_sim, tab_claims, tab_pack, tab_shelf, tab_video, tab_new, tab_invest, tab_market, tab_ins, tab_report, tab_diag = st.tabs([
    "🧪 Simulador",
    "🏷️ Claims Lab",
    "📦 Pack Vision+",
    "🧲 Shelf (Foto/Packs)",
    "🎥 Video Shelf",
    "🧊 Producto Nuevo",
    "💼 Inversionista",
    "📈 Market Intelligence",
    "📊 Insights",
    "📄 Reporte",
    "🧠 Diagnóstico",
])

# ----------------------------
# 🧪 SIMULADOR (idéntico base)
# ----------------------------
with tab_sim:
    st.subheader("🧪 Simulador What-If (éxito + ventas + KPIs)")

    marcas = sorted(df["marca"].unique().tolist())
    canales = sorted(df["canal"].unique().tolist())

    c1,c2,c3 = st.columns(3)
    marca = c1.selectbox("Marca", marcas, key="sim_marca")
    canal = c2.selectbox("Canal", canales, key="sim_canal")
    segmento = c3.selectbox("Segmento", ["fit","kids","premium","value"], key="sim_segmento")
    canal_norm = str(canal).lower().strip()

    b1,b2,b3,b4,b5 = st.columns(5)
    precio = b1.number_input("Precio", value=float(df["precio"].median()), key="sim_precio")
    competencia = b2.slider("Competencia (1-10)", 1, 10, int(df["competencia"].median()), key="sim_comp")
    demanda = b3.slider("Demanda (10-100)", 10, 100, int(df["demanda"].median()), key="sim_dem")
    tendencia = b4.slider("Tendencia (20-100)", 20, 100, int(df["tendencia"].median()), key="sim_tend")
    margen_pct = b5.slider("Margen %", 0, 90, int(np.clip(df["margen_pct"].median(), 0, 90)), key="sim_margen")

    st.markdown("### 🏷️ Claims")
    recs = recommend_claims(segmento, canal_norm, 10)
    claim_opts = [c for c,_ in recs]
    selected_claims = st.multiselect("Selecciona claims (ideal 2-3)", claim_opts, default=claim_opts[:2], key="sim_claims")
    cscore = claims_score(selected_claims, canal_norm)

    st.markdown("### 📦 Empaque (sliders)")
    p1,p2,p3 = st.columns(3)
    pack_leg = p1.slider("Legibilidad (0-100)", 0, 100, 65, key="sim_leg")
    pack_pop = p2.slider("Shelf pop (0-100)", 0, 100, 70, key="sim_pop")
    pack_cla = p3.slider("Claridad (0-100)", 0, 100, 65, key="sim_cla")

    attention = float(np.clip(0.55*pack_pop + 0.45*pack_cla, 0, 100))
    memory = float(np.clip(0.45*pack_pop + 0.35*pack_cla + 0.20*pack_leg, 0, 100))
    emotion_pos = float(np.clip(0.45*pack_pop + 0.25*pack_leg + 0.30*cscore, 0, 100))

    conexion_score = clip(0.45*demanda + 0.35*pack_pop + 0.20*cscore, 0, 100)

    entrada = pd.DataFrame([{
        "precio": float(precio),
        "competencia": float(competencia),
        "demanda": float(demanda),
        "tendencia": float(tendencia),
        "margen_pct": float(margen_pct),
        "conexion_score": float(conexion_score),
        "rating_conexion": 7.0,
        "sentiment_score": 1.0,
        "marca": str(marca).lower(),
        "canal": str(canal_norm).lower(),
    }])

    if st.button("🚀 Simular", key="btn_sim"):
        prob = float(success_model.predict_proba(entrada)[0][1])
        ventas = max(0.0, float(sales_model.predict(entrada)[0]))

        utilities = np.array([prob*10, 5.0, 4.8], dtype=float)
        prob_choice = float(softmax(utilities)[0] * 100)

        st.session_state.last_sim = {
            "marca": marca, "canal": canal_norm, "segmento": segmento,
            "precio": float(precio), "competencia": float(competencia),
            "demanda": float(demanda), "tendencia": float(tendencia),
            "margen_pct": float(margen_pct),
            "claims": selected_claims, "claims_score": float(cscore),
            "pack_leg": float(pack_leg), "pack_pop": float(pack_pop), "pack_cla": float(pack_cla),
            "kpi_attention": float(attention),
            "kpi_emotion_pos": float(emotion_pos),
            "kpi_memory": float(memory),
            "prob_choice_est": float(prob_choice),
            "conexion_score": float(conexion_score),
            "prob_exito": float(prob),
            "ventas_unidades": float(ventas),
        }

        o1,o2,o3,o4,o5 = st.columns(5)
        o1.metric("Prob. éxito", f"{prob*100:.1f}%")
        o2.metric("Ventas predichas", f"{ventas:,.0f} u.")
        o3.metric("Atención visual", f"{attention:.1f}/100")
        o4.metric("Emoción positiva", f"{emotion_pos:.1f}/100")
        o5.metric("Recordación", f"{memory:.1f}/100")
        st.metric("Probabilidad estimada de elección", f"{prob_choice:.1f}%")

        st.markdown("### ✅ Conclusión")
        st.write("• Una identidad bien construida + un empaque diferenciado + datos = más elección real.")

# ----------------------------
# 🏷️ CLAIMS LAB
# ----------------------------
with tab_claims:
    st.subheader("🏷️ Claims Lab")
    c1,c2 = st.columns(2)
    seg = c1.selectbox("Segmento", ["fit","kids","premium","value"], key="cl_seg")
    can = c2.selectbox("Canal", ["retail","marketplace"], key="cl_can")

    recs = recommend_claims(seg, can, 12)
    rec_df = pd.DataFrame(recs, columns=["claim","score"])
    rec_df["score"] = (rec_df["score"]*100).round(1)
    st.dataframe(rec_df, use_container_width=True)

    selected = st.multiselect("Selecciona 2–3 claims", rec_df["claim"].tolist(), default=rec_df["claim"].tolist()[:2], key="cl_sel")
    st.metric("Claims Score", f"{claims_score(selected, can):.1f}/100")
    st.info("Validar compliance/regulatorio antes de imprimir en empaque.")

# ----------------------------
# 📦 PACK VISION+
# ----------------------------
with tab_pack:
    st.subheader("📦 Pack Vision+ (imagen → KPIs → quick wins)")
    pro_mode = st.toggle("Modo PRO (sentiment con fallback)", value=False, key="pack_pro")
    text_for_sent = st.text_input("Texto/copy para sentimiento (opcional)", "se ve premium y me encanta", key="pack_txt")

    img_file = st.file_uploader("Sube empaque (png/jpg)", type=["png","jpg","jpeg"], key="pack_img")
    if img_file is None:
        st.info("Sube tu empaque para análisis.")
    else:
        img = Image.open(img_file)
        st.image(img, caption="Empaque cargado", use_container_width=True)

        m = image_metrics(img)
        sc = pack_scores_from_metrics(m)
        heat = pack_heatmap_image_from_edges(img)

        claims_proxy = 65.0
        emo_deep = deep_sentiment_from_text(text_for_sent, enabled=pro_mode)

        attention = kpi_attention_visual(m, sc)
        memory = kpi_visual_memory(sc, m)
        emotion_pos = float(np.clip(0.55*emotion_proxy(sc, claims_proxy) + 0.45*emo_deep, 0, 100))
        choice_3s = pack_3sec_choice_score(sc["pack_legibility_score"], sc["pack_shelf_pop_score"], sc["pack_clarity_score"], emotion_pos)

        a1,a2,a3,a4 = st.columns(4)
        a1.metric("Atención visual", f"{attention:.1f}/100")
        a2.metric("Emoción positiva", f"{emotion_pos:.1f}/100")
        a3.metric("Recordación", f"{memory:.1f}/100")
        a4.metric("Elección 3s", f"{choice_3s:.1f}/100")

        st.image(heat, caption="Heatmap (proxy atención visual)", use_container_width=True)

        st.markdown("### Quick wins")
        for w in pack_quick_wins(sc, m):
            st.write("•", w)

        st.session_state.last_pack = {
            "attention": attention,
            "emotion_pos": emotion_pos,
            "memory": memory,
            "choice_3s": choice_3s,
            "legibility": sc["pack_legibility_score"],
            "shelf_pop": sc["pack_shelf_pop_score"],
            "clarity": sc["pack_clarity_score"],
            "pro_mode": bool(pro_mode),
        }

# ----------------------------
# 🧲 SHELF (packs sueltos + foto) — usa funciones del bloque 4
# ----------------------------
with tab_shelf:
    st.subheader("🧲 Shelf 3s (Packs sueltos o Foto de anaquel)")

    st.markdown("## A) Packs sueltos")
    p0 = st.file_uploader("Tu pack", type=["png","jpg","jpeg"], key="s0")
    p1 = st.file_uploader("Competidor 1", type=["png","jpg","jpeg"], key="s1")
    p2 = st.file_uploader("Competidor 2", type=["png","jpg","jpeg"], key="s2")
    p3 = st.file_uploader("Competidor 3 (opcional)", type=["png","jpg","jpeg"], key="s3")

    packs = [p0,p1,p2,p3]
    rows = []
    if any(packs):
        for i,f in enumerate(packs):
            if f is None: 
                continue
            im = Image.open(f)
            m = image_metrics(im)
            sc = pack_scores_from_metrics(m)
            emotion_pos = emotion_proxy(sc, 65.0)
            choice = pack_3sec_choice_score(sc["pack_legibility_score"], sc["pack_shelf_pop_score"], sc["pack_clarity_score"], emotion_pos)
            rows.append({"pack": f"pack_{i}", "choice_3s": round(choice, 1)})

        if rows:
            rank_df = shelf_rank_from_pack_scores(rows)
            st.dataframe(rank_df, use_container_width=True)

    st.divider()
    st.markdown("## B) Foto de anaquel + ROIs")
    shelf_img_file = st.file_uploader("Sube foto de anaquel", type=["png","jpg","jpeg"], key="shelf_photo")
    if shelf_img_file:
        shelf_img = Image.open(shelf_img_file)
        st.image(shelf_img, caption="Anaquel (original)", use_container_width=True)
        w,h = shelf_img.size

        labels = [
            st.text_input("Etiqueta ROI 0 (tu pack)", "tu_pack", key="roi_lab_0"),
            st.text_input("Etiqueta ROI 1", "comp_1", key="roi_lab_1"),
            st.text_input("Etiqueta ROI 2", "comp_2", key="roi_lab_2"),
            st.text_input("Etiqueta ROI 3", "comp_3", key="roi_lab_3"),
        ]

        def roi_controls(idx):
            c1,c2,c3,c4 = st.columns(4)
            x1 = c1.slider("x1", 0, w-1, int(w*0.05), key=f"roi_{idx}_x1")
            y1 = c2.slider("y1", 0, h-1, int(h*0.10), key=f"roi_{idx}_y1")
            x2 = c3.slider("x2", 1, w, int(w*0.25), key=f"roi_{idx}_x2")
            y2 = c4.slider("y2", 1, h, int(h*0.40), key=f"roi_{idx}_y2")
            return (x1,y1,x2,y2)

        rois = [roi_controls(i) for i in range(4)]
        st.image(draw_rois(shelf_img, rois, labels), caption="Preview ROIs", use_container_width=True)

        if st.button("🧲 Calcular Shelf 3s (FOTO)", key="btn_shelf_photo"):
            rows2 = []
            cols = st.columns(4)
            for idx,(roi,lab) in enumerate(zip(rois, labels)):
                crop = crop_image(shelf_img, *roi)
                m = image_metrics(crop)
                sc = pack_scores_from_metrics(m)
                emotion_pos = emotion_proxy(sc, 65.0)
                choice = pack_3sec_choice_score(sc["pack_legibility_score"], sc["pack_shelf_pop_score"], sc["pack_clarity_score"], emotion_pos)
                rows2.append({"pack": lab, "choice_3s": round(choice, 1)})
                cols[idx].image(crop, caption=f"{lab} ({choice:.1f})", use_container_width=True)

            rank2 = shelf_rank_from_pack_scores(rows2)
            st.dataframe(rank2, use_container_width=True)

            st.session_state.last_shelf = {"mode": "photo", "rank": rank2.to_dict(orient="records")}
            st.session_state.learning_log.append({"timestamp": pd.Timestamp.now().isoformat(), "type": "shelf_photo", "rank": rank2.to_dict(orient="records")})
            st.success("Guardado en learning log.")

# ----------------------------
# 🎥 VIDEO SHELF
# ----------------------------
with tab_video:
    st.subheader("🎥 Video de anaquel (frame + ROIs + promedio)")
    video_file = st.file_uploader("Sube video (mp4/mov/webm)", type=["mp4","mov","webm"], key="video_upl")
    if video_file is not None:
        vb = video_file.getvalue()
        st.video(vb)

        frames, fps = _read_video_frames_bytes(vb, max_frames=180)
        if not frames:
            st.warning("No pude extraer frames. Prueba mp4 H.264 o video más corto.")
        else:
            frame_idx = st.slider("Frame", 0, len(frames)-1, min(5, len(frames)-1), key="vid_frame")
            frame = frames[frame_idx]
            st.image(frame, caption=f"Frame #{frame_idx}", use_container_width=True)

            w,h = frame.size
            labels = [
                st.text_input("Etiqueta ROI 0", "tu_pack", key="v_lab0"),
                st.text_input("Etiqueta ROI 1", "comp_1", key="v_lab1"),
                st.text_input("Etiqueta ROI 2", "comp_2", key="v_lab2"),
                st.text_input("Etiqueta ROI 3", "comp_3", key="v_lab3"),
            ]

            def roi_controls_video(idx):
                c1,c2,c3,c4 = st.columns(4)
                x1 = c1.slider("x1", 0, w-1, int(w*0.05), key=f"v_{idx}_x1")
                y1 = c2.slider("y1", 0, h-1, int(h*0.10), key=f"v_{idx}_y1")
                x2 = c3.slider("x2", 1, w, int(w*0.25), key=f"v_{idx}_x2")
                y2 = c4.slider("y2", 1, h, int(h*0.40), key=f"v_{idx}_y2")
                return (x1,y1,x2,y2)

            rois = [roi_controls_video(i) for i in range(4)]
            st.image(draw_rois(frame, rois, labels), caption="Preview ROIs", use_container_width=True)

            sample_n = st.slider("Frames a promediar", 1, min(25, len(frames)), 6, key="v_n")
            stride = st.slider("Stride", 1, 10, 3, key="v_stride")

            if st.button("🧲 Calcular Shelf 3s (VIDEO)", key="btn_video"):
                rows_acc = {lab: [] for lab in labels}
                picks = []
                cur = frame_idx
                while len(picks) < sample_n and cur < len(frames):
                    picks.append(cur)
                    cur += stride

                for fi in picks:
                    fr = frames[fi]
                    for (roi, lab) in zip(rois, labels):
                        crop = crop_image(fr, *roi)
                        m = image_metrics(crop)
                        sc = pack_scores_from_metrics(m)
                        emotion_pos = emotion_proxy(sc, 65.0)
                        choice = pack_3sec_choice_score(sc["pack_legibility_score"], sc["pack_shelf_pop_score"], sc["pack_clarity_score"], emotion_pos)
                        rows_acc[lab].append(float(choice))

                rows_out = [{"pack": lab, "choice_3s": round(float(np.mean(rows_acc[lab])) if rows_acc[lab] else 0.0, 1)} for lab in labels]
                rank = shelf_rank_from_pack_scores(rows_out)
                st.dataframe(rank, use_container_width=True)

                st.session_state.last_shelf = {"mode": "video", "frames_used": picks, "rank": rank.to_dict(orient="records")}
                st.session_state.learning_log.append({"timestamp": pd.Timestamp.now().isoformat(), "type": "shelf_video", "frames_used": picks, "rank": rank.to_dict(orient="records")})
                st.success("Guardado en learning log (video).")

# ----------------------------
# 🧊 PRODUCTO NUEVO
# ----------------------------
with tab_new:
    # (Mismo módulo que te puse antes, completo)
    st.subheader("🧊 Producto Nuevo — Cold Start (sin histórico)")
    st.caption("Predice éxito/ventas con atributos + proxies (claims/pack), aunque la marca no exista.")

    categorias = sorted(df["categoria"].unique().tolist()) if "categoria" in df.columns else ["cereales"]
    canales = sorted(df["canal"].unique().tolist())

    c1,c2,c3 = st.columns(3)
    categoria = c1.selectbox("Categoría comparable", categorias, key="new_cat")
    canal = c2.selectbox("Canal", canales, key="new_can")
    segmento = c3.selectbox("Segmento objetivo", ["fit","kids","premium","value"], key="new_seg")
    canal_norm = str(canal).lower().strip()

    b1,b2,b3,b4,b5 = st.columns(5)
    precio = b1.number_input("Precio", 1.0, 999999.0, float(df["precio"].median()), step=1.0, key="new_precio")
    margen = b2.slider("Margen %", 0, 90, int(np.clip(df["margen_pct"].median(), 0, 90)), key="new_margen")
    competencia = b3.slider("Competencia (1-10)", 1, 10, int(df["competencia"].median()), key="new_comp")
    demanda = b4.slider("Demanda (10-100)", 10, 100, int(df["demanda"].median()), key="new_dem")
    tendencia = b5.slider("Tendencia (20-100)", 20, 100, int(df["tendencia"].median()), key="new_tend")

    st.markdown("### 🏷️ Claims")
    recs = recommend_claims(segmento, canal_norm, 10)
    claim_opts = [c for c,_ in recs]
    claims_sel = st.multiselect("Selecciona 2-3 claims", claim_opts, default=claim_opts[:2], key="new_claims")
    cscore = float(claims_score(claims_sel, canal_norm))
    st.metric("Claims Score", f"{cscore:.1f}/100")

    st.markdown("### 📦 Empaque (opcional)")
    img = st.file_uploader("Sube empaque (PNG/JPG)", type=["png","jpg","jpeg"], key="new_pack")
    pack_choice = 60.0
    pack_emotion = 60.0
    pack_label = "neutral"

    if img is not None:
        im = Image.open(img)
        st.image(im, caption="Empaque cargado", use_container_width=True)
        m = image_metrics(im)
        sc = pack_scores_from_metrics(m)
        pack_choice = float(np.clip(0.6*sc["pack_shelf_pop_score"] + 0.4*sc["pack_clarity_score"], 0, 100))
        pack_emotion = float(np.clip(0.5*sc["pack_shelf_pop_score"] + 0.3*sc["pack_legibility_score"] + 0.2*cscore, 0, 100))
        pack_label = "exciting" if pack_emotion >= 75 else ("positive" if pack_emotion >= 60 else ("neutral" if pack_emotion >= 45 else "confusing"))

    conexion_score = float(clip(0.45*float(demanda) + 0.35*float(pack_choice) + 0.20*float(cscore), 0, 100))

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

    prob = float(success_model.predict_proba(entrada)[0][1])
    ventas_point = float(sales_model.predict(entrada)[0])

    comp = df[df["categoria"] == str(categoria).lower()].copy() if "categoria" in df.columns else df.copy()
    if comp.empty:
        comp = df.copy()

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
        0.45*(prob*100.0) +
        0.25*float(pack_choice) +
        0.15*float(cscore) +
        0.15*float(pack_emotion)
    )

    o1,o2,o3,o4 = st.columns(4)
    o1.metric("Prob. éxito", f"{prob*100:.1f}%")
    o2.metric("Ventas (punto)", f"{ventas_point:,.0f} u.")
    o3.metric("Rango comparables (p25–p75)", f"{p25:,.0f} — {p75:,.0f} u.")
    o4.metric("Launch Score", f"{launch_score:.1f}/100")

    st.session_state.last_new = {
        "categoria": categoria, "canal": canal_norm, "segmento": segmento,
        "precio": float(precio), "margen_pct": float(margen),
        "competencia": float(competencia), "demanda": float(demanda), "tendencia": float(tendencia),
        "claims": claims_sel, "claims_score": float(cscore),
        "pack_choice": float(pack_choice), "pack_emotion": float(pack_emotion),
        "pack_emotion_label": str(pack_label),
        "conexion_score_proxy": float(conexion_score),
        "prob_exito": float(prob), "ventas_point": float(ventas_point),
        "ventas_p25": float(p25), "ventas_p50": float(p50), "ventas_p75": float(p75),
        "launch_score": float(launch_score),
    }

    st.divider()
    st.markdown("## 🧠 Recomendaciones What-If (Producto Nuevo)")
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
        "canal": canal_norm,
    }

    if st.button("🚀 Generar recomendaciones (what-if)", key="btn_new_recos"):
        st.session_state.new_recos_error = ""
        try:
            out_df, recs_txt, summary = coldstart_recommendations(success_model, sales_model, base_row)
            st.session_state.new_recos_out = out_df
            st.session_state.new_recos_txt = recs_txt
            st.session_state.new_recos_sum = summary
        except Exception as e:
            st.session_state.new_recos_out = None
            st.session_state.new_recos_txt = None
            st.session_state.new_recos_sum = None
            st.session_state.new_recos_error = f"❌ Error: {repr(e)}"

    if st.session_state.new_recos_error:
        st.error(st.session_state.new_recos_error)

    if st.session_state.new_recos_out is not None:
        out_df = st.session_state.new_recos_out
        recs_txt = st.session_state.new_recos_txt or []
        summary = st.session_state.new_recos_sum or {}

        cA,cB,cC = st.columns(3)
        cA.metric("Prob base", f"{summary.get('base_prob_%', 0):.1f}%")
        cB.metric("Mejor prob", f"{summary.get('best_prob_%', 0):.1f}%")
        cC.metric("Uplift prob", f"+{summary.get('uplift_prob_pp', 0):.1f} pp")

        st.markdown("### ✅ Quick wins")
        for r in recs_txt:
            st.write("•", r)

# ----------------------------
# 💼 INVERSIONISTA (NUEVO, COMPLETO)
# ----------------------------
with tab_invest:
    st.subheader("💼 Vista Inversionista (TAM + escenarios + unit economics + launch score)")
    st.caption("Modo financiero para pitch: tamaño de mercado, capturable y upside.")

    a1,a2,a3 = st.columns(3)
    tam = a1.number_input("TAM anual (MXN)", value=5_000_000_000.0, step=100_000_000.0, key="inv_tam")
    som = a2.slider("SOM % (capturable)", 0.0, 10.0, 1.0, step=0.1, key="inv_som")
    share = a3.slider("Share objetivo %", 0.0, 5.0, 0.3, step=0.1, key="inv_share")

    st.markdown("### Unit Economics")
    b1,b2,b3 = st.columns(3)
    asp = b1.number_input("ASP ($/unidad)", value=55.0, key="inv_asp")
    cogs = b2.number_input("COGS ($/unidad)", value=32.0, key="inv_cogs")
    mkt = b3.number_input("Marketing mensual (MXN)", value=500_000.0, key="inv_mkt")

    gross = (asp - cogs)
    gross_pct = gross / max(asp, 1e-9)
    st.metric("Gross margin / unidad", f"${gross:.1f} ({gross_pct*100:.1f}%)")

    # Launch score: usa lo último (Producto Nuevo > Simulador)
    base_prob = None
    if st.session_state.last_new:
        base_prob = st.session_state.last_new.get("prob_exito")
        base_units = st.session_state.last_new.get("ventas_point")
    elif st.session_state.last_sim:
        base_prob = st.session_state.last_sim.get("prob_exito")
        base_units = st.session_state.last_sim.get("ventas_unidades")
    else:
        base_prob = 0.55
        base_units = 120000

    launch_score = float(base_prob * 100.0)

    st.markdown("### Escenarios (unidades/mes)")
    s1,s2,s3 = st.columns(3)
    units_low = s1.number_input("Low", value=float(max(20000, base_units*0.5)), key="inv_u_low")
    units_mid = s2.number_input("Mid", value=float(max(60000, base_units*1.0)), key="inv_u_mid")
    units_high = s3.number_input("High", value=float(max(120000, base_units*1.8)), key="inv_u_high")

    def scenario(units):
        rev = units * asp
        gp = units * gross
        op = gp - mkt
        return rev, gp, op

    rows = []
    for name, u in [("Low", units_low), ("Mid", units_mid), ("High", units_high)]:
        rev,gp,op = scenario(u)
        rows.append({
            "escenario": name,
            "unidades_mes": u,
            "ingresos_mes": rev,
            "gross_profit_mes": gp,
            "operating_profit_mes": op
        })

    inv_df = pd.DataFrame(rows)
    st.dataframe(inv_df.style.format({
        "unidades_mes":"{:.0f}",
        "ingresos_mes":"${:,.0f}",
        "gross_profit_mes":"${:,.0f}",
        "operating_profit_mes":"${:,.0f}",
    }), use_container_width=True)

    st.markdown("### Tamaño capturable (top-down)")
    som_value = tam * (som/100.0)
    share_value = som_value * (share/100.0)
    c1,c2,c3 = st.columns(3)
    c1.metric("SOM (MXN/año)", f"${som_value:,.0f}")
    c2.metric("Share objetivo (MXN/año)", f"${share_value:,.0f}")
    c3.metric("Launch Score", f"{launch_score:.1f}/100")

    st.session_state.last_invest = {
        "tam": float(tam), "som_pct": float(som), "share_pct": float(share),
        "asp": float(asp), "cogs": float(cogs), "gross": float(gross), "gross_pct": float(gross_pct),
        "mkt_mensual": float(mkt),
        "base_prob": float(base_prob),
        "base_units": float(base_units),
        "launch_score_base": float(launch_score),
        "escenarios": rows,
        "som_value": float(som_value),
        "share_value": float(share_value),
    }

# ----------------------------
# 📈 MARKET INTELLIGENCE
# ----------------------------
with tab_market:
    st.subheader("📈 Market Intelligence")
    if market_df is None or getattr(market_df, "empty", True):
        st.warning("No hay market_intel.csv cargado. Sube uno en el sidebar.")
    else:
        mdf = market_df.copy()
        mdf.columns = [str(c).strip().lower() for c in mdf.columns]
        possible_sub_cols = ["subcategoria","sub_categ","subcategoria_nombre","sub_category","subcategory","sub_cat"]
        sub_col = next((c for c in possible_sub_cols if c in mdf.columns), None)

        required = {"categoria","canal"}
        missing = required - set(mdf.columns)
        if missing:
            st.error(f"Faltan columnas obligatorias en market_intel.csv: {sorted(list(missing))}")
            st.write("Columnas detectadas:", mdf.columns.tolist())
        else:
            col1,col2,col3 = st.columns(3)
            categorias = sorted(mdf["categoria"].dropna().astype(str).unique().tolist())
            canales = sorted(mdf["canal"].dropna().astype(str).unique().tolist())

            cat = col1.selectbox("Categoría", ["Todas"] + categorias, key="mk_cat")
            can = col2.selectbox("Canal", ["Todos"] + canales, key="mk_can")

            dfm = mdf.copy()
            if cat != "Todas":
                dfm = dfm[dfm["categoria"].astype(str) == str(cat)]
            if can != "Todos":
                dfm = dfm[dfm["canal"].astype(str) == str(can)]

            if sub_col:
                subcats = sorted(dfm[sub_col].dropna().astype(str).unique().tolist())
                sub = col3.selectbox("Subcategoría", ["Todas"] + subcats, key="mk_sub")
                if sub != "Todas":
                    dfm = dfm[dfm[sub_col].astype(str) == str(sub)]
            else:
                col3.info("ℹ️ Sin subcategoría en tu CSV (filtro apagado).")

            st.divider()
            prefer_cols = [
                "marca","categoria", sub_col if sub_col else None, "canal",
                "tendencia","precio_promedio","claim_top","insight_consumidor",
                "crecimiento_categoria_pct","competencia_intensidad",
                "precio_p25","precio_p75",
                "tendencia_claim_top","tendencia_pack_top","share_lider_pct"
            ]
            show_cols = [c for c in prefer_cols if c and c in dfm.columns]
            st.dataframe(dfm[show_cols].head(200) if show_cols else dfm.head(200), use_container_width=True)

# ----------------------------
# 📊 INSIGHTS
# ----------------------------
with tab_ins:
    st.subheader("📊 Insights")
    left,right = st.columns(2)
    with left:
        st.markdown("**Ranking por marca (Conexión promedio)**")
        st.dataframe(df.groupby("marca")[["conexion_score"]].mean().sort_values("conexion_score", ascending=False).round(2), use_container_width=True)

        st.markdown("**Ranking por marca (Éxito %)**")
        ex_m = df.groupby("marca")[["exito"]].mean().sort_values("exito", ascending=False)
        ex_m["exito_%"] = (ex_m["exito"]*100).round(1)
        st.dataframe(ex_m[["exito_%"]], use_container_width=True)

    with right:
        st.markdown("**Ranking por marca (Ventas promedio)**")
        st.dataframe(df.groupby("marca")[["ventas_unidades"]].mean().sort_values("ventas_unidades", ascending=False).round(0), use_container_width=True)

# (Reporte y Diagnóstico van en el bloque 8)

# ============================================================
# REPORT + DIAGNOSTIC (incluye Inversionista)
# ============================================================

with tab_report:
    st.subheader("📄 Reporte Ejecutivo (TXT + Inputs CSV)")

    def build_report_txt():
        lines = []
        lines.append("PRODUCT LAB IA — REPORTE EJECUTIVO")
        lines.append("="*55)

        if st.session_state.last_sim:
            s = st.session_state.last_sim
            lines.append("\n[SIMULADOR]")
            lines.append(f"Marca/Canal/Segmento: {s['marca']} / {s['canal']} / {s['segmento']}")
            lines.append(f"Precio: {s['precio']:.2f} | Margen%: {s['margen_pct']:.1f}")
            lines.append(f"Prob éxito: {s['prob_exito']*100:.1f}% | Ventas: {s['ventas_unidades']:.0f} u.")
            lines.append(f"Atención visual: {s['kpi_attention']:.1f}/100")
            lines.append(f"Emoción positiva: {s['kpi_emotion_pos']:.1f}/100")
            lines.append(f"Recordación: {s['kpi_memory']:.1f}/100")
            lines.append(f"Prob. elección estimada: {s['prob_choice_est']:.1f}%")
            lines.append("Conclusión: identidad + empaque + datos = más elección real.")

        if st.session_state.last_pack:
            p = st.session_state.last_pack
            lines.append("\n[PACK VISION+]")
            lines.append(f"Atención: {p['attention']:.1f}/100 | Emoción+: {p['emotion_pos']:.1f}/100 | Recordación: {p['memory']:.1f}/100")
            lines.append(f"Elección 3s: {p['choice_3s']:.1f}/100 | PRO mode: {p['pro_mode']}")

        if st.session_state.last_shelf:
            sh = st.session_state.last_shelf
            lines.append("\n[SHELF 3-SECOND]")
            lines.append(f"Modo: {sh.get('mode','-')}")
            rank = sh.get("rank", [])
            if rank:
                lines.append("Ranking (MNL prob %):")
                for r in rank:
                    lines.append(f" - {r.get('pack','')}: {r.get('mnl_prob_%','')}% (choice {r.get('choice_3s','')})")

        if st.session_state.last_new:
            n = st.session_state.last_new
            lines.append("\n[PRODUCTO NUEVO]")
            lines.append(f"Categoría/Canal/Segmento: {n['categoria']} / {n['canal']} / {n['segmento']}")
            lines.append(f"Precio: {n['precio']:.2f} | Margen%: {n['margen_pct']:.1f}")
            lines.append(f"Prob éxito: {n['prob_exito']*100:.1f}% | Ventas punto: {n['ventas_point']:.0f} u.")
            lines.append(f"Comparables p25-p75: {n['ventas_p25']:.0f} — {n['ventas_p75']:.0f} u.")
            lines.append(f"Launch Score: {n['launch_score']:.1f}/100")

        if st.session_state.last_invest:
            inv = st.session_state.last_invest
            lines.append("\n[INVERSIONISTA]")
            lines.append(f"TAM: ${inv['tam']:.0f} | SOM%: {inv['som_pct']:.2f} | Share%: {inv['share_pct']:.2f}")
            lines.append(f"SOM (MXN/año): ${inv['som_value']:.0f} | Share objetivo (MXN/año): ${inv['share_value']:.0f}")
            lines.append(f"ASP: {inv['asp']:.2f} | COGS: {inv['cogs']:.2f} | GM%: {inv['gross_pct']*100:.1f}")
            lines.append(f"Marketing mensual: ${inv['mkt_mensual']:.0f}")
            lines.append(f"Launch Score base: {inv['launch_score_base']:.1f}/100")

        return "\n".join(lines)

    report_txt = build_report_txt()
    st.download_button("⬇️ Descargar reporte_ejecutivo.txt", report_txt, "reporte_ejecutivo.txt", "text/plain", key="dl_txt")

    rows = []
    if st.session_state.last_sim: rows.append({"tipo": "simulador", **st.session_state.last_sim})
    if st.session_state.last_pack: rows.append({"tipo": "pack", **st.session_state.last_pack})
    if st.session_state.last_shelf: rows.append({"tipo": "shelf", **st.session_state.last_shelf})
    if st.session_state.last_new: rows.append({"tipo": "producto_nuevo", **st.session_state.last_new})
    if st.session_state.last_invest: rows.append({"tipo": "inversionista", **st.session_state.last_invest})

    if rows:
        inputs_df = pd.json_normalize(rows)
        st.download_button("⬇️ Descargar inputs_reporte.csv", df_to_csv_bytes(inputs_df), "inputs_reporte.csv", "text/csv", key="dl_csv")
        st.dataframe(inputs_df.head(80), use_container_width=True)
    else:
        st.info("Corre módulos para generar inputs.")

    st.divider()
    st.markdown("## 📥 Learning log")
    if st.session_state.learning_log:
        log_df = pd.DataFrame(st.session_state.learning_log)
        st.dataframe(log_df.tail(50), use_container_width=True)
        st.download_button("Descargar learning_log.csv", df_to_csv_bytes(log_df), "learning_log.csv", "text/csv", key="dl_log")
    else:
        st.info("Aún no hay registros.")

with tab_diag:
    st.subheader("🧠 Diagnóstico")
    st.markdown("### Matriz de confusión (éxito)")
    st.dataframe(pd.DataFrame(CM, index=["Real 0","Real 1"], columns=["Pred 0","Pred 1"]), use_container_width=True)
    st.markdown("### Métricas")
    st.write(f"Precisión: **{ACC*100:.2f}%**")
    st.write(f"AUC: **{AUC:.3f}**")
    st.write(f"MAE ventas: **{MAE:,.0f}** unidades")
