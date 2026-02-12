# ============================================================
# app_v6.py — Plataforma IA Producto Lab (v4)
# Producto + Empaque + Claims + Shelf (3s) + Video + PRO CNN+Deep Sentiment (fallback)
# ============================================================

import os, io, time, tempfile
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

import requests
import cv2

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, mean_absolute_error


# ----------------------------
# App config
# ----------------------------
st.set_page_config(
    page_title="Plataforma IA | Producto + Empaque + Claims + Shelf (v4)",
    layout="wide"
)

DATA_PATH_DEFAULT = "mercado_cereales_5000_con_ventas.csv"
MARKET_PATH = "market_intel.csv"

REQUIRED_BASE = {
    "marca", "categoria", "canal", "precio", "costo", "margen", "margen_pct",
    "competencia", "demanda", "tendencia", "estacionalidad",
    "rating_conexion", "comentario", "sentiment_score",
    "conexion_score", "conexion_alta", "score_latente", "exito"
}
REQUIRED_SALES = {"ventas_unidades", "ventas_ingresos", "utilidad"}

# ----------------------------
# Helpers
# ----------------------------
def _clean_str_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()

def clip(v, a, b):
    return float(max(a, min(b, v)))

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def bar_df_from_value_counts(vc: pd.Series) -> pd.DataFrame:
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
        draw.text((x1+4, y1+4), str(lab), fill=(255,0,0))
    return im

# ----------------------------
# Session defaults
# ----------------------------
if "learning_log" not in st.session_state: st.session_state.learning_log = []
if "last_sim" not in st.session_state: st.session_state.last_sim = None
if "last_shelf" not in st.session_state: st.session_state.last_shelf = None
if "last_new" not in st.session_state: st.session_state.last_new = None
if "last_invest" not in st.session_state: st.session_state.last_invest = None
if "new_recos_out" not in st.session_state: st.session_state.new_recos_out = None
if "new_recos_txt" not in st.session_state: st.session_state.new_recos_txt = None
if "new_recos_sum" not in st.session_state: st.session_state.new_recos_sum = None
if "new_recos_error" not in st.session_state: st.session_state.new_recos_error = ""

# ============================================================
# Claims Engine
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
# Pack Vision (proxy) + Heatmap
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


# ============================================================
# Emotion + 3-second choice + MNL
# ============================================================
def pack_emotion_score(pack_legibility, pack_pop, pack_clarity, claims_score_val, copy_tone: int):
    visual = 0.40*(pack_pop/100) + 0.30*(pack_clarity/100) + 0.15*(pack_legibility/100)
    claims = 0.15*(claims_score_val/100)
    tone_boost = 0.06 if copy_tone > 0 else (-0.06 if copy_tone < 0 else 0.0)
    score = (visual + claims + tone_boost) * 100
    return float(np.clip(score, 0, 100))

def pack_3sec_choice_score(leg, pop, clarity, emotion):
    s = 0.35*(pop/100) + 0.25*(clarity/100) + 0.20*(leg/100) + 0.20*(emotion/100)
    return float(np.clip(s*100, 0, 100))

def shelf_rank_from_pack_scores(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows).copy()
    df["utility"] = df["choice_3s"] / 10.0
    probs = softmax(df["utility"].values)
    df["mnl_prob"] = probs
    df = df.sort_values("mnl_prob", ascending=False).reset_index(drop=True)
    df["mnl_prob_%"] = (df["mnl_prob"]*100).round(1)
    return df[["pack", "choice_3s", "mnl_prob_%"]]

# ============================================================
# ✅ MODO PRO (CNN + Deep Sentiment) — Ruta B (ligero)
# Hugging Face Inference API con fallback automático
# ============================================================
HF_TOKEN = os.getenv("HF_TOKEN", "") or (st.secrets.get("HF_TOKEN", "") if hasattr(st, "secrets") else "")
HF_API_BASE = "https://api-inference.huggingface.co/models/"
HF_MODEL_CAPTION = "Salesforce/blip-image-captioning-base"
HF_MODEL_SENTIMENT = "distilbert-base-uncased-finetuned-sst-2-english"

def _hf_headers():
    if HF_TOKEN:
        return {"Authorization": f"Bearer {HF_TOKEN}"}
    return {}

def hf_infer(model_id: str, payload=None, binary: bytes | None = None, timeout=35):
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN no configurado")
    url = HF_API_BASE + model_id
    headers = _hf_headers()
    if binary is not None:
        r = requests.post(url, headers=headers, data=binary, timeout=timeout)
    else:
        r = requests.post(url, headers=headers, json=payload or {}, timeout=timeout)

    if r.status_code == 503:
        time.sleep(2.0)
        if binary is not None:
            r = requests.post(url, headers=headers, data=binary, timeout=timeout)
        else:
            r = requests.post(url, headers=headers, json=payload or {}, timeout=timeout)

    r.raise_for_status()
    return r.json()

def pil_to_jpeg_bytes(img: Image.Image, quality=92) -> bytes:
    b = io.BytesIO()
    img.convert("RGB").save(b, format="JPEG", quality=quality)
    return b.getvalue()

def hf_caption(img: Image.Image) -> str:
    raw = hf_infer(HF_MODEL_CAPTION, binary=pil_to_jpeg_bytes(img))
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        return str(raw[0].get("generated_text", "")).strip()
    if isinstance(raw, dict) and "generated_text" in raw:
        return str(raw["generated_text"]).strip()
    return ""

def hf_sentiment_from_text(text: str) -> dict:
    if not text:
        return {"label": "NEUTRAL", "score": 0.0}
    raw = hf_infer(HF_MODEL_SENTIMENT, payload={"inputs": text})
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        return {"label": raw[0].get("label", "NEUTRAL"), "score": float(raw[0].get("score", 0.0))}
    return {"label": "NEUTRAL", "score": 0.0}

def pro_pack_sentiment(img: Image.Image) -> dict:
    cap = hf_caption(img)
    sent = hf_sentiment_from_text(cap)
    return {"caption": cap, "sent_label": sent["label"], "sent_score": float(sent["score"])}

def pro_sentiment_to_boost(sent_label: str, sent_score: float) -> float:
    lab = str(sent_label).upper().strip()
    sc = float(sent_score)
    if lab == "POSITIVE":
        return float(np.clip(8.0 * sc, 0, 8))
    if lab == "NEGATIVE":
        return float(np.clip(-8.0 * sc, -8, 0))
    return 0.0

def emotion_score_with_pro(proxy_emotion: float, pro_boost: float) -> float:
    return float(np.clip(float(proxy_emotion) + float(pro_boost), 0, 100))

def analyze_pack_image(img: Image.Image, claims_score_val: float, copy_tone: int, use_pro: bool):
    m = image_metrics(img)
    sc = pack_scores_from_metrics(m)
    heat = pack_heatmap_image_from_edges(img)

    proxy_emotion = pack_emotion_score(
        sc["pack_legibility_score"], sc["pack_shelf_pop_score"], sc["pack_clarity_score"],
        claims_score_val, copy_tone
    )

    pro = {
        "enabled": False,
        "caption": "",
        "sent_label": "NEUTRAL",
        "sent_score": 0.0,
        "boost": 0.0,
        "emotion_final": float(proxy_emotion),
        "error": ""
    }

    if use_pro:
        try:
            pro_out = pro_pack_sentiment(img)
            boost = pro_sentiment_to_boost(pro_out["sent_label"], pro_out["sent_score"])
            pro.update({
                "enabled": True,
                "caption": pro_out["caption"],
                "sent_label": pro_out["sent_label"],
                "sent_score": float(pro_out["sent_score"]),
                "boost": float(boost),
                "emotion_final": emotion_score_with_pro(proxy_emotion, boost),
            })
        except Exception as e:
            pro["error"] = f"{type(e).__name__}: {e}"
            pro["enabled"] = False
            pro["emotion_final"] = float(proxy_emotion)

    return {
        "metrics": m,
        "scores": sc,
        "heatmap": heat,
        "proxy_emotion": float(proxy_emotion),
        "pro": pro,
    }

def sample_frames_count(frames_len: int, start_idx: int, sample_n: int, stride: int) -> list[int]:
    picks = []
    cur = int(start_idx)
    while len(picks) < int(sample_n) and cur < int(frames_len):
        picks.append(cur)
        cur += int(stride)
    return picks

def analyze_video_rois_with_pro(frames: list[Image.Image], picks: list[int], rois: list[tuple], labels: list[str], use_pro: bool):
    rows_accum = {lab: [] for lab in labels}
    pro_meta = {lab: {"pos":0, "neg":0, "last_caption":"", "last_label":"", "last_score":0.0} for lab in labels}

    for fi in picks:
        fr = frames[fi]
        for roi, lab in zip(rois, labels):
            crop = crop_image(fr, *roi)
            res = analyze_pack_image(crop, claims_score_val=0.0, copy_tone=0, use_pro=use_pro)
            sc = res["scores"]
            emotion_final = res["pro"]["emotion_final"]

            choice = pack_3sec_choice_score(
                sc["pack_legibility_score"],
                sc["pack_shelf_pop_score"],
                sc["pack_clarity_score"],
                emotion_final
            )
            rows_accum[lab].append(float(choice))

            if res["pro"]["enabled"]:
                lab_sent = str(res["pro"]["sent_label"]).upper()
                if lab_sent == "POSITIVE": pro_meta[lab]["pos"] += 1
                if lab_sent == "NEGATIVE": pro_meta[lab]["neg"] += 1
                pro_meta[lab]["last_caption"] = res["pro"]["caption"]
                pro_meta[lab]["last_label"] = res["pro"]["sent_label"]
                pro_meta[lab]["last_score"] = float(res["pro"]["sent_score"])

    rows2 = []
    for lab in labels:
        vals = rows_accum.get(lab, [])
        rows2.append({"pack": lab, "choice_3s": round(float(np.mean(vals)) if vals else 0.0, 1)})

    return rows2, pro_meta

# ============================================================
# Data loading + training
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
        "conexion_score","conexion_alta",
        "score_latente","exito"
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ["ventas_unidades","ventas_ingresos","utilidad"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=[
        "marca","categoria","canal","precio","competencia",
        "demanda","tendencia","margen_pct",
        "conexion_score","rating_conexion","sentiment_score","exito"
    ])

    df["exito"] = df["exito"].astype(int)

    df["ventas_unidades"] = pd.to_numeric(df.get("ventas_unidades", np.nan), errors="coerce")
    if "ventas_unidades" in df.columns:
        df["ventas_unidades"] = df["ventas_unidades"].fillna(df["ventas_unidades"].median())
    else:
        df["ventas_unidades"] = 0.0

    return df

@st.cache_data
def load_market_intel(path_or_file):
    try:
        mdf = pd.read_csv(path_or_file).copy()
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

    if not REQUIRED_SALES.issubset(set(df.columns)):
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


# ============================================================
# Sidebar: load dataset + market
# ============================================================
st.sidebar.title("⚙️ Datos")
uploaded = st.sidebar.file_uploader("Sube tu dataset (CSV)", type=["csv"], key="uploader_dataset")

if uploaded is not None:
    df = load_data(uploaded)
else:
    if Path(DATA_PATH_DEFAULT).exists():
        df = load_data(DATA_PATH_DEFAULT)
    else:
        st.sidebar.warning(f"No encontré '{DATA_PATH_DEFAULT}'. Sube tu CSV.")
        st.stop()

st.sidebar.subheader("📈 Market Intelligence")
market_up = st.sidebar.file_uploader("Sube market_intel.csv (opcional)", type=["csv"], key="uploader_market")
market_df = load_market_intel(market_up) if market_up else (pd.read_csv(MARKET_PATH) if Path(MARKET_PATH).exists() else None)

try:
    success_model, sales_model, ACC, AUC, CM, MAE = train_models(df)
except Exception as e:
    st.error(f"Error entrenando modelos: {e}")
    st.stop()

# ============================================================
# Header
# ============================================================
st.title("🧠 Plataforma IA: Producto + Empaque + Claims + Shelf (v4)")
st.caption("Éxito + Ventas + Insights + Pack Vision + Shelf 3s + Video + Modo PRO (CNN+Deep Sentiment) con fallback")

k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("Registros", f"{len(df):,}")
k2.metric("Precisión", f"{ACC*100:.2f}%")
k3.metric("AUC", f"{AUC:.3f}")
k4.metric("Éxito base", f"{df['exito'].mean()*100:.1f}%")
k5.metric("MAE ventas", f"{MAE:,.0f} u.")
st.divider()

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
# 📦 PACK VISION+ (incluye PRO con fallback)
# ============================================================
with tab_pack:
    st.subheader("📦 Pack Vision+ (imagen -> métricas -> heatmap -> quick wins + PRO)")

    use_pro = st.toggle("Modo PRO (CNN + Deep Sentiment) — HF API si hay token", value=False, key="pack_use_pro")
    if use_pro and not HF_TOKEN:
        st.warning("Modo PRO activado pero no veo HF_TOKEN. Haré fallback automático al proxy.")

    img_file = st.file_uploader("Sube imagen del empaque (PNG/JPG)", type=["png","jpg","jpeg"], key="pack_uploader")
    if img_file is None:
        st.info("Sube tu empaque para análisis visual.")
    else:
        img = Image.open(img_file)
        st.image(img, caption="Empaque cargado", use_container_width=True)

        res = analyze_pack_image(img, claims_score_val=0.0, copy_tone=0, use_pro=use_pro)
        m = res["metrics"]
        sc = res["scores"]
        heat = res["heatmap"]
        pro = res["pro"]

        a1,a2,a3,a4 = st.columns(4)
        a1.metric("Brillo", f"{m['brightness']:.2f}")
        a2.metric("Contraste", f"{m['contrast']:.2f}")
        a3.metric("Colorfulness", f"{m['colorfulness']:.2f}")
        a4.metric("Edge density", f"{m['edge_density']:.3f}")

        b1,b2,b3,b4 = st.columns(4)
        b1.metric("Legibilidad", f"{sc['pack_legibility_score']}/100")
        b2.metric("Shelf Pop", f"{sc['pack_shelf_pop_score']}/100")
        b3.metric("Claridad", f"{sc['pack_clarity_score']}/100")
        b4.metric("Emotion FINAL", f"{pro['emotion_final']:.1f}/100")

        st.image(heat, caption="Heatmap (proxy de atención visual)", use_container_width=True)

        if use_pro:
            if pro["enabled"]:
                st.success("PRO activo ✅ (caption + deep sentiment)")
                st.write(f"**Caption:** {pro['caption']}")
                st.write(f"**Sentiment:** {pro['sent_label']} ({pro['sent_score']:.3f}) | **Boost:** {pro['boost']:+.1f}")
            else:
                st.warning("PRO no disponible → fallback al proxy")
                if pro.get("error"):
                    st.caption(f"Detalle: {pro['error']}")

        st.markdown("### Quick wins (pack)")
        for w in pack_quick_wins(sc, m):
            st.write("•", w)


# ============================================================
# 🧲 SHELF & EMOTION (3s) — packs/foto/video (incluye PRO)
# ============================================================
with tab_shelf:
    st.subheader("🧲 Shelf & Emotion Predictor (3-Second Test)")
    st.caption("Sube tu pack y competidores (o foto/VIDEO de anaquel + ROIs). PRO opcional con fallback.")

    use_pro_shelf = st.toggle("Modo PRO en Shelf/Video (Deep Sentiment por ROI)", value=False, key="shelf_use_pro")
    if use_pro_shelf and not HF_TOKEN:
        st.warning("Modo PRO activado pero no hay HF_TOKEN. Fallback al proxy.")

    # ----------------------------
    # A) Packs sueltos
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

            res = analyze_pack_image(im, claims_score_val=0.0, copy_tone=0, use_pro=use_pro_shelf)
            sc = res["scores"]
            emotion = res["pro"]["emotion_final"]  # proxy o PRO

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
    # B) Foto de anaquel + ROIs
    # ----------------------------
    st.divider()
    st.markdown("## B) Foto de anaquel + ROIs (tu pack + hasta 3 competidores)")
    shelf_img_file = st.file_uploader("Sube foto de anaquel", type=["png","jpg","jpeg"], key="shelf_photo")

    if shelf_img_file:
        shelf_img = Image.open(shelf_img_file)
        st.image(shelf_img, caption="Anaquel (original)", use_container_width=True)

        w,h = shelf_img.size
        st.markdown("### Define ROIs con sliders (x1,y1,x2,y2)")

        labels = [
            st.text_input("Etiqueta ROI 0 (tu pack)", "tu_pack", key="roi_lab_0"),
            st.text_input("Etiqueta ROI 1", "comp_1", key="roi_lab_1"),
            st.text_input("Etiqueta ROI 2", "comp_2", key="roi_lab_2"),
            st.text_input("Etiqueta ROI 3", "comp_3", key="roi_lab_3"),
        ]

        def roi_controls(idx, label):
            st.markdown(f"**ROI {idx}: {label}**")
            c1,c2,c3,c4 = st.columns(4)
            x1 = c1.slider("x1", 0, w-1, int(w*0.05), key=f"roi_{idx}_x1")
            y1 = c2.slider("y1", 0, h-1, int(h*0.10), key=f"roi_{idx}_y1")
            x2 = c3.slider("x2", 1, w, int(w*0.25), key=f"roi_{idx}_x2")
            y2 = c4.slider("y2", 1, h, int(h*0.40), key=f"roi_{idx}_y2")
            return (x1,y1,x2,y2)

        rois = [
            roi_controls(0, labels[0]),
            roi_controls(1, labels[1]),
            roi_controls(2, labels[2]),
            roi_controls(3, labels[3]),
        ]

        preview = draw_rois(shelf_img, rois, labels)
        st.image(preview, caption="Preview ROIs", use_container_width=True)

        if st.button("🧲 Calcular Shelf 3s (anaquel)", key="btn_shelf_calc"):
            rows2 = []
            crops_show = st.columns(4)

            for idx,(roi,lab) in enumerate(zip(rois, labels)):
                crop = crop_image(shelf_img, *roi)

                res = analyze_pack_image(crop, claims_score_val=0.0, copy_tone=0, use_pro=use_pro_shelf)
                sc = res["scores"]
                emotion = res["pro"]["emotion_final"]

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

            st.session_state.last_shelf = {"mode": "shelf_photo", "labels": labels, "rank": rank2.to_dict(orient="records")}
            st.session_state.learning_log.append({
                "timestamp": pd.Timestamp.now().isoformat(),
                "type": "shelf_3s_photo",
                "labels": labels,
                "rank": rank2.to_dict(orient="records"),
            })
            st.success("Guardado en learning log.")

    # ----------------------------
    # C) Video de anaquel + frame + ROIs + promedio
    # ----------------------------
    st.divider()
    st.markdown("## C) Video de anaquel (frame selector + ROIs + ranking MNL)")
    st.caption("Sube un video (mp4/mov). Selecciona un frame y define ROIs. Opcional: promediar varios frames.")

    video_file = st.file_uploader("Sube video de anaquel (mp4/mov/webm)", type=["mp4","mov","webm"], key="shelf_video")

    def _read_video_frames_bytes(video_bytes: bytes, max_frames: int = 140):
        frames = []
        fps = None

        # guardo bytes a tmp
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return [], None

        fps = cap.get(cv2.CAP_PROP_FPS) or None
        idx = 0
        while idx < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
            idx += 1
        cap.release()
        return frames, float(fps) if fps else None

    if video_file is not None:
        vb = video_file.getvalue()
        st.video(vb)

        frames, fps = _read_video_frames_bytes(vb, max_frames=160)

        if not frames:
            st.warning("No pude extraer frames. Prueba mp4 h264 o video más corto.")
        else:
            st.success(f"Frames cargados: {len(frames)}" + (f" | FPS aprox: {fps:.1f}" if fps else ""))

            frame_idx = st.slider("Selecciona frame", 0, len(frames)-1, min(5, len(frames)-1), key="vid_frame_idx")
            frame = frames[frame_idx]
            st.image(frame, caption=f"Frame #{frame_idx}", use_container_width=True)

            w, h = frame.size
            labels = [
                st.text_input("Etiqueta ROI 0 (tu pack)", "tu_pack", key="vid_roi_lab_0"),
                st.text_input("Etiqueta ROI 1", "comp_1", key="vid_roi_lab_1"),
                st.text_input("Etiqueta ROI 2", "comp_2", key="vid_roi_lab_2"),
                st.text_input("Etiqueta ROI 3", "comp_3", key="vid_roi_lab_3"),
            ]

            def roi_controls_video(idx, label):
                st.markdown(f"**ROI {idx}: {label}**")
                c1,c2,c3,c4 = st.columns(4)
                x1 = c1.slider("x1", 0, w-1, int(w*0.05), key=f"vid_roi_{idx}_x1")
                y1 = c2.slider("y1", 0, h-1, int(h*0.10), key=f"vid_roi_{idx}_y1")
                x2 = c3.slider("x2", 1, w, int(w*0.25), key=f"vid_roi_{idx}_x2")
                y2 = c4.slider("y2", 1, h, int(h*0.40), key=f"vid_roi_{idx}_y2")
                return (x1,y1,x2,y2)

            rois = [
                roi_controls_video(0, labels[0]),
                roi_controls_video(1, labels[1]),
                roi_controls_video(2, labels[2]),
                roi_controls_video(3, labels[3]),
            ]

            preview = draw_rois(frame, rois, labels)
            st.image(preview, caption="Preview ROIs (Frame)", use_container_width=True)

            st.markdown("### Robustez (opcional): promediar varios frames")
            sample_n = st.slider("Frames a muestrear (promedio)", 1, min(20, len(frames)), 5, key="vid_sample_n")
            stride = st.slider("Stride (cada N frames)", 1, 10, 3, key="vid_stride")

            if st.button("🧲 Calcular Shelf 3s (VIDEO)", key="btn_shelf_video_calc"):
                picks = sample_frames_count(len(frames), frame_idx, sample_n, stride)
                st.write(f"Frames usados: {picks}")

                rows2, pro_meta = analyze_video_rois_with_pro(frames, picks, rois, labels, use_pro=use_pro_shelf)
                rank = shelf_rank_from_pack_scores(rows2)
                st.dataframe(rank, use_container_width=True)

                if use_pro_shelf:
                    st.markdown("### PRO resumen (por ROI)")
                    for lab in labels:
                        m = pro_meta.get(lab, {})
                        st.write(f"• **{lab}** | pos:{m.get('pos',0)} neg:{m.get('neg',0)} | last: {m.get('last_label','')} ({m.get('last_score',0):.3f})")
                        if m.get("last_caption"):
                            st.caption(f"caption: {m['last_caption']}")

                st.session_state.last_shelf = {
                    "mode": "shelf_video",
                    "frame_idx": int(frame_idx),
                    "frames_used": picks,
                    "labels": labels,
                    "rank": rank.to_dict(orient="records"),
                }
                st.session_state.learning_log.append({
                    "timestamp": pd.Timestamp.now().isoformat(),
                    "type": "shelf_3s_video",
                    "frame_idx": int(frame_idx),
                    "frames_used": picks,
                    "labels": labels,
                    "rank": rank.to_dict(orient="records"),
                })
                st.success("Guardado en learning log (video).")

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
        st.info("Aún no hay learning log. Corre Shelf 3s para registrar resultados.")

# ============================================================
# Producto Nuevo — What-if
# ============================================================
def coldstart_recommendations(success_model, sales_model, base_row: dict, top_k: int = 12):
    def _safe_float(x, default=0.0):
        try: return float(x)
        except Exception: return float(default)

    precio_base = max(_safe_float(base_row.get("precio", 100)), 1e-6)
    margen_base = np.clip(_safe_float(base_row.get("margen_pct", 30)), 0, 90)
    conn_base = np.clip(_safe_float(base_row.get("conexion_score", 60)), 0, 100)

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
                    conn2 = float(np.clip(conn_base + 0.55*dc + 0.55*dp, 0, 100))
                    row["conexion_score"] = conn2

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
        recs_txt.append("Optimiza claims: 2–3 claims máximo, jerarquía clara (beneficio principal primero).")
    if best["precio"] < precio_base:
        recs_txt.append("Prueba precio ligeramente menor: el escenario top sube probabilidad.")
    if best["margen_pct"] > margen_base:
        recs_txt.append("Ajuste de margen no mata ventas en el escenario top: revisa costo/gramaje/promo.")
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
# 🧊 PRODUCTO NUEVO
# ============================================================
with tab_new:
    st.subheader("🧊 Producto Nuevo — Cold Start (sin histórico)")
    st.caption("Predice éxito y ventas con atributos + proxies. Pack opcional. What-if incluido.")

    c1, c2, c3 = st.columns(3)
    categorias = sorted(df["categoria"].unique().tolist()) if "categoria" in df.columns else ["cereales"]
    canales = sorted(df["canal"].unique().tolist())
    categoria = c1.selectbox("Categoría comparable", categorias, key="new_cat_fix")
    canal = c2.selectbox("Canal", canales, key="new_canal_fix")
    segmento = c3.selectbox("Segmento objetivo", ["fit","kids","premium","value"], key="new_seg_fix")
    canal_norm = str(canal).lower().strip()

    b1, b2, b3, b4, b5 = st.columns(5)
    precio = b1.number_input("Precio", 1.0, 99999.0, float(df["precio"].median()), step=1.0, key="new_precio_fix")
    margen = b2.slider("Margen %", 0, 90, int(np.clip(df["margen_pct"].median(), 0, 90)), key="new_margen_fix")
    competencia = b3.slider("Competencia (1-10)", 1, 10, int(df["competencia"].median()), key="new_comp_fix")
    demanda = b4.slider("Demanda (10-100)", 10, 100, int(df["demanda"].median()), key="new_dem_fix")
    tendencia = b5.slider("Tendencia (20-100)", 20, 100, int(df["tendencia"].median()), key="new_tend_fix")

    st.markdown("### 🏷️ Claims (Cold Start)")
    recs = recommend_claims(segmento, canal_norm, 10)
    claim_opts = [c for c,_ in recs]
    claims_sel = st.multiselect("Selecciona 2-3 claims", claim_opts, default=claim_opts[:2], key="new_claims_fix")
    cscore = float(claims_score(claims_sel, canal_norm))
    st.metric("Claims Score", f"{cscore:.1f}/100")

    st.markdown("### 📦 Empaque (opcional, recomendado)")
    use_pro_new = st.toggle("Modo PRO en pack (caption+sentiment)", value=False, key="new_use_pro")
    if use_pro_new and not HF_TOKEN:
        st.warning("PRO activo pero no hay HF_TOKEN. Fallback al proxy.")

    img = st.file_uploader("Sube empaque (PNG/JPG)", type=["png","jpg","jpeg"], key="new_pack_fix")

    pack_choice = 60.0
    pack_emotion = 60.0
    pack_label = "neutral"

    if img is not None:
        im = Image.open(img)
        st.image(im, caption="Empaque cargado", use_container_width=True)

        res = analyze_pack_image(im, claims_score_val=cscore, copy_tone=0, use_pro=use_pro_new)
        sc = res["scores"]
        pro = res["pro"]

        pack_choice = float(clip(0.6*sc["pack_shelf_pop_score"] + 0.4*sc["pack_clarity_score"], 0, 100))
        pack_emotion = float(pro["emotion_final"])

        if pack_emotion >= 75: pack_label = "exciting"
        elif pack_emotion >= 60: pack_label = "positive"
        elif pack_emotion >= 45: pack_label = "neutral"
        else: pack_label = "confusing"

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Elección (3s) proxy", f"{pack_choice:.1f}/100")
        k2.metric("Emoción dominante", pack_label.upper())
        k3.metric("Legibilidad", f"{sc['pack_legibility_score']}/100")
        k4.metric("Claridad", f"{sc['pack_clarity_score']}/100")

        if use_pro_new:
            if pro["enabled"]:
                st.success("PRO activo ✅")
                st.write(f"**Caption:** {pro['caption']}")
                st.write(f"**Sentiment:** {pro['sent_label']} ({pro['sent_score']:.3f}) | Boost {pro['boost']:+.1f}")
            else:
                st.warning("PRO no disponible → fallback")
                if pro.get("error"): st.caption(pro["error"])

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

    comp = df.copy()
    comp2 = df[df["categoria"] == str(categoria).lower()].copy() if "categoria" in df.columns else df.copy()
    comp = comp2 if not comp2.empty else df.copy()

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
        0.45 * (prob*100.0) +
        0.25 * float(pack_choice) +
        0.15 * float(cscore) +
        0.15 * float(pack_emotion)
    )

    st.markdown("## 🎯 Resultado (Producto Nuevo)")
    o1, o2, o3, o4 = st.columns(4)
    o1.metric("Prob. éxito", f"{prob*100:.1f}%")
    o2.metric("Ventas (punto)", f"{ventas_point:,.0f} u.")
    o3.metric("Rango comparables (p25–p75)", f"{p25:,.0f} — {p75:,.0f} u.")
    o4.metric("Launch Score", f"{launch_score:.1f}/100")

    if launch_score >= 75: st.success("✅ GO — Alto potencial")
    elif launch_score >= 60: st.warning("🟡 AJUSTAR — Optimiza pack/claims/precio")
    else: st.error("🔴 NO-GO — Riesgo alto")

    st.markdown("### 🔍 Top comparables usados (20)")
    show_cols = [c for c in ["marca","precio","margen_pct","demanda","tendencia","ventas_unidades","exito"] if c in top.columns]
    st.dataframe(top[show_cols].copy(), use_container_width=True)

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
            with st.spinner("Calculando escenarios what-if..."):
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

        cA, cB, cC = st.columns(3)
        cA.metric("Prob base", f"{summary.get('base_prob_%', 0):.1f}%")
        cB.metric("Mejor prob", f"{summary.get('best_prob_%', 0):.1f}%")
        cC.metric("Uplift prob", f"+{summary.get('uplift_prob_pp', 0):.1f} pp")

        dA, dB, dC = st.columns(3)
        dA.metric("Ventas base", f"{summary.get('base_sales', 0):,.0f} u.")
        dB.metric("Mejor ventas", f"{summary.get('best_sales', 0):,.0f} u.")
        dC.metric("Uplift ventas", f"+{summary.get('uplift_sales', 0):,.0f} u.")

        st.markdown("### ✅ Quick wins")
        for r in recs_txt:
            st.write("•", r)

        st.markdown("### 🧪 Top escenarios")
        show = out_df.copy()
        show["prob_exito_%"] = (show["prob_exito"]*100).round(1)
        show["ventas_unidades"] = show["ventas_unidades"].round(0).astype(int)
        st.dataframe(show, use_container_width=True)
    else:
        st.info("Presiona el botón para generar recomendaciones.")


# ============================================================
# 💼 INVERSIONISTA
# ============================================================
with tab_invest:
    st.subheader("💼 Vista Inversionista (TAM + escenarios + unit economics + launch score)")

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

    base_prob = None
    if st.session_state.last_new: base_prob = st.session_state.last_new.get("prob_exito", None)
    elif st.session_state.last_sim: base_prob = st.session_state.last_sim.get("prob_exito", None)
    if base_prob is None: base_prob = 0.55
    launch_score = (base_prob*100)

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
    st.subheader("📈 Market Intelligence")

    if market_df is None or (isinstance(market_df, pd.DataFrame) and market_df.empty):
        st.warning("No hay datos de mercado cargados (market_intel.csv).")
    else:
        market_df = market_df.copy()
        market_df.columns = [str(c).strip().lower() for c in market_df.columns]

        possible_sub_cols = ["subcategoria","sub_categ","subcategoria_nombre","sub_category","subcategory","sub_cat"]
        sub_col = next((c for c in possible_sub_cols if c in market_df.columns), None)

        required = {"categoria","canal"}
        missing = required - set(market_df.columns)
        if missing:
            st.error(f"Faltan columnas obligatorias en market_intel.csv: {sorted(list(missing))}")
            st.write("Columnas detectadas:", market_df.columns.tolist())
        else:
            col1, col2, col3 = st.columns(3)
            categorias = sorted(market_df["categoria"].dropna().astype(str).unique().tolist())
            canales = sorted(market_df["canal"].dropna().astype(str).unique().tolist())
            cat = col1.selectbox("Categoría", ["Todas"] + categorias, key="mk_cat_v4")
            can = col2.selectbox("Canal", ["Todos"] + canales, key="mk_canal_v4")

            dfm = market_df.copy()
            if cat != "Todas":
                dfm = dfm[dfm["categoria"].astype(str) == str(cat)]
            if can != "Todos":
                dfm = dfm[dfm["canal"].astype(str) == str(can)]

            if sub_col:
                subcats = sorted(dfm[sub_col].dropna().astype(str).unique().tolist())
                sub = col3.selectbox("Subcategoría", ["Todas"] + subcats, key="mk_sub_v4")
                if sub != "Todas":
                    dfm = dfm[dfm[sub_col].astype(str) == str(sub)]
            else:
                col3.info("ℹ️ No hay subcategoría en tu market_intel.csv")

            st.divider()
            prefer_cols = [
                "marca", "categoria", sub_col if sub_col else None, "canal",
                "tendencia", "precio_promedio", "claim_top", "insight_consumidor",
                "crecimiento_categoria_pct", "competencia_intensidad", "precio_p25", "precio_p75",
                "tendencia_claim_top", "tendencia_pack_top", "share_lider_pct"
            ]
            show_cols = [c for c in prefer_cols if c and c in dfm.columns]
            st.dataframe(dfm[show_cols].head(200) if show_cols else dfm.head(200), use_container_width=True)

            st.subheader("🧠 Recomendaciones de Mercado")
            def safe_mean(col, default=None):
                return float(dfm[col].mean()) if col in dfm.columns and dfm[col].notna().any() else default
            def safe_mode(col, default="(sin dato)"):
                if col in dfm.columns:
                    s = dfm[col].dropna().astype(str)
                    if not s.empty:
                        return s.mode().iloc[0]
                return default

            growth = safe_mean("crecimiento_categoria_pct")
            comp = safe_mean("competencia_intensidad")
            p25 = safe_mean("precio_p25")
            p75 = safe_mean("precio_p75")
            claim_top = safe_mode("tendencia_claim_top", safe_mode("claim_top"))
            pack_top = safe_mode("tendencia_pack_top")

            recs = []
            if growth is not None:
                if growth > 12: recs.append("📈 Crecimiento alto → oportunidad clara.")
                elif growth < 0: recs.append("⚠️ Crecimiento negativo → validar con piloto.")
            if comp is not None and comp > 8:
                recs.append("🧱 Competencia alta → diferenciador fuerte (pack + claim + precio).")
            if p25 is not None and p75 is not None:
                recs.append(f"💰 Banda de precio (p25–p75): ${p25:.0f} — ${p75:.0f}")
            if claim_top and claim_top != "(sin dato)":
                recs.append(f"🏷 Claim/tema dominante: {claim_top}")
            if pack_top and pack_top != "(sin dato)":
                recs.append(f"📦 Pack tendencia: {pack_top}")

            if not recs: recs = ["No hay suficientes señales en el dataset para recomendar."]
            for r in recs: st.write("•", r)


# ============================================================
# 📄 REPORTE
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

    rows = []
    if st.session_state.last_sim: rows.append({"tipo":"simulador", **st.session_state.last_sim})
    if st.session_state.last_new: rows.append({"tipo":"producto_nuevo", **st.session_state.last_new})
    if st.session_state.last_invest: rows.append({"tipo":"inversionista", **st.session_state.last_invest})

    if rows:
        inputs_df = pd.json_normalize(rows)
        st.download_button("⬇️ Descargar Inputs CSV", df_to_csv_bytes(inputs_df), file_name="inputs_reporte.csv", key="dl_report_csv")
        st.dataframe(inputs_df.head(50), use_container_width=True)
    else:
        st.info("Corre Simulador / Producto Nuevo / Inversionista para generar inputs.")


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

