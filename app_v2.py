# app_V2.py
# ============================================================
# Plataforma IA: Producto + Empaque + Claims + Shelf 3-Second Test (v2.2)
# Incluye:
# ✅ Simulador (éxito + ventas) con pack+claims
# ✅ Insights
# ✅ Pack Vision+ (imagen -> métricas -> emoción -> quick wins)
# ✅ Claims Lab (librería ampliada)
# ✅ Experimentos A/B
# ✅ Vista Inversionista
# ✅ Motor de Recomendaciones (what-if optimizer)
# ✅ Reporte Ejecutivo descargable (TXT + CSV inputs)
# ✅ Shelf & Emotion Predictor (3-Second Test): ranking vs competidores + heatmap + multinomial + learning loop
# ✅ Producto Nuevo (Cold Start) + ✅ Recomendaciones accionables para subir probabilidad de éxito (what-if)
# ============================================================

import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, mean_absolute_error
from sklearn.linear_model import LogisticRegression

# ----------------------------
# Config
# ----------------------------
st.set_page_config(page_title="Plataforma IA | Producto + Empaque + Shelf 3-Second", layout="wide")

DATA_PATH_DEFAULT = "mercado_cereales_5000_con_ventas.csv"

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

def safe_percent(x):
    return f"{x*100:.2f}%"

def bar_from_value_counts(vc_series: pd.Series, title: str | None = None):
    """Convierte value_counts (Serie) a DF y grafica sin depender del nombre del índice."""
    if title:
        st.markdown(f"**{title}**")

    dfp = vc_series.reset_index()
    if dfp.shape[1] >= 2:
        dfp = dfp.iloc[:, :2].copy()
        dfp.columns = ["bucket", "count"]
    else:
        dfp = pd.DataFrame({"bucket": ["(sin buckets)"], "count": [0]})

    dfp["bucket"] = dfp["bucket"].astype(str)
    dfp["count"] = pd.to_numeric(dfp["count"], errors="coerce").fillna(0)
    st.bar_chart(dfp.set_index("bucket"), use_container_width=True)

# ----------------------------
# Image metrics (sin OCR pesado / sin OpenCV)
# ----------------------------
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

    pack_brand_contrast_score = clip(0.65 * m["contrast"] + 0.35 * m["pop_score"], 0, 1) * 100
    pack_text_overload_score = clip(m["edge_density"] / 0.35, 0, 1) * 100  # más edges ≈ más “ruido”

    return {
        "pack_legibility_score": round(legibility, 1),
        "pack_shelf_pop_score": round(shelf_pop, 1),
        "pack_clarity_score": round(clarity, 1),
        "pack_brand_contrast_score": round(pack_brand_contrast_score, 1),
        "pack_text_overload_score": round(pack_text_overload_score, 1),
    }

# ----------------------------
# Claims engine (base + extendida)
# ----------------------------
CLAIMS_LIBRARY = {
    "fit": [
        ("Alto en proteína", 0.90),
        ("Sin azúcar añadida", 0.88),
        ("Alto en fibra", 0.86),
        ("Integral", 0.80),
        ("Sin colorantes artificiales", 0.74),
        ("Con granos enteros", 0.72),
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
        ("Hecho con avena real", 0.78),
        ("Sabor intenso", 0.76),
        ("Calidad premium", 0.70),
        ("Receta artesanal", 0.64),
    ],
    "value": [
        ("Rinde más", 0.78),
        ("Gran sabor a mejor precio", 0.74),
        ("Ideal para la familia", 0.72),
        ("Económico y práctico", 0.66),
    ],
}

CANAL_CLAIM_BOOST = {
    "retail": {
        "Sin azúcar añadida": 1.04, "Alto en fibra": 1.03, "Integral": 1.02,
        "Ideal para la familia": 1.02, "Rinde más": 1.02
    },
    "marketplace": {
        "Alto en proteína": 1.05, "Sin colorantes artificiales": 1.04,
        "Ingredientes seleccionados": 1.03, "Hecho con avena real": 1.03
    },
}

EXTRA_CLAIMS = [
    ("Sin gluten", 0.76),
    ("Sin jarabe de maíz", 0.74),
    ("Endulzado naturalmente", 0.72),
    ("Bajo en sodio", 0.70),
    ("Fuente de energía", 0.69),
    ("Hecho con ingredientes naturales", 0.75),
    ("Sin sabores artificiales", 0.73),
    ("Con prebióticos", 0.71),
    ("Con avena integral", 0.74),
    ("Alto en calcio", 0.70),
]

def recommend_claims(segment: str, canal: str, max_claims: int = 6):
    seg = str(segment).lower().strip()
    canal = str(canal).lower().strip()
    items = (CLAIMS_LIBRARY.get(seg, []) + EXTRA_CLAIMS)[:]
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
    boosts = [CANAL_CLAIM_BOOST.get(canal, {}).get(str(c).strip(), 1.0) for c in selected_claims]
    base = float(np.mean(boosts))
    n = len(selected_claims)
    clarity_penalty = 1.0 if n <= 3 else max(0.65, 1.0 - 0.12 * (n - 3))
    score = 75 * base * clarity_penalty
    return float(np.clip(score, 0, 100))

# ----------------------------
# Emoción del empaque (solo con imagen)
# ----------------------------
def pack_emotion_from_image(scores: dict, raw_metrics: dict | None = None) -> dict:
    leg = clip(scores.get("pack_legibility_score", 0) / 100, 0, 1)
    pop = clip(scores.get("pack_shelf_pop_score", 0) / 100, 0, 1)
    clarity = clip(scores.get("pack_clarity_score", 0) / 100, 0, 1)
    brand = clip(scores.get("pack_brand_contrast_score", 60) / 100, 0, 1)
    overload = clip(scores.get("pack_text_overload_score", 45) / 100, 0, 1)

    brightness = None
    colorfulness = None
    if raw_metrics:
        brightness = raw_metrics.get("brightness", None)
        colorfulness = raw_metrics.get("colorfulness", None)

    confianza = 100 * clip(0.35*clarity + 0.30*leg + 0.20*brand + 0.15*(1-overload), 0, 1)

    premium = 100 * clip(0.40*clarity + 0.35*(1-overload) + 0.15*(1-abs(pop-0.55)) + 0.10*brand, 0, 1)

    energy_boost = 0.0
    if colorfulness is not None:
        energy_boost = clip(colorfulness / 0.20, 0, 1) * 0.10
    energia = 100 * clip(0.60*pop + 0.20*brand + 0.20*(1-abs(clarity-0.55)) + energy_boost, 0, 1)

    bright_fit = 0.0
    if brightness is not None:
        target = 0.55
        bright_fit = clip(1 - abs(brightness-target)/target, 0, 1) * 0.10
    salud = 100 * clip(0.45*clarity + 0.25*(1-overload) + 0.20*leg + 0.10*brand + bright_fit, 0, 1)

    ahorro = 100 * clip(0.40*clarity + 0.30*leg + 0.15*brand + 0.15*(1-clip(overload-0.3, 0, 1)), 0, 1)

    indulgencia = 100 * clip(0.55*pop + 0.20*brand + 0.15*(1-abs(overload-0.45)) + 0.10*(1-abs(clarity-0.55)), 0, 1)

    emotion_vector = {
        "confianza": round(confianza, 1),
        "salud": round(salud, 1),
        "energia": round(energia, 1),
        "premium": round(premium, 1),
        "ahorro": round(ahorro, 1),
        "indulgencia": round(indulgencia, 1),
    }
    emotion_label = max(emotion_vector, key=emotion_vector.get)

    vals = sorted(emotion_vector.values(), reverse=True)
    confidence = clip((vals[0] - vals[1]) / 25.0, 0, 1)

    rationale = []
    if clarity >= 0.70:
        rationale.append("Alta claridad visual → baja fricción cognitiva.")
    if leg >= 0.65:
        rationale.append("Buena legibilidad → más confianza.")
    if pop >= 0.70:
        rationale.append("Shelf pop alto → más energía/placer.")
    if overload >= 0.70:
        rationale.append("Ruido visual alto → baja premium/confianza.")
    if brand <= 0.55:
        rationale.append("Marca poco visible → baja recordación.")
    if not rationale:
        rationale = ["Balance visual estable: claridad, pop y jerarquía moderados."]

    return {
        "emotion_label": emotion_label,
        "emotion_vector": emotion_vector,
        "confidence": round(confidence, 2),
        "rationale": rationale[:4],
    }

def pack_emotion_score(pack_legibility, pack_pop, pack_clarity, claims_score_val, copy_tone: int):
    visual = 0.40 * (pack_pop / 100) + 0.30 * (pack_clarity / 100) + 0.15 * (pack_legibility / 100)
    claims = 0.15 * (claims_score_val / 100)
    tone_boost = 0.06 if copy_tone > 0 else (-0.06 if copy_tone < 0 else 0.0)
    score = (visual + claims + tone_boost) * 100
    return float(np.clip(score, 0, 100))

# ============================================================
# ✅ Recomendaciones para Producto Nuevo (Cold Start) — WHAT-IF
# ============================================================
def _predict_success_prob(success_model, row: dict) -> float:
    X = pd.DataFrame([row])
    return float(success_model.predict_proba(X)[0][1])

def _predict_sales_units(sales_model, row: dict) -> float:
    X = pd.DataFrame([row])
    return float(sales_model.predict(X)[0])

def coldstart_recommendations(success_model, sales_model, base_row: dict):
    """
    Genera recomendaciones accionables para subir probabilidad de éxito:
    - Barrido what-if en precio/margen y proxies (claims/pack) vía conexion_score.
    """
    base_prob = _predict_success_prob(success_model, base_row)
    base_sales = _predict_sales_units(sales_model, base_row)

    price_grid = [0.90, 0.95, 1.00, 1.05, 1.10]  # +/-10%
    margin_grid = [-5, 0, +5, +10]               # puntos de margen
    claims_grid = [-10, 0, +10, +20]             # puntos proxy
    pack_grid = [-10, 0, +10, +20]               # puntos proxy

    rows = []
    for pm in price_grid:
        for dm in margin_grid:
            for dc in claims_grid:
                for dp in pack_grid:
                    r = dict(base_row)
                    r["precio"] = float(base_row["precio"]) * pm
                    r["margen_pct"] = float(np.clip(float(base_row["margen_pct"]) + dm, 0, 90))

                    improved_conn = float(base_row["conexion_score"]) + 0.20*dc + 0.30*dp
                    r["conexion_score"] = float(np.clip(improved_conn, 0, 100))

                    prob = _predict_success_prob(success_model, r)
                    sales = _predict_sales_units(sales_model, r)

                    rows.append({
                        "precio": r["precio"],
                        "margen_pct": r["margen_pct"],
                        "delta_claims_proxy": dc,
                        "delta_pack_proxy": dp,
                        "prob_exito": prob,
                        "ventas_unidades": sales,
                        "uplift_prob_pp": (prob - base_prob) * 100,
                        "uplift_sales": (sales - base_sales),
                    })

    out = pd.DataFrame(rows).sort_values(["prob_exito", "ventas_unidades"], ascending=False).head(12).copy()
    best = out.iloc[0]

    recs = []
    if best["precio"] < base_row["precio"]:
        recs.append("💲 Baja precio ligeramente (o mejora valor percibido) para subir prob. de éxito.")
    elif best["precio"] > base_row["precio"]:
        recs.append("💲 Puedes subir precio si aumentas conexión (pack/claims).")

    if best["margen_pct"] > base_row["margen_pct"]:
        recs.append("📈 Sube margen si mantienes conexión fuerte (pack + claims claros).")
    elif best["margen_pct"] < base_row["margen_pct"]:
        recs.append("📉 Baja margen para ganar competitividad (si el canal lo permite).")

    if best["delta_claims_proxy"] >= 10:
        recs.append("🏷️ Mejora claims: quédate con 2–3 claims TOP alineados al segmento/canal.")
    else:
        recs.append("🏷️ Claims en rango: evita saturar (demasiados claims bajan claridad).")

    if best["delta_pack_proxy"] >= 10:
        recs.append("📦 Empaque: sube atención/claridad (3-second) — es la palanca más fuerte para subir éxito.")
    else:
        recs.append("📦 Empaque en rango: enfócate más en precio o claims.")

    summary = {
        "base_prob_%": round(base_prob*100, 1),
        "best_prob_%": round(float(best["prob_exito"])*100, 1),
        "uplift_prob_pp": round(float(best["uplift_prob_pp"]), 1),
        "base_sales": round(float(base_sales), 0),
        "best_sales": round(float(best["ventas_unidades"]), 0),
        "uplift_sales": round(float(best["uplift_sales"]), 0),
    }
    return out, recs, summary

# ----------------------------
# Data loading
# ----------------------------
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

    df = df.dropna(
        subset=[
            "marca", "categoria", "canal", "precio", "competencia", "demanda", "tendencia",
            "margen_pct", "conexion_score", "rating_conexion", "sentiment_score", "exito"
        ]
    )
    df["exito"] = df["exito"].astype(int)
    return df

# ----------------------------
# Models
# ----------------------------
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

    clf = Pipeline(
        steps=[
            ("preprocessor", pre),
            ("model", RandomForestClassifier(
                n_estimators=350, random_state=42, class_weight="balanced_subsample"
            )),
        ]
    )

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
    missing_sales = sorted(list(REQUIRED_SALES - set(df.columns)))
    if missing_sales:
        raise ValueError(f"Este CSV no trae ventas. Faltan: {missing_sales}. Usa mercado_cereales_5000_con_ventas.csv")

    features = [
        "precio", "competencia", "demanda", "tendencia", "margen_pct",
        "conexion_score", "rating_conexion", "sentiment_score",
        "marca", "canal"
    ]
    X = df[features]
    y = df["ventas_unidades"].astype(float)

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

    reg = Pipeline(
        steps=[
            ("preprocessor", pre),
            ("model", RandomForestRegressor(n_estimators=350, random_state=42)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    reg.fit(X_train, y_train)

    pred = reg.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    return reg, mae

# ============================================================
# Shelf & Emotion Predictor (3-Second Test)
# ============================================================
def saliency_heatmap(img: Image.Image, downscale: int = 2) -> np.ndarray:
    im = img.convert("RGB")
    arr = np.asarray(im).astype(np.float32) / 255.0
    if downscale > 1:
        arr = arr[::downscale, ::downscale, :]

    gray = 0.299 * arr[...,0] + 0.587 * arr[...,1] + 0.114 * arr[...,2]

    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:,1:-1] = gray[:,2:] - gray[:,:-2]
    gy[1:-1,:] = gray[2:,:] - gray[:-2,:]
    mag = np.sqrt(gx**2 + gy**2)

    pad = np.pad(gray, 1, mode="edge")
    local_std = np.zeros_like(gray)
    for dy in [-1,0,1]:
        for dx in [-1,0,1]:
            local_std += (pad[1+dy:1+dy+gray.shape[0], 1+dx:1+dx+gray.shape[1]] - gray)**2
    local_std = np.sqrt(local_std/9.0)

    cvar = np.std(arr[...,0]-arr[...,1]) + 0.3*np.std(0.5*(arr[...,0]+arr[...,1])-arr[...,2])

    heat = 0.65*mag + 0.35*local_std
    heat = heat / (np.max(heat) + 1e-9)
    heat = np.clip(heat + 0.10*np.clip(cvar/0.15,0,1), 0, 1)
    return heat

def overlay_heatmap(img: Image.Image, heat: np.ndarray, alpha: float = 0.45) -> plt.Figure:
    fig = plt.figure(figsize=(6,6), dpi=140)
    ax = fig.add_subplot(111)
    ax.imshow(img.convert("RGB"))
    ax.imshow(heat, cmap="jet", alpha=alpha, interpolation="bilinear")
    ax.axis("off")
    return fig

def softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / (np.sum(ex) + 1e-9)

def pack_feature_vector(scores: dict, emo: dict, raw_metrics: dict) -> dict:
    return {
        "attention": float(scores.get("pack_shelf_pop_score", 0)),
        "legibility": float(scores.get("pack_legibility_score", 0)),
        "clarity": float(scores.get("pack_clarity_score", 0)),
        "emotion_dom": float(emo["emotion_vector"][emo["emotion_label"]]),
        "contrast": float(raw_metrics.get("contrast", 0)),
        "colorfulness": float(raw_metrics.get("colorfulness", 0)),
        "edge_density": float(raw_metrics.get("edge_density", 0)),
        "brightness": float(raw_metrics.get("brightness", 0)),
        "choice_proxy": float(raw_metrics.get("pop_score", 0))*100.0,
    }

def shelf_3sec_predictor(scores: dict, emo: dict, raw_metrics: dict) -> dict:
    pop = clip(scores.get("pack_shelf_pop_score", 0)/100, 0, 1)
    clarity = clip(scores.get("pack_clarity_score", 0)/100, 0, 1)
    leg = clip(scores.get("pack_legibility_score", 0)/100, 0, 1)
    brand = clip(scores.get("pack_brand_contrast_score", 60)/100, 0, 1)
    overload = clip(scores.get("pack_text_overload_score", 45)/100, 0, 1)

    contrast = clip(raw_metrics.get("contrast", 0), 0, 1)
    colorfulness = clip(raw_metrics.get("colorfulness", 0)/0.20, 0, 1)
    brightness = raw_metrics.get("brightness", 0.55)

    attention = 100 * clip(0.45*pop + 0.25*contrast + 0.20*colorfulness + 0.10*brand, 0, 1)
    emotion = float(emo["emotion_vector"][emo["emotion_label"]])
    memorability = 100 * clip(0.40*brand + 0.30*clarity + 0.20*pop + 0.10*(1-overload), 0, 1)

    choice = 100 * clip(
        0.35*(attention/100) +
        0.30*(emotion/100) +
        0.20*(memorability/100) +
        0.15*clarity,
        0, 1
    )

    if choice >= 78:
        verdict = "🔥 3-Second Verdict: GANA ANAQUEL"
    elif choice >= 62:
        verdict = "👍 3-Second Verdict: COMPETITIVO (optimizable)"
    else:
        verdict = "⚠️ 3-Second Verdict: PIERDE EN 3 SEGUNDOS (mejorar)"

    wins = []
    if attention < 65:
        wins.append("Subir atención: aumenta contraste y un color acento (sin saturar).")
    if leg < 62:
        wins.append("Subir legibilidad: texto más grueso + mejor contraste + jerarquía.")
    if clarity < 62:
        wins.append("Subir claridad: menos elementos + 1 beneficio dominante + 2–3 claims máximo.")
    if overload > 0.70:
        wins.append("Reducir ruido: elimina microtextos/patrones; deja aire alrededor de mensajes clave.")
    if brightness < 0.40 or brightness > 0.75:
        wins.append("Balancear brillo: evita pack muy oscuro o lavado; mejora lectura en anaquel.")
    if brand < 0.60:
        wins.append("Aumentar presencia de marca: logo más visible o mejor contraste.")
    if not wins:
        wins = ["Pack sólido: corre A/B de 2 variantes (color acento vs jerarquía) y mide elección."]

    return {
        "attention": round(attention, 1),
        "emotion_label": emo["emotion_label"],
        "emotion": round(emotion, 1),
        "memorability": round(memorability, 1),
        "choice": round(choice, 1),
        "verdict": verdict,
        "quick_wins": wins[:5],
    }

# ----------------------------
# Sidebar: load CSV robust
# ----------------------------
st.sidebar.title("⚙️ Datos")
uploaded = st.sidebar.file_uploader("Sube tu CSV (con ventas)", type=["csv"], key="uploader_csv_v22")

if uploaded is not None:
    df = load_data(uploaded)
else:
    if Path(DATA_PATH_DEFAULT).exists():
        df = load_data(DATA_PATH_DEFAULT)
    else:
        st.warning(f"❗No encontré '{DATA_PATH_DEFAULT}' en el repo. Sube el CSV para arrancar.")
        st.stop()

# Train models
try:
    success_model, acc, auc, cm = train_success_model(df)
    sales_model, mae = train_sales_model(df)
except Exception as e:
    st.error(f"Error entrenando modelos: {e}")
    st.stop()

# ----------------------------
# Session state setup
# ----------------------------
if "last_run" not in st.session_state:
    st.session_state.last_run = {}
if "last_pack" not in st.session_state:
    st.session_state.last_pack = {}
if "last_new" not in st.session_state:
    st.session_state.last_new = {}
if "experiments" not in st.session_state:
    st.session_state.experiments = []
if "choice_labels" not in st.session_state:
    st.session_state.choice_labels = []

# ----------------------------
# Header
# ----------------------------
st.title("🧠 Plataforma IA: Producto + Empaque + Claims + Shelf Intelligence")
st.caption("Éxito + Ventas estimadas + Insights + Pack Vision+ + Claims + Experimentos + 3-Second Test + Producto Nuevo")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Registros", f"{len(df):,}")
k2.metric("Precisión", f"{acc * 100:.2f}%")
k3.metric("AUC", f"{auc:.3f}")
k4.metric("Éxito base", f"{df['exito'].mean() * 100:.1f}%")
k5.metric("MAE ventas", f"{mae:,.0f} u.")
st.divider()

# ----------------------------
# Tabs
# ----------------------------
tabs = st.tabs(
    [
        "🧪 Simulador",
        "📊 Insights",
        "📦 Pack Lab",
        "🏷️ Claims Lab",
        "🧪 Experimentos",
        "💼 Inversionista",
        "🧠 Recomendaciones",
        "📄 Reporte Ejecutivo",
        "📂 Datos",
        "🧠 Modelo",
        "👀 Shelf & Emotion Predictor",
        "🧊 Producto Nuevo",
    ]
)

tab_sim, tab_ins, tab_pack, tab_claims, tab_exp, tab_inv, tab_rec, tab_report, tab_data, tab_model, tab_shelf, tab_new = tabs

# ============================================================
# 🧪 Simulador
# ============================================================
with tab_sim:
    st.subheader("🧪 Simulador What-If (incluye empaque + claims)")
    marcas = sorted(df["marca"].unique().tolist())
    canales = sorted(df["canal"].unique().tolist())

    c1, c2, c3 = st.columns(3)
    marca = c1.selectbox("Marca", marcas, 0, key="sim_marca_v22")
    canal = c2.selectbox("Canal", canales, 0, key="sim_canal_v22")
    segmento = c3.selectbox("Segmento objetivo", ["fit", "kids", "premium", "value"], 0, key="sim_segmento_v22")

    st.markdown("### Variables de negocio")
    b1, b2, b3, b4, b5 = st.columns(5)
    precio = b1.number_input("Precio", value=float(df["precio"].median()), step=1.0, key="sim_precio_v22")
    competencia = b2.slider("Competencia (1-10)", 1, 10, int(df["competencia"].median()), key="sim_comp_v22")
    demanda = b3.slider("Demanda (10-100)", 10, 100, int(df["demanda"].median()), key="sim_dem_v22")
    tendencia = b4.slider("Tendencia (20-100)", 20, 100, int(df["tendencia"].median()), key="sim_tend_v22")
    margen_pct = b5.slider("Margen %", 0, 90, int(np.clip(df["margen_pct"].median(), 0, 90)), key="sim_margen_v22")

    st.markdown("### Empaque + Claims")
    p1, p2, p3 = st.columns(3)
    pack_legibility_score = p1.slider("Pack legibilidad (0-100)", 0, 100, 65, key="sim_pack_leg_v22")
    pack_shelf_pop_score = p2.slider("Pack shelf pop (0-100)", 0, 100, 70, key="sim_pack_pop_v22")
    pack_clarity_score = p3.slider("Pack claridad (0-100)", 0, 100, 65, key="sim_pack_cla_v22")

    recs = recommend_claims(segmento, canal, max_claims=10)
    claim_options = [c for c, _ in recs]
    selected_claims = st.multiselect(
        "Selecciona claims (ideal 2-3)",
        claim_options,
        default=claim_options[:2],
        key="sim_claims_v22",
    )

    copy = st.text_input("Copy corto (opcional)", value="Energía y nutrición para tu día", key="sim_copy_v22")
    pos_kw = ["energía", "nutrición", "saludable", "delicioso", "me encanta", "premium", "calidad", "proteína", "fibra"]
    neg_kw = ["caro", "no", "malo", "rechazo", "no me gusta", "pésimo", "horrible"]
    t = copy.lower()
    tone = 0
    if any(k in t for k in pos_kw):
        tone += 1
    if any(k in t for k in neg_kw):
        tone -= 1
    copy_tone = 1 if tone > 0 else (-1 if tone < 0 else 0)

    cscore = claims_score(selected_claims, canal)
    pem = pack_emotion_score(pack_legibility_score, pack_shelf_pop_score, pack_clarity_score, cscore, copy_tone)
    uplift = clip((pem - 50) / 50, -0.35, 0.35)

    rating_conexion = st.slider("Rating conexión producto (1-10)", 1, 10, 7, key="sim_rating_v22")
    sentiment_score = st.select_slider("Sentimiento del producto (-1/0/1)", options=[-1, 0, 1], value=1, key="sim_sent_v22")

    base_conexion = (rating_conexion / 10) * 70 + sentiment_score * 15 + 5
    conexion_score = clip(base_conexion * (1 + uplift), 0, 100)

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

    s1, s2, s3 = st.columns(3)
    s1.metric("Claims Score", f"{cscore:.1f}/100")
    s2.metric("Emotion Pack Score", f"{pem:.1f}/100")
    s3.metric("Conexión final", f"{conexion_score:.1f}/100")

    if st.button("🚀 Simular", key="sim_btn_v22"):
        p = float(success_model.predict_proba(entrada)[0][1])
        pred = int(success_model.predict(entrada)[0])

        ventas = max(0, round(float(sales_model.predict(entrada)[0])))
        ingresos = ventas * float(precio)
        utilidad = ventas * (float(precio) * (float(margen_pct) / 100.0))

        r1, r2, r3 = st.columns(3)
        r1.metric("Prob. éxito", safe_percent(p))
        r2.metric("Predicción", "✅ Éxito" if pred else "⚠️ Riesgo")
        r3.metric("Ventas (unidades)", f"{ventas:,.0f}")

        r4, r5 = st.columns(2)
        r4.metric("Ingresos ($)", f"${ingresos:,.0f}")
        r5.metric("Utilidad ($)", f"${utilidad:,.0f}")

        st.session_state.last_run = {
            "marca": marca, "canal": canal, "segmento": segmento,
            "precio": float(precio), "competencia": float(competencia),
            "demanda": float(demanda), "tendencia": float(tendencia),
            "margen_pct": float(margen_pct),
            "claims": selected_claims,
            "claims_score": float(cscore),
            "pack_emotion_score": float(pem),
            "conexion_score": float(conexion_score),
            "prob_exito": float(p),
            "pred_exito": int(pred),
            "ventas_pred": int(ventas),
            "ingresos_pred": float(ingresos),
            "utilidad_pred": float(utilidad),
        }

        st.dataframe(entrada, use_container_width=True)

# ============================================================
# 📊 Insights
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

        st.markdown("**Ranking por marca (Ventas promedio)**")
        v_marca = df.groupby("marca")[["ventas_unidades"]].mean().sort_values("ventas_unidades", ascending=False).round(0)
        st.dataframe(v_marca, use_container_width=True)

    with right:
        st.markdown("**Marca + Canal (Conexión promedio)**")
        ins_mc = df.groupby(["marca","canal"])[["conexion_score"]].mean().sort_values("conexion_score", ascending=False).round(2)
        st.dataframe(ins_mc.head(25), use_container_width=True)

        st.markdown("**Marca + Canal (Éxito %)**")
        ex_mc = df.groupby(["marca","canal"])[["exito"]].mean().sort_values("exito", ascending=False).round(3)
        ex_mc["exito_%"] = (ex_mc["exito"] * 100).round(1)
        st.dataframe(ex_mc.head(25)[["exito_%"]], use_container_width=True)

        st.markdown("**Marca + Canal (Ventas promedio)**")
        v_mc = df.groupby(["marca","canal"])[["ventas_unidades"]].mean().sort_values("ventas_unidades", ascending=False).round(0)
        st.dataframe(v_mc.head(25), use_container_width=True)

    st.divider()
    d1, d2 = st.columns(2)

    with d1:
        bins = pd.cut(df["conexion_score"], bins=[0, 20, 40, 60, 80, 100], include_lowest=True)
        dist = bins.value_counts().sort_index()
        bar_from_value_counts(dist, title="Distribución: Conexión emocional (bucket)")

    with d2:
        bins2 = pd.cut(df["ventas_unidades"].clip(0, 40000), bins=[0, 2000, 5000, 10000, 20000, 40000], include_lowest=True)
        dist2 = bins2.value_counts().sort_index()
        bar_from_value_counts(dist2, title="Distribución: Ventas unidades (bucket)")

# ============================================================
# 📦 Pack Lab
# ============================================================
with tab_pack:
    st.subheader("📦 Pack Vision+ (sube tu empaque y te doy emoción + quick wins)")

    img_file = st.file_uploader("Sube imagen del empaque (PNG/JPG)", type=["png", "jpg", "jpeg"], key="pack_uploader_v22")
    if img_file is None:
        st.info("Sube una imagen para generar análisis del empaque.")
    else:
        img = Image.open(img_file)
        st.image(img, caption="Empaque cargado", use_container_width=True)

        m = image_metrics(img)
        scores = pack_scores_from_metrics(m)
        emo = pack_emotion_from_image(scores, raw_metrics=m)
        pred = shelf_3sec_predictor(scores, emo, m)

        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Brillo", f"{m['brightness']:.2f}")
        a2.metric("Contraste", f"{m['contrast']:.2f}")
        a3.metric("Colorfulness", f"{m['colorfulness']:.2f}")
        a4.metric("Edge density", f"{m['edge_density']:.3f}")

        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Legibilidad", f"{scores['pack_legibility_score']}/100")
        b2.metric("Shelf Pop", f"{scores['pack_shelf_pop_score']}/100")
        b3.metric("Claridad", f"{scores['pack_clarity_score']}/100")
        b4.metric("Texto/Ruido", f"{scores['pack_text_overload_score']}/100")

        st.subheader("🧠 Emoción que provoca (solo por imagen)")
        c1, c2 = st.columns(2)
        c1.metric("Emoción dominante", emo["emotion_label"].upper())
        c2.metric("Confianza del diagnóstico", f"{emo['confidence']*100:.0f}%")

        st.markdown("**Mapa emocional (0–100)**")
        st.dataframe(pd.DataFrame([emo["emotion_vector"]]).T.rename(columns={0: "score"}), use_container_width=True)

        st.markdown("**Drivers (por qué)**")
        for b in emo["rationale"]:
            st.write(f"• {b}")

        st.markdown("### ⚡ 3-Second Verdict (pack solo)")
        st.success(pred["verdict"])
        st.markdown("**Top Quick Wins**")
        for w in pred["quick_wins"]:
            st.write(f"• {w}")

        st.session_state.last_pack = {"pack_scores": scores, "emotion": emo, "shelf_pred": pred}

# ============================================================
# 🏷️ Claims Lab
# ============================================================
with tab_claims:
    st.subheader("🏷️ Claims Lab (claims ganadores)")

    c1, c2 = st.columns(2)
    seg = c1.selectbox("Segmento", ["fit", "kids", "premium", "value"], 0, key="claims_segmento_v22")
    canal_c = c2.selectbox("Canal", ["retail", "marketplace"], 0, key="claims_canal_v22")

    recs = recommend_claims(seg, canal_c, max_claims=12)
    rec_df = pd.DataFrame(recs, columns=["claim", "score_base"])
    rec_df["score_base"] = (rec_df["score_base"] * 100).round(1)
    st.dataframe(rec_df, use_container_width=True)

    selected = st.multiselect(
        "Selecciona 2-3 claims",
        rec_df["claim"].tolist(),
        default=rec_df["claim"].tolist()[:2],
        key="claims_selected_v22"
    )
    cscore = claims_score(selected, canal_c)
    st.metric("Claims Score", f"{cscore:.1f}/100")
    st.warning("Nota: recomendación comercial (no legal/regulatoria).")

# ============================================================
# 🧪 Experimentos (A/B)
# ============================================================
with tab_exp:
    st.subheader("🧪 Experimentos (A/B)")

    c1, c2, c3 = st.columns(3)
    exp_name = c1.text_input("Nombre experimento", value="Test Pack v1 vs v2", key="exp_name_v22")
    variant = c2.selectbox("Variante", ["A", "B"], 0, key="exp_variant_v22")
    metric = c3.selectbox("Métrica", ["intencion_compra", "conexion_pack", "ventas_piloto"], 0, key="exp_metric_v22")

    v1, v2, v3 = st.columns(3)
    marca_e = v1.selectbox("Marca", sorted(df["marca"].unique().tolist()), 0, key="exp_marca_v22")
    canal_e = v2.selectbox("Canal", sorted(df["canal"].unique().tolist()), 0, key="exp_canal_v22")
    value = v3.number_input("Valor observado", value=7.0, step=0.1, key="exp_value_v22")

    if st.button("➕ Guardar medición", key="exp_save_v22"):
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
            st.dataframe(pivot.round(3), use_container_width=True)
        except Exception:
            st.info("Necesitas registros en A y B para calcular lift.")
    else:
        st.info("Aún no hay experimentos guardados.")

# ============================================================
# 💼 Vista Inversionista
# ============================================================
with tab_inv:
    st.subheader("💼 Vista Inversionista (TAM + escenarios + upside + unit economics)")

    colA, colB, colC, colD = st.columns(4)
    tam = colA.number_input("TAM (MXN)", value=8_000_000_000, step=100_000_000, key="inv_tam_v22")
    som_pct = colB.slider("SOM % (capturable)", 0.1, 10.0, 1.5, 0.1, key="inv_som_v22")
    asp_pct = colC.slider("Aspiracional % (3 años)", 0.1, 20.0, 4.0, 0.1, key="inv_asp_v22")
    gross_margin = colD.slider("Margen bruto %", 10, 80, 45, key="inv_gm_v22")

    base = tam * (som_pct/100)
    upside = tam * (asp_pct/100)

    s1, s2, s3 = st.columns(3)
    s1.metric("SOM anual (base)", f"${base:,.0f}")
    s2.metric("Aspiracional (3y)", f"${upside:,.0f}")
    s3.metric("Upside incremental", f"${(upside-base):,.0f}")

    st.markdown("### Unit Economics (simplificado)")
    u1, u2, u3, u4 = st.columns(4)
    price = u1.number_input("Precio promedio (MXN)", value=120.0, step=1.0, key="inv_price_v22")
    cogs = u2.number_input("COGS (MXN)", value=70.0, step=1.0, key="inv_cogs_v22")
    cac = u3.number_input("CAC por cliente (MXN)", value=25_000.0, step=1000.0, key="inv_cac_v22")
    arpa = u4.number_input("ARPA anual (MXN)", value=150_000.0, step=5000.0, key="inv_arpa_v22")

    gross = max(0.0, price - cogs)
    gross_pct = (gross/price)*100 if price > 0 else 0
    payback = (cac/arpa)*12 if arpa > 0 else 999

    m1, m2, m3 = st.columns(3)
    m1.metric("Gross profit / unidad", f"${gross:,.1f}")
    m2.metric("Gross margin %", f"{gross_pct:.1f}%")
    m3.metric("Payback (meses)", f"{payback:.1f}")

    st.markdown("### Score lanzamiento (mix: modelo + pack + claims)")
    lr = st.session_state.last_run or {}
    pack = st.session_state.last_pack.get("shelf_pred", {})
    score_lanz = 0.0
    if lr:
        score_lanz = (
            0.45 * (lr.get("prob_exito", 0)*100) +
            0.25 * (pack.get("choice", 60)) +
            0.15 * (lr.get("claims_score", 50)) +
            0.15 * (lr.get("pack_emotion_score", 60))
        )
    st.metric("Launch Score (0–100)", f"{score_lanz:.1f}/100")

# ============================================================
# 🧠 Motor de Recomendaciones (what-if optimizer)
# ============================================================
with tab_rec:
    st.subheader("🧠 Motor de Recomendaciones (optimiza precio + margen + claims + pack)")

    lr = st.session_state.last_run or {}
    if not lr:
        st.info("Primero corre una simulación en el tab 🧪 Simulador para usarlo como base.")
    else:
        base_precio = float(lr["precio"])
        base_margen = float(lr["margen_pct"])
        base_pack_emotion = float(lr.get("pack_emotion_score", 60))

        c1, c2, c3 = st.columns(3)
        price_range = c1.slider("Rango precio (±%)", 0, 40, 15, key="opt_price_rng_v22")
        margin_range = c2.slider("Rango margen (± puntos)", 0, 30, 10, key="opt_margin_rng_v22")
        tries = c3.slider("Iteraciones", 50, 400, 200, 50, key="opt_tries_v22")

        seg = lr.get("segmento", "fit")
        canal = lr.get("canal", "retail")
        claim_pool = [c for c, _ in recommend_claims(seg, canal, 12)]

        rng = np.random.default_rng(42)
        best = None
        rows = []

        for _ in range(int(tries)):
            p = base_precio * (1 + rng.uniform(-price_range/100, price_range/100))
            mrg = clip(base_margen + rng.uniform(-margin_range, margin_range), 0, 90)

            k = int(rng.integers(2, 4))
            claims_try = list(rng.choice(claim_pool, size=k, replace=False))
            cscore = claims_score(claims_try, canal)

            pack_choice = st.session_state.last_pack.get("shelf_pred", {}).get("choice", 60.0)
            pack_em = clip(0.6*base_pack_emotion + 0.4*pack_choice, 0, 100)

            entrada = pd.DataFrame([{
                "precio": float(p),
                "competencia": float(lr["competencia"]),
                "demanda": float(lr["demanda"]),
                "tendencia": float(lr["tendencia"]),
                "margen_pct": float(mrg),
                "conexion_score": float(lr["conexion_score"]),
                "rating_conexion": float(7),
                "sentiment_score": float(1),
                "marca": str(lr["marca"]).lower(),
                "canal": str(lr["canal"]).lower(),
            }])

            prob = float(success_model.predict_proba(entrada)[0][1]) * 100
            ventas = max(0, float(sales_model.predict(entrada)[0]))
            utilidad = ventas * (p * (mrg/100.0))

            launch = 0.50*prob + 0.20*pack_choice + 0.15*cscore + 0.15*pack_em
            row = {
                "launch_score": round(launch, 2),
                "prob_exito_%": round(prob, 2),
                "ventas_pred": round(ventas, 0),
                "utilidad_pred": round(utilidad, 0),
                "precio": round(p, 1),
                "margen_pct": round(mrg, 1),
                "claims": ", ".join(claims_try),
                "claims_score": round(cscore, 1),
                "pack_choice": round(pack_choice, 1),
            }
            rows.append(row)

            if best is None or row["launch_score"] > best["launch_score"]:
                best = row

        out = pd.DataFrame(rows).sort_values("launch_score", ascending=False).head(25)
        st.dataframe(out, use_container_width=True)

        st.markdown("### ✅ Mejor recomendación")
        st.json(best)

# ============================================================
# 👀 Shelf & Emotion Predictor (3-Second Test)
# ============================================================
with tab_shelf:
    st.subheader("👀 Shelf & Emotion Predictor (3-Second Test)")

    files = st.file_uploader(
        "Sube 2–10 imágenes (tu pack + competidores)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="shelf_multi_upload_v22",
    )

    show_heat = st.checkbox("Mostrar heatmap (saliency) por pack", value=True, key="shelf_show_heat_v22")
    alpha = st.slider("Intensidad heatmap", 0.15, 0.75, 0.45, 0.05, key="shelf_heat_alpha_v22")

    if not files or len(files) < 2:
        st.info("Sube al menos 2 imágenes para ranking vs competencia.")
    else:
        results = []
        thumbs = []

        for i, f in enumerate(files):
            img = Image.open(f)
            m = image_metrics(img)
            scores = pack_scores_from_metrics(m)
            emo = pack_emotion_from_image(scores, m)
            pred = shelf_3sec_predictor(scores, emo, m)
            feats = pack_feature_vector(scores, emo, m)

            row = {
                "id": f"pack_{i+1}",
                "filename": getattr(f, "name", f"pack_{i+1}"),
                "attention": pred["attention"],
                "emotion_label": pred["emotion_label"],
                "emotion": pred["emotion"],
                "memorability": pred["memorability"],
                "choice": pred["choice"],
            }
            row.update({f"feat_{k}": v for k, v in feats.items()})
            results.append(row)
            thumbs.append((row["id"], row["filename"], img, m, scores, emo, pred))

        dfc = pd.DataFrame(results)
        dfc["rank_choice"] = dfc["choice"].rank(ascending=False, method="min").astype(int)
        dfc = dfc.sort_values(["choice", "attention"], ascending=False).reset_index(drop=True)

        st.markdown("### 🏁 Ranking vs competencia")
        st.dataframe(
            dfc[["rank_choice", "filename", "attention", "emotion_label", "emotion", "memorability", "choice"]],
            use_container_width=True,
        )

        st.markdown("### 🛒 Simulación multinomial (probabilidad relativa de elección)")
        probs = softmax(dfc["choice"].values)
        dfp = dfc[["filename", "choice"]].copy()
        dfp["prob_elegido_%"] = (probs * 100).round(1)
        dfp = dfp.sort_values("prob_elegido_%", ascending=False)
        st.dataframe(dfp, use_container_width=True)

        top = dfc.iloc[0]
        st.success(f"⚡ 3-Second Verdict (Top): **{top['filename']}** | elección **{top['choice']:.1f}**")

        st.markdown("### 🔧 Top 5 Quick Wins (del ganador)")
        for (_, fname, _, _, _, _, pred) in thumbs:
            if fname == top["filename"]:
                for w in pred["quick_wins"]:
                    st.write("•", w)
                break

        st.divider()
        st.markdown("### 🖼️ Visual + Heatmap de atención")

        names = [t[1] for t in thumbs]
        sel = st.selectbox("Selecciona pack para ver detalle", names, index=0, key="shelf_sel_pack_v22")

        colL, colR = st.columns(2)
        for (_, fname, img, m, scores, emo, pred) in thumbs:
            if fname == sel:
                with colL:
                    st.image(img, caption=f"{fname}", use_container_width=True)
                    st.write(f"**Verdict:** {pred['verdict']}")
                    kk1, kk2, kk3, kk4 = st.columns(4)
                    kk1.metric("Atención", pred["attention"])
                    kk2.metric("Emoción", f"{pred['emotion_label']} ({pred['emotion']})")
                    kk3.metric("Recordación", pred["memorability"])
                    kk4.metric("Elección", pred["choice"])
                with colR:
                    if show_heat:
                        heat = saliency_heatmap(img, downscale=2)
                        fig = overlay_heatmap(img, heat, alpha=alpha)
                        st.pyplot(fig, clear_figure=True)
                    else:
                        st.info("Activa el heatmap para ver mapa de atención.")
                st.markdown("**Mapa emocional (0–100)**")
                st.dataframe(pd.DataFrame([emo["emotion_vector"]]).T.rename(columns={0: "score"}), use_container_width=True)
                break

        st.divider()
        st.markdown("### 🧪 Learning con resultados reales (feedback loop)")

        winner = st.selectbox("¿Cuál pack fue elegido (ganador real)?", options=names, index=0, key="shelf_winner_select_v22")
        context = st.text_input("Contexto (opcional): canal / tienda / público / fecha", value="", key="shelf_context_v22")

        if st.button("➕ Guardar resultado real", key="shelf_save_label_v22"):
            for row in results:
                y = 1 if row["filename"] == winner else 0
                st.session_state.choice_labels.append({
                    "winner_filename": winner,
                    "filename": row["filename"],
                    "y_chosen": y,
                    "context": context,
                    "attention": row["feat_attention"],
                    "legibility": row["feat_legibility"],
                    "clarity": row["feat_clarity"],
                    "emotion_dom": row["feat_emotion_dom"],
                    "contrast": row["feat_contrast"],
                    "colorfulness": row["feat_colorfulness"],
                    "edge_density": row["feat_edge_density"],
                    "brightness": row["feat_brightness"],
                    "choice_proxy": row["feat_choice_proxy"],
                })
            st.success("Guardado ✅ (ya estás alimentando el feedback loop).")

        if st.session_state.choice_labels:
            lab = pd.DataFrame(st.session_state.choice_labels)
            st.dataframe(lab.tail(30), use_container_width=True)

            st.download_button(
                "📥 Descargar learning dataset (CSV)",
                data=lab.to_csv(index=False).encode("utf-8"),
                file_name="shelf_choice_learning.csv",
                mime="text/csv",
                key="shelf_dl_learning_v22",
            )

            st.markdown("#### 🧠 Modelo de elección (entrenado con tus datos reales)")
            if len(lab) >= 60 and lab["y_chosen"].nunique() == 2:
                X = lab[["attention","legibility","clarity","emotion_dom","contrast","colorfulness","edge_density","brightness","choice_proxy"]].copy()
                y = lab["y_chosen"].astype(int)

                clf = LogisticRegression(max_iter=200, solver="lbfgs")
                clf.fit(X, y)

                X_now = dfc[
                    ["feat_attention","feat_legibility","feat_clarity","feat_emotion_dom",
                     "feat_contrast","feat_colorfulness","feat_edge_density","feat_brightness","feat_choice_proxy"]
                ].copy()
                X_now.columns = ["attention","legibility","clarity","emotion_dom","contrast","colorfulness","edge_density","brightness","choice_proxy"]

                p = clf.predict_proba(X_now)[:,1]
                df_learn = dfc[["filename","choice"]].copy()
                df_learn["prob_elegido_modelo_%"] = (p*100).round(1)
                df_learn = df_learn.sort_values("prob_elegido_modelo_%", ascending=False)

                st.dataframe(df_learn, use_container_width=True)
                st.info("Este modelo mejora conforme agregas resultados reales (A/B, encuestas, pilotos).")
            else:
                st.info("Para entrenar un modelo estable, junta ~60+ registros (ej. 10 tests con 6 packs).")
        else:
            st.info("Aún no has guardado resultados reales.")

# ============================================================
# 🧊 Producto Nuevo — Cold Start + Recomendaciones
# ============================================================
with tab_new:
    st.subheader("🧊 Producto Nuevo — Cold Start (sin histórico en dataset)")
    st.caption("Predice éxito y ventas usando atributos + pack (imagen) + claims + comparables (rango p25/p50/p75).")

    c1, c2, c3 = st.columns(3)
    categorias = sorted(df["categoria"].unique().tolist())
    canales = sorted(df["canal"].unique().tolist())

    categoria = c1.selectbox("Categoría comparable", categorias, key="new_cat_v22")
    canal = c2.selectbox("Canal", canales, key="new_canal_v22")
    segmento = c3.selectbox("Segmento objetivo", ["fit", "kids", "premium", "value"], key="new_seg_v22")

    b1, b2, b3, b4, b5 = st.columns(5)
    precio = b1.number_input("Precio", 1.0, 9999.0, float(df["precio"].median()), step=1.0, key="new_precio_v22")
    margen = b2.slider("Margen %", 0, 90, int(np.clip(df["margen_pct"].median(), 0, 90)), key="new_margen_v22")
    competencia = b3.slider("Competencia (1-10)", 1, 10, int(df["competencia"].median()), key="new_comp_v22")
    demanda = b4.slider("Demanda (10-100)", 10, 100, int(df["demanda"].median()), key="new_dem_v22")
    tendencia = b5.slider("Tendencia (20-100)", 20, 100, int(df["tendencia"].median()), key="new_tend_v22")

    # Claims
    st.markdown("### 🏷️ Claims (cold start)")
    recs = recommend_claims(segmento, canal, 10)
    claim_opts = [c for c, _ in recs]
    claims_sel = st.multiselect("Selecciona 2-3 claims", claim_opts, default=claim_opts[:2], key="new_claims_v22")
    cscore = claims_score(claims_sel, canal)
    st.metric("Claims Score", f"{cscore:.1f}/100")

    # Pack
    st.markdown("### 📦 Empaque (opcional, recomendado)")
    img = st.file_uploader("Sube empaque (PNG/JPG)", type=["png","jpg","jpeg"], key="new_pack_v22")

    pack_choice = 60.0
    pack_emotion = 60.0
    pack_label = ""
    pack_quickwins = []

    if img:
        im = Image.open(img)
        st.image(im, caption="Empaque nuevo", use_container_width=True)

        m = image_metrics(im)
        sc = pack_scores_from_metrics(m)
        emo = pack_emotion_from_image(sc, m)
        pred_pack = shelf_3sec_predictor(sc, emo, m)

        pack_choice = float(pred_pack["choice"])
        pack_emotion = float(emo["emotion_vector"][emo["emotion_label"]])
        pack_label = emo["emotion_label"]
        pack_quickwins = pred_pack["quick_wins"]

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Elección (3s)", f"{pack_choice:.1f}/100")
        k2.metric("Emoción dominante", pack_label.upper())
        k3.metric("Legibilidad", f"{sc['pack_legibility_score']}/100")
        k4.metric("Claridad", f"{sc['pack_clarity_score']}/100")

        st.markdown("**Quick wins (pack)**")
        for w in pack_quickwins:
            st.write("•", w)

    # Conexión proxy
    conexion_score = clip(0.45 * demanda + 0.35 * pack_choice + 0.20 * cscore, 0, 100)

    entrada = pd.DataFrame([{
        "precio": float(precio),
        "competencia": float(competencia),
        "demanda": float(demanda),
        "tendencia": float(tendencia),
        "margen_pct": float(margen),
        "conexion_score": float(conexion_score),
        "rating_conexion": float(7),
        "sentiment_score": float(1),
        "marca": "nueva",
        "canal": str(canal).lower(),
    }])

    prob = float(success_model.predict_proba(entrada)[0][1])
    ventas_point = float(sales_model.predict(entrada)[0])

    # Comparables
    comp = df[df["categoria"] == str(categoria).lower()].copy()
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

    launch_score = (
        0.45 * (prob * 100) +
        0.25 * pack_choice +
        0.15 * cscore +
        0.15 * pack_emotion
    )

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
    st.dataframe(
        top[["marca","precio","margen_pct","demanda","tendencia","ventas_unidades","exito"]].copy(),
        use_container_width=True
    )

    # ✅ Guardar para Reporte Ejecutivo
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
        "pack_emotion_label": pack_label,
        "conexion_score_proxy": float(conexion_score),
        "prob_exito": float(prob),
        "ventas_point": float(ventas_point),
        "ventas_p25": float(p25),
        "ventas_p50": float(p50),
        "ventas_p75": float(p75),
        "launch_score": float(launch_score),
    }

    # ============================================================
    # ✅ NUEVO: Recomendaciones para subir probabilidad de éxito
    # ============================================================
    st.divider()
    st.markdown("## 🧠 Recomendaciones para subir probabilidad de éxito (Producto Nuevo)")

    base_row = {
        "precio": float(precio),
        "competencia": float(competencia),
        "demanda": float(demanda),
        "tendencia": float(tendencia),
        "margen_pct": float(margen),
        "conexion_score": float(conexion_score),
        "rating_conexion": float(7),
        "sentiment_score": float(1),
        "marca": "nueva",
        "canal": str(canal).lower(),
    }

    if st.button("🚀 Generar recomendaciones (what-if)", key="new_recos_btn_v22"):
        out, recs_txt, summary = coldstart_recommendations(success_model, sales_model, base_row)

        cA, cB, cC = st.columns(3)
        cA.metric("Prob base", f"{summary['base_prob_%']:.1f}%")
        cB.metric("Mejor prob", f"{summary['best_prob_%']:.1f}%")
        cC.metric("Uplift prob", f"+{summary['uplift_prob_pp']:.1f} pp")

        dA, dB, dC = st.columns(3)
        dA.metric("Ventas base", f"{summary['base_sales']:,.0f} u.")
        dB.metric("Mejor ventas", f"{summary['best_sales']:,.0f} u.")
        dC.metric("Uplift ventas", f"+{summary['uplift_sales']:,.0f} u.")

        st.markdown("### ✅ Acciones recomendadas (Quick Wins)")
        for r in recs_txt:
            st.write("•", r)

        st.markdown("### 🧪 Top escenarios sugeridos (para iterar)")
        show = out.copy()
        show["prob_exito_%"] = (show["prob_exito"]*100).round(1)
        show["precio"] = show["precio"].round(1)
        show["ventas_unidades"] = show["ventas_unidades"].round(0).astype(int)
        st.dataframe(
            show[["prob_exito_%","uplift_prob_pp","precio","margen_pct","delta_claims_proxy","delta_pack_proxy","ventas_unidades","uplift_sales"]],
            use_container_width=True
        )

# ============================================================
# 📄 Reporte Ejecutivo (TXT + CSV inputs)
# ============================================================
with tab_report:
    st.subheader("📄 Reporte Ejecutivo descargable")

    lr = st.session_state.last_run or {}
    lp = st.session_state.last_pack or {}
    ln = st.session_state.last_new or {}

    lines = []
    lines.append("PLATAFORMA IA - REPORTE EJECUTIVO")
    lines.append("--------------------------------")
    lines.append("")

    if lr:
        lines.append("== Simulación (Producto) ==")
        for k in ["marca","canal","segmento","precio","competencia","demanda","tendencia","margen_pct"]:
            if k in lr:
                lines.append(f"{k}: {lr[k]}")
        lines.append(f"claims: {', '.join(lr.get('claims', []))}")
        lines.append(f"claims_score: {lr.get('claims_score')}")
        lines.append(f"pack_emotion_score: {lr.get('pack_emotion_score')}")
        lines.append(f"conexion_score: {lr.get('conexion_score')}")
        lines.append(f"prob_exito: {lr.get('prob_exito')}")
        lines.append(f"pred_exito: {lr.get('pred_exito')}")
        lines.append(f"ventas_pred: {lr.get('ventas_pred')}")
        lines.append(f"ingresos_pred: {lr.get('ingresos_pred')}")
        lines.append(f"utilidad_pred: {lr.get('utilidad_pred')}")
        lines.append("")

    if lp:
        lines.append("== Pack Lab ==")
        if lp.get("pack_scores"):
            lines.append(f"pack_scores: {lp['pack_scores']}")
        if lp.get("emotion"):
            lines.append(f"emotion_dominante: {lp['emotion'].get('emotion_label')}")
            lines.append(f"emotion_confianza: {lp['emotion'].get('confidence')}")
            lines.append(f"emotion_vector: {lp['emotion'].get('emotion_vector')}")
        if lp.get("shelf_pred"):
            lines.append(f"shelf_3sec: {lp['shelf_pred']}")
        lines.append("")

    if ln:
        lines.append("== Producto Nuevo (Cold Start) ==")
        for k, v in ln.items():
            if isinstance(v, list):
                lines.append(f"{k}: {', '.join([str(x) for x in v])}")
            else:
                lines.append(f"{k}: {v}")
        lines.append("")

    report_txt = "\n".join(lines)
    st.text_area("Vista previa", report_txt, height=320, key="rep_preview_v22")

    st.download_button(
        label="📥 Descargar reporte (TXT)",
        data=report_txt.encode("utf-8"),
        file_name="reporte_ejecutivo.txt",
        mime="text/plain",
        key="dl_report_txt_v22",
    )

    inputs = {}
    if lr:
        inputs.update({f"sim_{k}": lr.get(k) for k in lr.keys()})
    if lp.get("pack_scores"):
        for k, v in lp["pack_scores"].items():
            inputs[f"pack_{k}"] = v
    if lp.get("emotion"):
        inputs["pack_emotion_label"] = lp["emotion"].get("emotion_label")
        inputs["pack_emotion_confidence"] = lp["emotion"].get("confidence")
    if ln:
        for k, v in ln.items():
            inputs[f"new_{k}"] = v if not isinstance(v, list) else ", ".join([str(x) for x in v])

    if inputs:
        df_inputs = pd.DataFrame([inputs])
        st.download_button(
            label="📥 Descargar inputs (CSV)",
            data=df_inputs.to_csv(index=False).encode("utf-8"),
            file_name="inputs_reporte.csv",
            mime="text/csv",
            key="dl_inputs_csv_v22",
        )

# ============================================================
# 📂 Datos
# ============================================================
with tab_data:
    st.subheader("📂 Datos + Descarga")
    st.download_button(
        label="📥 Descargar dataset (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="dataset_con_ventas.csv",
        mime="text/csv",
        key="download_csv_v22",
    )
    st.dataframe(df.head(300), use_container_width=True)

# ============================================================
# 🧠 Modelo
# ============================================================
with tab_model:
    st.subheader("🧠 Diagnóstico")
    st.dataframe(pd.DataFrame(cm, index=["Real 0", "Real 1"], columns=["Pred 0", "Pred 1"]), use_container_width=True)
    st.write(f"Error absoluto medio (ventas): **{mae:,.0f}** unidades.")
