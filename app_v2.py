# app.py
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
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from io import BytesIO
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
    page_title="Plataforma IA | Producto + Empaque + Claims + Cold Start",
    layout="wide",
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
# Helpers
# ----------------------------
def _clean_str_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()

def clip(v, a, b):
    return float(np.clip(v, a, b))

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
    st.bar_chart(dfp.set_index("bucket"), use_container_width=True)

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def dict_to_txt(d: dict, title: str = "Reporte Ejecutivo") -> str:
    lines = []
    lines.append(title)
    lines.append("=" * len(title))
    lines.append(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    for k, v in d.items():
        if isinstance(v, (list, tuple)):
            v = ", ".join([str(x) for x in v])
        lines.append(f"- {k}: {v}")
    return "\n".join(lines)

# ----------------------------
# Image metrics (sin OCR pesado)
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
    # 0-100 scores (heurísticos)
    legibility = 70 * m["contrast"] + 30 * (1 - abs(m["edge_density"] - 0.18) / 0.18)
    legibility = clip(legibility, 0, 1) * 100

    target_brightness = 0.55
    brightness_fit = 1 - abs(m["brightness"] - target_brightness) / target_brightness
    shelf_pop = clip(0.75 * m["pop_score"] + 0.25 * clip(brightness_fit, 0, 1), 0, 1) * 100

    clarity = clip(0.6 * m["contrast"] + 0.4 * (1 - clip(m["edge_density"] / 0.35, 0, 1)), 0, 1) * 100

    return {
        "pack_legibility_score": round(float(legibility), 1),
        "pack_shelf_pop_score": round(float(shelf_pop), 1),
        "pack_clarity_score": round(float(clarity), 1),
    }

def pack_emotion_from_image(sc: dict, m: dict) -> dict:
    """
    Emoción proxy (sin modelos pesados):
    - ENERGIA: alto pop + alto colorfulness
    - CONFIANZA: buena legibilidad + claridad
    - CURIOSIDAD: edge_density medio + pop medio
    - CALMA: brillo medio-alto + edge bajo
    """
    leg = sc["pack_legibility_score"] / 100.0
    pop = sc["pack_shelf_pop_score"] / 100.0
    cla = sc["pack_clarity_score"] / 100.0
    bright = m["brightness"]
    edge = m["edge_density"]
    col = m["colorfulness"]

    energia = clip(0.55*pop + 0.45*clip(col/0.35, 0, 1), 0, 1)
    confianza = clip(0.55*leg + 0.45*cla, 0, 1)
    curiosidad = clip(1 - abs(edge - 0.20)/0.20, 0, 1) * clip(0.6*pop + 0.4*cla, 0, 1)
    calma = clip(1 - abs(bright - 0.62)/0.62, 0, 1) * clip(1 - edge/0.35, 0, 1)

    vec = {
        "energia": float(energia),
        "confianza": float(confianza),
        "curiosidad": float(curiosidad),
        "calma": float(calma),
    }
    label = max(vec, key=vec.get)
    return {"emotion_label": label, "emotion_vector": vec}

def shelf_3sec_predictor(sc: dict, emo: dict, m: dict) -> dict:
    """
    Shelf & Emotion Predictor (3-Second Test) — sin DL pesado:
    Output pro:
    - choice (0-100)
    - attention (0-100)
    - recall (0-100)
    - quick_wins (lista)
    """
    leg = sc["pack_legibility_score"]
    pop = sc["pack_shelf_pop_score"]
    cla = sc["pack_clarity_score"]

    # Atención: pop + contraste
    attention = clip(0.60*pop + 0.40*(m["contrast"]*100.0), 0, 100)

    # Recordación: atención + simplicidad (claridad) + emoción dominante
    emo_boost = 8.0 if emo["emotion_label"] in ["energia", "confianza"] else 4.0
    recall = clip(0.50*attention + 0.40*cla + 0.10*leg + emo_boost, 0, 100)

    # Elección: atención + claridad + legibilidad
    choice = clip(0.45*attention + 0.35*cla + 0.20*leg, 0, 100)

    quick_wins = []
    if leg < 60:
        quick_wins.append("Sube legibilidad: mayor contraste texto/fondo y tipografía más gruesa.")
    if cla < 60:
        quick_wins.append("Mejora claridad: reduce ruido visual y limita a 2–3 claims.")
    if pop < 60:
        quick_wins.append("Sube shelf-pop: agrega color acento y mejora jerarquía de marca/beneficio.")
    if m["edge_density"] > 0.28:
        quick_wins.append("Saturación visual alta: simplifica fondos, patrones y microtextos.")
    if not quick_wins:
        quick_wins.append("Buen desempeño: refuerza jerarquía Marca → Beneficio → Variedad → Prueba.")

    return {
        "choice": float(choice),
        "attention": float(attention),
        "recall": float(recall),
        "quick_wins": quick_wins,
        "three_second_verdict": "GANA" if choice >= 70 else ("COMPITE" if choice >= 58 else "PIERDE"),
    }

def pack_emotion_score(pack_legibility, pack_pop, pack_clarity, claims_score_val, copy_tone: int):
    visual = 0.40 * (pack_pop / 100) + 0.30 * (pack_clarity / 100) + 0.15 * (pack_legibility / 100)
    claims = 0.15 * (claims_score_val / 100)
    tone_boost = 0.06 if copy_tone > 0 else (-0.06 if copy_tone < 0 else 0.0)
    score = (visual + claims + tone_boost) * 100
    return float(np.clip(score, 0, 100))


# ----------------------------
# Claims engine (ampliado)
# ----------------------------
CLAIMS_LIBRARY = {
    "fit": [
        ("Alto en proteína", 0.92),
        ("Sin azúcar añadida", 0.90),
        ("Alto en fibra", 0.88),
        ("Integral", 0.84),
        ("Sin colorantes artificiales", 0.80),
        ("Con avena real", 0.78),
        ("Sin conservadores", 0.76),
        ("Bajo en sodio", 0.72),
        ("Ingredientes naturales", 0.70),
        ("Fuente de energía", 0.68),
    ],
    "kids": [
        ("Con vitaminas y minerales", 0.88),
        ("Sabor chocolate", 0.86),
        ("Energía para su día", 0.82),
        ("Con calcio", 0.78),
        ("Hecho con granos", 0.76),
        ("Sin conservadores", 0.74),
        ("Con hierro", 0.72),
        ("Sabor delicioso", 0.70),
    ],
    "premium": [
        ("Ingredientes seleccionados", 0.86),
        ("Sabor intenso", 0.82),
        ("Hecho con avena real", 0.80),
        ("Receta especial", 0.76),
        ("Calidad premium", 0.74),
        ("Textura crujiente", 0.72),
        ("Sin jarabe de maíz", 0.70),
    ],
    "value": [
        ("Rinde más", 0.82),
        ("Ideal para la familia", 0.78),
        ("Gran sabor a mejor precio", 0.76),
        ("Económico y práctico", 0.72),
        ("Buen balance nutricional", 0.70),
    ],
}

CANAL_CLAIM_BOOST = {
    "retail": {
        "Sin azúcar añadida": 1.05, "Alto en fibra": 1.04, "Integral": 1.03,
        "Ideal para la familia": 1.03, "Rinde más": 1.02
    },
    "marketplace": {
        "Alto en proteína": 1.06, "Ingredientes seleccionados": 1.05,
        "Sin colorantes artificiales": 1.05, "Ingredientes naturales": 1.03,
        "Hecho con avena real": 1.03
    },
}

def recommend_claims(segment: str, canal: str, max_claims: int = 8):
    seg = segment.lower().strip()
    canal = canal.lower().strip()
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
    canal = canal.lower().strip()
    boosts = [CANAL_CLAIM_BOOST.get(canal, {}).get(c, 1.0) for c in selected_claims]
    base = float(np.mean(boosts))
    n = len(selected_claims)
    clarity_penalty = 1.0 if n <= 3 else max(0.65, 1.0 - 0.12 * (n - 3))
    score = 78 * base * clarity_penalty
    return float(np.clip(score, 0, 100))


# ----------------------------
# Cold Start Recommendations Engine (robusto)
# ----------------------------
def coldstart_recommendations(success_model, sales_model, base_row: dict, max_rows: int = 18):
    """
    Genera recomendaciones What-If para Producto Nuevo (cold start).
    Devuelve:
      - out: dataframe con escenarios rankeados (con columnas esperadas)
      - recs_txt: lista de quick wins
      - summary: base vs mejor
    """
    def _predict(row_dict):
        X = pd.DataFrame([row_dict])
        prob = float(success_model.predict_proba(X)[0][1])
        sales = float(sales_model.predict(X)[0])
        return prob, max(0.0, sales)

    def _clip(v, lo, hi):
        return float(np.clip(v, lo, hi))

    required = ["precio","competencia","demanda","tendencia","margen_pct","conexion_score",
                "rating_conexion","sentiment_score","marca","canal"]
    missing = [k for k in required if k not in base_row]
    if missing:
        out = pd.DataFrame([{
            "prob_exito": np.nan,
            "uplift_prob_pp": 0.0,
            "precio": base_row.get("precio", np.nan),
            "margen_pct": base_row.get("margen_pct", np.nan),
            "delta_claims_proxy": 0.0,
            "delta_pack_proxy": 0.0,
            "ventas_unidades": np.nan,
            "uplift_sales": 0.0,
            "score_objetivo": 0.0,
        }])
        recs_txt = [f"Faltan llaves en base_row para recomendar: {missing}"]
        summary = {"base_prob_%": np.nan, "best_prob_%": np.nan, "uplift_prob_pp": 0.0,
                   "base_sales": np.nan, "best_sales": np.nan, "uplift_sales": 0.0}
        return out, recs_txt, summary

    base = base_row.copy()
    base_prob, base_sales = _predict(base)

    p0 = float(base["precio"])
    m0 = float(base["margen_pct"])
    d0 = float(base["demanda"])
    t0 = float(base["tendencia"])
    cx0 = float(base["conexion_score"])

    precio_vals = np.linspace(max(1.0, p0*0.85), p0*1.15, 9)
    margen_vals = np.linspace(_clip(m0-10, 0, 90), _clip(m0+10, 0, 90), 7)

    delta_claims_vals = [-10, -5, 0, +5, +10]  # proxy en conexión
    delta_pack_vals   = [-12, -6, 0, +6, +12]  # proxy en conexión

    scenarios = []
    for p in precio_vals:
        for m in margen_vals:
            for dc in delta_claims_vals:
                for dp in delta_pack_vals:
                    s = base.copy()
                    s["precio"] = float(p)
                    s["margen_pct"] = float(m)

                    # elasticidad simple
                    price_delta = (float(p) - p0) / max(p0, 1e-6)
                    s["demanda"] = _clip(d0 * (1 - 0.30*price_delta), 10, 100)
                    s["tendencia"] = _clip(t0 * (1 - 0.10*price_delta), 20, 100)

                    # conexión
                    s["conexion_score"] = _clip(cx0 + dc + dp, 0, 100)

                    prob, sales = _predict(s)
                    scenarios.append({
                        "prob_exito": prob,
                        "uplift_prob_pp": (prob - base_prob)*100.0,
                        "precio": float(p),
                        "margen_pct": float(m),
                        "delta_claims_proxy": float(dc),
                        "delta_pack_proxy": float(dp),
                        "ventas_unidades": float(sales),
                        "uplift_sales": float(sales - base_sales),
                    })

    out = pd.DataFrame(scenarios)
    if out.empty:
        recs_txt = ["No se pudieron generar escenarios (grid vacío). Revisa rangos."]
        summary = {
            "base_prob_%": base_prob*100.0,
            "best_prob_%": base_prob*100.0,
            "uplift_prob_pp": 0.0,
            "base_sales": base_sales,
            "best_sales": base_sales,
            "uplift_sales": 0.0,
        }
        return out, recs_txt, summary

    sales_norm = out["ventas_unidades"] / max(out["ventas_unidades"].max(), 1.0)
    out["score_objetivo"] = (out["prob_exito"]*100.0)*0.65 + (sales_norm*100.0)*0.35

    good = out[(out["uplift_prob_pp"] >= 1.0) | (out["uplift_sales"] >= 250)].copy()
    if good.empty:
        good = out.copy()

    good = good.sort_values(["score_objetivo","uplift_prob_pp","uplift_sales"], ascending=False).head(max_rows).reset_index(drop=True)
    best = good.iloc[0].to_dict()

    recs_txt = []
    if best["precio"] < p0:
        recs_txt.append(f"Baja precio hacia ~{best['precio']:.0f} para subir demanda (y prob. de éxito).")
    elif best["precio"] > p0:
        recs_txt.append(f"Subir precio a ~{best['precio']:.0f} sólo conviene si sostienes conexión (pack/claims).")

    if best["margen_pct"] > m0:
        recs_txt.append(f"Sube margen a ~{best['margen_pct']:.1f}% (mejor unit economics) sin perder conversión.")
    else:
        recs_txt.append("Si bajas margen, compénsalo con claims/pack para sostener la conexión.")

    if best["delta_pack_proxy"] > 0:
        recs_txt.append("Pack: sube claridad/legibilidad y reduce ruido (mejora elección 3s y recordación).")
    else:
        recs_txt.append("Sube imagen del empaque: así el sistema recomienda con el 3-second real (no proxy).")

    if best["delta_claims_proxy"] > 0:
        recs_txt.append("Claims: usa 2–3 claims (beneficio principal + prueba). Evita sobrecargar.")

    summary = {
        "base_prob_%": base_prob*100.0,
        "best_prob_%": float(best["prob_exito"]*100.0),
        "uplift_prob_pp": float(best["uplift_prob_pp"]),
        "base_sales": float(base_sales),
        "best_sales": float(best["ventas_unidades"]),
        "uplift_sales": float(best["uplift_sales"]),
    }

    return good, recs_txt, summary


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
            "marca", "canal", "precio", "competencia", "demanda", "tendencia",
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
                n_estimators=350,
                random_state=42,
                class_weight="balanced_subsample"
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
        raise ValueError(f"Este CSV no trae ventas. Faltan: {missing_sales}.")

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


# ----------------------------
# Sidebar: load CSV robust
# ----------------------------
st.sidebar.title("⚙️ Datos")
uploaded = st.sidebar.file_uploader("Sube tu CSV (con ventas)", type=["csv"], key="uploader_csv_v20")

if uploaded is not None:
    df = load_data(uploaded)
else:
    if Path(DATA_PATH_DEFAULT).exists():
        df = load_data(DATA_PATH_DEFAULT)
    else:
        st.sidebar.warning(f"❗No encontré '{DATA_PATH_DEFAULT}' en el repo. Sube el CSV para arrancar.")
        st.stop()

# Train models
try:
    success_model, acc, auc, cm = train_success_model(df)
    sales_model, mae = train_sales_model(df)
except Exception as e:
    st.error(f"Error entrenando modelos: {e}")
    st.stop()


# ----------------------------
# Header
# ----------------------------
st.title("🧠 Plataforma IA: Producto + Empaque + Claims + Producto Nuevo")
st.caption("Éxito + Ventas estimadas + Insights + Pack Vision+ + Claims Lab + Cold Start + Reporte Ejecutivo")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Registros", f"{len(df):,}")
k2.metric("Precisión (test)", f"{acc * 100:.2f}%")
k3.metric("AUC (test)", f"{auc:.3f}")
k4.metric("Éxito base", f"{df['exito'].mean() * 100:.1f}%")
k5.metric("MAE ventas", f"{mae:,.0f} u.")

st.divider()

tab_sim, tab_ins, tab_pack, tab_claims, tab_exp, tab_new, tab_report, tab_data, tab_model = st.tabs(
    ["🧪 Simulador", "📊 Insights", "📦 Pack Vision+", "🏷️ Claims Lab", "🧪 Experimentos", "🧊 Producto Nuevo", "📄 Reporte", "📂 Datos", "🧠 Modelo"]
)


# ============================================================
# 🧪 Simulador
# ============================================================
with tab_sim:
    st.subheader("🧪 Simulador What-If (producto con variables + pack/claims proxy)")

    marcas = sorted(df["marca"].unique().tolist())
    canales = sorted(df["canal"].unique().tolist())

    c1, c2, c3 = st.columns(3)
    marca = c1.selectbox("Marca", marcas, 0, key="sim_marca_v20")
    canal = c2.selectbox("Canal", canales, 0, key="sim_canal_v20")
    segmento = c3.selectbox("Segmento objetivo", ["fit", "kids", "premium", "value"], 0, key="sim_segmento_v20")

    st.markdown("### Variables de negocio")
    b1, b2, b3, b4, b5 = st.columns(5)
    precio = b1.number_input("Precio", value=float(df["precio"].median()), step=1.0, key="sim_precio_v20")
    competencia = b2.slider("Competencia (1-10)", 1, 10, int(df["competencia"].median()), key="sim_comp_v20")
    demanda = b3.slider("Demanda (10-100)", 10, 100, int(df["demanda"].median()), key="sim_dem_v20")
    tendencia = b4.slider("Tendencia (20-100)", 20, 100, int(df["tendencia"].median()), key="sim_tend_v20")
    margen_pct = b5.slider("Margen %", 0, 90, int(np.clip(df["margen_pct"].median(), 0, 90)), key="sim_margen_v20")

    st.markdown("### Empaque + Claims (proxy)")
    p1, p2, p3 = st.columns(3)
    pack_legibility_score = p1.slider("Pack legibilidad (0-100)", 0, 100, 65, key="sim_pack_leg_v20")
    pack_shelf_pop_score = p2.slider("Pack shelf pop (0-100)", 0, 100, 70, key="sim_pack_pop_v20")
    pack_clarity_score = p3.slider("Pack claridad (0-100)", 0, 100, 65, key="sim_pack_cla_v20")

    recs = recommend_claims(segmento, canal, max_claims=8)
    claim_options = [c for c, _ in recs]
    selected_claims = st.multiselect(
        "Selecciona claims (ideal 2-3)",
        claim_options,
        default=claim_options[:2],
        key="sim_claims_v20",
    )

    copy = st.text_input("Copy corto (opcional)", value="Energía y nutrición para tu día", key="sim_copy_v20")
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
    pack_emotion = pack_emotion_score(pack_legibility_score, pack_shelf_pop_score, pack_clarity_score, cscore, copy_tone)

    rating_conexion = st.slider("Rating conexión producto (1-10)", 1, 10, 7, key="sim_rating_v20")
    sentiment_score = st.select_slider("Sentimiento del producto (-1/0/1)", options=[-1, 0, 1], value=1, key="sim_sent_v20")

    base_conexion = (rating_conexion / 10) * 70 + sentiment_score * 15 + 5
    # conexión final pondera pack_emotion + claims
    conexion_score = clip(0.70*base_conexion + 0.20*pack_emotion + 0.10*cscore, 0, 100)

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
    s2.metric("Emotion Pack Score", f"{pack_emotion:.1f}/100")
    s3.metric("Conexión final", f"{conexion_score:.1f}/100")

    if st.button("🚀 Simular", key="sim_btn_v20"):
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

        st.dataframe(entrada, use_container_width=True)

        st.session_state.last_sim = {
            "marca": marca,
            "canal": canal,
            "segmento": segmento,
            "precio": float(precio),
            "margen_pct": float(margen_pct),
            "competencia": float(competencia),
            "demanda": float(demanda),
            "tendencia": float(tendencia),
            "claims": selected_claims,
            "claims_score": float(cscore),
            "pack_emotion_score": float(pack_emotion),
            "conexion_score": float(conexion_score),
            "prob_exito": float(p),
            "ventas": float(ventas),
            "ingresos": float(ingresos),
            "utilidad": float(utilidad),
        }


# ============================================================
# 📊 Insights
# ============================================================
with tab_ins:
    st.subheader("📊 Insights")

    left, right = st.columns(2)

    with left:
        st.markdown("**Ranking por marca (Conexión promedio)**")
        ins_marca = (df.groupby("marca")[["conexion_score"]].mean().sort_values("conexion_score", ascending=False).round(2))
        st.dataframe(ins_marca, use_container_width=True)

        st.markdown("**Ranking por marca (Éxito %)**")
        ex_marca = (df.groupby("marca")[["exito"]].mean().sort_values("exito", ascending=False).round(3))
        ex_marca["exito_%"] = (ex_marca["exito"] * 100).round(1)
        st.dataframe(ex_marca[["exito_%"]], use_container_width=True)

        st.markdown("**Ranking por marca (Ventas promedio)**")
        v_marca = (df.groupby("marca")[["ventas_unidades"]].mean().sort_values("ventas_unidades", ascending=False).round(0))
        st.dataframe(v_marca, use_container_width=True)

    with right:
        st.markdown("**Marca + Canal (Conexión promedio)**")
        ins_mc = (df.groupby(["marca", "canal"])[["conexion_score"]].mean().sort_values("conexion_score", ascending=False).round(2))
        st.dataframe(ins_mc.head(25), use_container_width=True)

        st.markdown("**Marca + Canal (Éxito %)**")
        ex_mc = (df.groupby(["marca", "canal"])[["exito"]].mean().sort_values("exito", ascending=False).round(3))
        ex_mc["exito_%"] = (ex_mc["exito"] * 100).round(1)
        st.dataframe(ex_mc.head(25)[["exito_%"]], use_container_width=True)

        st.markdown("**Marca + Canal (Ventas promedio)**")
        v_mc = (df.groupby(["marca", "canal"])[["ventas_unidades"]].mean().sort_values("ventas_unidades", ascending=False).round(0))
        st.dataframe(v_mc.head(25), use_container_width=True)

    st.divider()
    d1, d2 = st.columns(2)
    with d1:
        bins = pd.cut(df["conexion_score"], bins=[0,20,40,60,80,100], include_lowest=True)
        dist = bins.value_counts().sort_index()
        bar_from_value_counts(dist, title="Distribución: Conexión emocional (bucket)")
    with d2:
        bins2 = pd.cut(df["ventas_unidades"].clip(0, 40000), bins=[0,2000,5000,10000,20000,40000], include_lowest=True)
        dist2 = bins2.value_counts().sort_index()
        bar_from_value_counts(dist2, title="Distribución: Ventas unidades (bucket)")


# ============================================================
# 📦 Pack Vision+
# ============================================================
with tab_pack:
    st.subheader("📦 Pack Vision+ (sube tu empaque y te doy métricas + quick wins + 3-second test)")
    img_file = st.file_uploader("Sube imagen del empaque (PNG/JPG)", type=["png","jpg","jpeg"], key="pack_uploader_v20")

    if img_file is None:
        st.info("Sube una imagen para generar análisis del empaque.")
    else:
        img = Image.open(img_file)
        st.image(img, caption="Empaque cargado", use_container_width=True)

        m = image_metrics(img)
        sc = pack_scores_from_metrics(m)
        emo = pack_emotion_from_image(sc, m)
        pred_pack = shelf_3sec_predictor(sc, emo, m)

        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Brillo", f"{m['brightness']:.2f}")
        a2.metric("Contraste", f"{m['contrast']:.2f}")
        a3.metric("Colorfulness", f"{m['colorfulness']:.2f}")
        a4.metric("Edge density", f"{m['edge_density']:.3f}")

        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Legibilidad", f"{sc['pack_legibility_score']}/100")
        b2.metric("Shelf Pop", f"{sc['pack_shelf_pop_score']}/100")
        b3.metric("Claridad", f"{sc['pack_clarity_score']}/100")
        b4.metric("3-Second Verdict", pred_pack["three_second_verdict"])

        c1, c2, c3 = st.columns(3)
        c1.metric("Atención", f"{pred_pack['attention']:.1f}/100")
        c2.metric("Recordación", f"{pred_pack['recall']:.1f}/100")
        c3.metric("Elección", f"{pred_pack['choice']:.1f}/100")

        st.markdown(f"**Emoción dominante:** `{emo['emotion_label']}` (proxy)")
        st.markdown("### Quick wins (pack)")
        for w in pred_pack["quick_wins"]:
            st.write("•", w)

        st.session_state.last_pack = {
            "pack_legibility_score": sc["pack_legibility_score"],
            "pack_shelf_pop_score": sc["pack_shelf_pop_score"],
            "pack_clarity_score": sc["pack_clarity_score"],
            "emotion_label": emo["emotion_label"],
            "choice_3s": pred_pack["choice"],
            "attention": pred_pack["attention"],
            "recall": pred_pack["recall"],
            "verdict": pred_pack["three_second_verdict"],
            "quick_wins": pred_pack["quick_wins"],
        }


# ============================================================
# 🏷️ Claims Lab
# ============================================================
with tab_claims:
    st.subheader("🏷️ Claims Lab (recomendaciones + score)")

    c1, c2 = st.columns(2)
    segmento = c1.selectbox("Segmento", ["fit", "kids", "premium", "value"], 0, key="claims_segmento_v20")
    canal_c = c2.selectbox("Canal", ["retail", "marketplace"], 0, key="claims_canal_v20")

    recs = recommend_claims(segmento, canal_c, max_claims=10)
    rec_df = pd.DataFrame(recs, columns=["claim", "score_base"])
    rec_df["score_base"] = (rec_df["score_base"] * 100).round(1)

    st.dataframe(rec_df, use_container_width=True)

    selected = st.multiselect(
        "Selecciona 2-3 claims",
        rec_df["claim"].tolist(),
        default=rec_df["claim"].tolist()[:2],
        key="claims_selected_v20",
    )
    cscore = claims_score(selected, canal_c)
    st.metric("Claims Score", f"{cscore:.1f}/100")
    st.warning("Nota: recomendación comercial (no legal/regulatoria).")


# ============================================================
# 🧪 Experimentos (A/B)
# ============================================================
with tab_exp:
    st.subheader("🧪 Experimentos (A/B)")

    if "experiments" not in st.session_state:
        st.session_state.experiments = []

    c1, c2, c3 = st.columns(3)
    exp_name = c1.text_input("Nombre experimento", value="Test Pack v1 vs v2", key="exp_name_v20")
    variant = c2.selectbox("Variante", ["A", "B"], 0, key="exp_variant_v20")
    metric = c3.selectbox("Métrica", ["intencion_compra", "conexion_pack", "ventas_piloto"], 0, key="exp_metric_v20")

    v1, v2, v3 = st.columns(3)
    marca_e = v1.selectbox("Marca", sorted(df["marca"].unique().tolist()), 0, key="exp_marca_v20")
    canal_e = v2.selectbox("Canal", sorted(df["canal"].unique().tolist()), 0, key="exp_canal_v20")
    value = v3.number_input("Valor observado", value=7.0, step=0.1, key="exp_value_v20")

    if st.button("➕ Guardar medición", key="exp_save_v20"):
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
# 🧊 Producto Nuevo — Cold Start + Recomendaciones (INTEGRADO)
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

    has_pack = img is not None
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
        pack_emotion = float(emo["emotion_vector"][emo["emotion_label"]]) * 100.0  # ✅ importante: 0–100
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

    # Conexión proxy (si no hay imagen, usa defaults; si hay, se vuelve más real)
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
    ventas_point = float(max(0.0, sales_model.predict(entrada)[0]))

    # Comparables
    comp = df[df["categoria"] == str(categoria).lower()].copy()
    if comp.empty:
        comp = df.copy()

    # (opcional) evita comparables con marca literal "nueva"
    if "marca" in comp.columns:
        comp = comp[comp["marca"] != "nueva"]

    comp["dist"] = (
        (comp["precio"] - float(precio)).abs() / max(float(precio), 1e-6) +
        (comp["margen_pct"] - float(margen)).abs() / 100.0 +
        (comp["demanda"] - float(demanda)).abs() / 100.0 +
        (comp["tendencia"] - float(tendencia)).abs() / 100.0
    )
    top = comp.sort_values("dist").head(20)

    # si por algo queda vacío, fallback a todo df
    if top.empty:
        top = df.sort_values("precio").head(20)

    p25 = float(np.percentile(top["ventas_unidades"], 25))
    p50 = float(np.percentile(top["ventas_unidades"], 50))
    p75 = float(np.percentile(top["ventas_unidades"], 75))

    # Launch score (si no hay pack, score parcial para no castigar)
    launch_score_full = (
        0.45 * (prob * 100) +
        0.25 * pack_choice +
        0.15 * cscore +
        0.15 * pack_emotion
    )
    launch_score_partial = (
        0.70 * (prob * 100) +
        0.30 * cscore
    )
    launch_score = launch_score_full if has_pack else launch_score_partial

    st.markdown("## 🎯 Resultado (Producto Nuevo)")
    o1, o2, o3, o4 = st.columns(4)
    o1.metric("Prob. éxito", f"{prob*100:.1f}%")
    o2.metric("Ventas (punto)", f"{ventas_point:,.0f} u.")
    o3.metric("Rango comparables (p25–p75)", f"{p25:,.0f} — {p75:,.0f} u.")
    o4.metric("Launch Score", f"{launch_score:.1f}/100" + ("" if has_pack else " (sin pack)"))

    # Semáforo coherente:
    if has_pack:
        if launch_score >= 75:
            st.success("✅ GO — Alto potencial (con empaque evaluado)")
        elif launch_score >= 60:
            st.warning("🟡 AJUSTAR — Optimiza pack/claims/precio para subir score")
        else:
            st.error("🔴 NO-GO — Riesgo alto (necesita rediseño o cambiar estrategia)")
    else:
        if prob >= 0.70:
            st.success("✅ GO preliminar — Buen potencial (sube empaque para validar 3-second)")
        elif prob >= 0.55:
            st.warning("🟡 AJUSTAR preliminar — Potencial medio (sube empaque + optimiza claims/precio)")
        else:
            st.error("🔴 NO-GO preliminar — Riesgo alto (necesita rediseño/estrategia)")

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
        "has_pack": bool(has_pack),
    }

    # ============================================================
    # ✅ Recomendaciones para subir probabilidad de éxito (INTEGRADO)
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
        show["uplift_sales"] = show["uplift_sales"].round(0).astype(int)
        show["uplift_prob_pp"] = show["uplift_prob_pp"].round(1)

        st.dataframe(
            show[[
                "prob_exito_%","uplift_prob_pp","precio","margen_pct",
                "delta_claims_proxy","delta_pack_proxy",
                "ventas_unidades","uplift_sales"
            ]],
            use_container_width=True
        )

        st.session_state.last_new_recs = {
            "summary": summary,
            "top_scenarios": show.head(15).to_dict(orient="records"),
            "quick_wins": recs_txt,
        }


# ============================================================
# 📄 Reporte Ejecutivo (TXT + CSV)
# ============================================================
with tab_report:
    st.subheader("📄 Reporte Ejecutivo descargable (TXT + CSV inputs)")
    st.caption("Genera un reporte estable sin librerías pesadas. Incluye último Simulador / Pack / Producto Nuevo.")

    report_dict = {"nota": "Corre simulaciones o Producto Nuevo para llenar el reporte."}

    if "last_new" in st.session_state or "last_sim" in st.session_state or "last_pack" in st.session_state:
        report_dict = {}
        if "last_sim" in st.session_state:
            report_dict["simulador"] = st.session_state.last_sim
        if "last_pack" in st.session_state:
            report_dict["pack_vision"] = st.session_state.last_pack
        if "last_new" in st.session_state:
            report_dict["producto_nuevo"] = st.session_state.last_new
        if "last_new_recs" in st.session_state:
            report_dict["recomendaciones_producto_nuevo"] = st.session_state.last_new_recs

    # Vista humana
    st.json(report_dict)

    # TXT (simple)
    txt = dict_to_txt({"contenido": report_dict}, title="Reporte Ejecutivo — Plataforma IA")
    st.download_button(
        "📥 Descargar Reporte (TXT)",
        data=txt.encode("utf-8"),
        file_name="reporte_ejecutivo_plataforma_ia.txt",
        mime="text/plain",
        key="download_report_txt_v20"
    )

    # CSV inputs (último producto nuevo)
    if "last_new" in st.session_state:
        last_new = st.session_state.last_new.copy()
        # aplanar claims
        last_new["claims"] = ", ".join(last_new.get("claims", []))
        df_inputs = pd.DataFrame([last_new])
        st.download_button(
            "📥 Descargar inputs Producto Nuevo (CSV)",
            data=df_to_csv_bytes(df_inputs),
            file_name="inputs_producto_nuevo.csv",
            mime="text/csv",
            key="download_inputs_new_csv_v20"
        )


# ============================================================
# 📂 Datos + Descarga
# ============================================================
with tab_data:
    st.subheader("📂 Datos + Descarga")
    st.download_button(
        label="📥 Descargar dataset cargado (CSV)",
        data=df_to_csv_bytes(df),
        file_name="dataset_con_ventas.csv",
        mime="text/csv",
        key="download_csv_v20",
    )
    st.dataframe(df.head(300), use_container_width=True)


# ============================================================
# 🧠 Modelo
# ============================================================
with tab_model:
    st.subheader("🧠 Diagnóstico")
    st.dataframe(pd.DataFrame(cm, index=["Real 0", "Real 1"], columns=["Pred 0", "Pred 1"]), use_container_width=True)
    st.write(f"Error absoluto medio (ventas): **{mae:,.0f}** unidades.")
