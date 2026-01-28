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
# Config
# ----------------------------
st.set_page_config(page_title="Plataforma IA | Producto + Empaque + Claims", layout="wide")
DATA_PATH_DEFAULT = "mercado_cereales_5000_con_ventas.csv"

REQUIRED_BASE = {
    "marca","categoria","canal","precio","costo","margen","margen_pct",
    "competencia","demanda","tendencia","estacionalidad",
    "rating_conexion","comentario","sentiment_score",
    "conexion_score","conexion_alta","score_latente","exito"
}
REQUIRED_SALES = {"ventas_unidades","ventas_ingresos","utilidad"}

# ----------------------------
# Helpers
# ----------------------------
def _clean_str_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()

def clip(v, a, b):
    return float(max(a, min(b, v)))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def safe_percent(x):
    return f"{x*100:.2f}%"

# ----------------------------
# Image metrics (sin OCR)
# ----------------------------
def image_metrics(img: Image.Image) -> dict:
    """
    Métricas simples (rápidas) para aproximar:
    - brillo, contraste, colorfulness, edge_density (pop/claridad)
    """
    im = img.convert("RGB")
    arr = np.asarray(im).astype(np.float32)

    # brightness
    gray = (0.299*arr[...,0] + 0.587*arr[...,1] + 0.114*arr[...,2])
    brightness = float(np.mean(gray) / 255.0)

    # contrast (std)
    contrast = float(np.std(gray) / 255.0)

    # colorfulness (Hasler & Süsstrunk approximation)
    rg = arr[...,0] - arr[...,1]
    yb = 0.5*(arr[...,0] + arr[...,1]) - arr[...,2]
    colorfulness = float((np.std(rg) + 0.3*np.std(yb)) / 255.0)

    # edge density: sobel magnitude threshold
    # (simple approximation)
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:,1:-1] = gray[:,2:] - gray[:,:-2]
    gy[1:-1,:] = gray[2:,:] - gray[:-2,:]
    mag = np.sqrt(gx**2 + gy**2)
    thresh = np.percentile(mag, 85)
    edges = (mag > thresh).astype(np.float32)
    edge_density = float(np.mean(edges))

    # "pop" proxy: balance contrast + colorfulness
    pop_score = clip(0.55*contrast + 0.45*colorfulness, 0, 1)

    return {
        "brightness": brightness,
        "contrast": contrast,
        "colorfulness": colorfulness,
        "edge_density": edge_density,
        "pop_score": pop_score
    }

def pack_scores_from_metrics(m: dict) -> dict:
    """
    Convierte métricas en scores 0-100 (interpretables).
    """
    # Legibilidad proxy: contraste alto y edge_density moderada (demasiados bordes = ruido)
    legibility = 70*m["contrast"] + 30*(1 - abs(m["edge_density"] - 0.18)/0.18)
    legibility = clip(legibility, 0, 1) * 100

    # Shelf pop proxy: pop_score + brillo medio (ni muy oscuro ni quemado)
    target_brightness = 0.55
    brightness_fit = 1 - abs(m["brightness"] - target_brightness)/target_brightness
    shelf_pop = clip(0.75*m["pop_score"] + 0.25*clip(brightness_fit, 0, 1), 0, 1) * 100

    # Clarity proxy: edge_density baja/moderada + contraste medio/alto
    clarity = clip(0.6*m["contrast"] + 0.4*(1 - clip(m["edge_density"]/0.35, 0, 1)), 0, 1) * 100

    return {
        "pack_legibility_score": round(legibility, 1),
        "pack_shelf_pop_score": round(shelf_pop, 1),
        "pack_clarity_score": round(clarity, 1),
    }

# ----------------------------
# Claims engine (reglas + scoring)
# ----------------------------
CLAIMS_LIBRARY = {
    "fit": [
        ("Alto en proteína", 0.90),
        ("Sin azúcar añadida", 0.88),
        ("Alto en fibra", 0.86),
        ("Integral", 0.80),
        ("Bajo en calorías", 0.78),
        ("Sin colorantes artificiales", 0.72)
    ],
    "kids": [
        ("Con vitaminas y minerales", 0.86),
        ("Sabor chocolate", 0.82),
        ("Energía para su día", 0.78),
        ("Hecho con granos", 0.74),
        ("Sin conservadores", 0.70)
    ],
    "premium": [
        ("Ingredientes seleccionados", 0.82),
        ("Sabor intenso", 0.78),
        ("Hecho con avena real", 0.76),
        ("Calidad premium", 0.70),
        ("Receta artesanal", 0.64)
    ],
    "value": [
        ("Rinde más", 0.78),
        ("Gran sabor a mejor precio", 0.74),
        ("Ideal para la familia", 0.72),
        ("Económico y práctico", 0.66)
    ]
}

CANAL_CLAIM_BOOST = {
    "retail": {"Sin azúcar añadida": 1.04, "Alto en fibra": 1.03, "Integral": 1.02},
    "marketplace": {"Alto en proteína": 1.05, "Sin colorantes artificiales": 1.04, "Ingredientes seleccionados": 1.03}
}

def recommend_claims(segment: str, canal: str, max_claims: int = 5):
    seg = segment.lower().strip()
    canal = canal.lower().strip()
    items = CLAIMS_LIBRARY.get(seg, [])[:]
    scored = []
    for claim, base in items:
        boost = CANAL_CLAIM_BOOST.get(canal, {}).get(claim, 1.0)
        score = base * boost
        scored.append((claim, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:max_claims]

def claims_score(selected_claims, canal: str) -> float:
    """
    Score 0-100 basado en:
    - base scores del segmento (aprox)
    - boosts por canal
    - penalización por saturación (más de 3 claims baja claridad)
    """
    if not selected_claims:
        return 0.0
    canal = canal.lower().strip()

    # base = promedio de boosts (si no existe claim en boost, =1)
    boosts = []
    for c in selected_claims:
        boosts.append(CANAL_CLAIM_BOOST.get(canal, {}).get(c, 1.0))
    base = np.mean(boosts)

    # claridad: penaliza demasiados claims
    n = len(selected_claims)
    clarity_penalty = 1.0 if n <= 3 else max(0.65, 1.0 - 0.12*(n-3))

    score = 75 * base * clarity_penalty
    return float(np.clip(score, 0, 100))

# ----------------------------
# Emoción del empaque
# ----------------------------
def pack_emotion_score(pack_legibility, pack_pop, pack_clarity, claims_score_val, copy_tone: int):
    """
    Score 0-100:
    - Pop + claridad + claims (impactan intención)
    - tono del copy: -1,0,1 (negativo, neutro, positivo)
    """
    # peso visual
    visual = 0.40*(pack_pop/100) + 0.30*(pack_clarity/100) + 0.15*(pack_legibility/100)
    # peso claims
    claims = 0.15*(claims_score_val/100)

    # tono
    tone_boost = 0.06 if copy_tone > 0 else (-0.06 if copy_tone < 0 else 0.0)

    score = (visual + claims + tone_boost) * 100
    return float(np.clip(score, 0, 100))

# ----------------------------
# Data loading
# ----------------------------
@st.cache_data
def load_data(path_or_file) -> pd.DataFrame:
    df = pd.read_csv(path_or_file).copy()

    for c in ["marca","categoria","canal","estacionalidad","comentario"]:
        if c in df.columns:
            df[c] = _clean_str_series(df[c])

    missing = sorted(list(REQUIRED_BASE - set(df.columns)))
    if missing:
        raise ValueError(f"Faltan columnas base en el CSV: {missing}")

    # numeric
    num_cols = [
        "precio","costo","margen","margen_pct","competencia","demanda","tendencia",
        "rating_conexion","sentiment_score","conexion_score","conexion_alta",
        "score_latente","exito"
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ["ventas_unidades","ventas_ingresos","utilidad"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["marca","canal","precio","competencia","demanda","tendencia","margen_pct","conexion_score","rating_conexion","sentiment_score","exito"])
    df["exito"] = df["exito"].astype(int)
    return df

# ----------------------------
# Models
# ----------------------------
@st.cache_resource
def train_success_model(df: pd.DataFrame):
    features = [
        "precio","competencia","demanda","tendencia","margen_pct",
        "conexion_score","rating_conexion","sentiment_score",
        "marca","canal"
    ]
    X = df[features]
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
        ("model", RandomForestClassifier(n_estimators=350, random_state=42, class_weight="balanced_subsample"))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
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
        "precio","competencia","demanda","tendencia","margen_pct",
        "conexion_score","rating_conexion","sentiment_score",
        "marca","canal"
    ]
    X = df[features]
    y = df["ventas_unidades"].astype(float)

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

    reg = Pipeline(steps=[
        ("preprocessor", pre),
        ("model", RandomForestRegressor(n_estimators=350, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    reg.fit(X_train, y_train)

    pred = reg.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    return reg, mae

# ----------------------------
# Sidebar: load CSV robust
# ----------------------------
st.sidebar.title("⚙️ Datos")
uploaded = st.sidebar.file_uploader("Sube tu CSV (con ventas)", type=["csv"])

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
# Header
# ----------------------------
st.title("🧠 Plataforma IA: Producto + Empaque + Claims")
st.caption("Éxito + Ventas estimadas + Pack Lab + Claims Lab + Experimentos (A/B)")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Registros", f"{len(df):,}")
k2.metric("Precisión", f"{acc*100:.2f}%")
k3.metric("AUC", f"{auc:.3f}")
k4.metric("Éxito base", f"{df['exito'].mean()*100:.1f}%")
k5.metric("MAE ventas", f"{mae:,.0f} u.")

st.divider()

# ----------------------------
# Tabs
# ----------------------------
tab_sim, tab_pack, tab_claims, tab_exp, tab_data, tab_model = st.tabs(
    ["🧪 Simulador", "📦 Pack Lab", "🏷️ Claims Lab", "🧪 Experimentos", "📂 Datos", "🧠 Modelo"]
)

# ============================================================
# 🧪 Simulador (usa outputs de Pack+Claims como uplift)
# ============================================================
with tab_sim:
    st.subheader("🧪 Simulador What-If (incluye empaque + claims)")
    st.write("La lógica: **Pack Score + Claims Score** ajustan tu **conexión emocional** y por eso impactan éxito y ventas.")

    marcas = sorted(df["marca"].unique().tolist())
    canales = sorted(df["canal"].unique().tolist())

    c1, c2, c3 = st.columns(3)
    marca = c1.selectbox("Marca", marcas, 0)
    canal = c2.selectbox("Canal", canales, 0)
    segmento = c3.selectbox("Segmento objetivo", ["fit", "kids", "premium", "value"], 0)

    st.markdown("### Variables de negocio")
    b1, b2, b3, b4, b5 = st.columns(5)
    precio = b1.number_input("Precio", value=float(df["precio"].median()), step=1.0)
    competencia = b2.slider("Competencia (1-10)", 1, 10, int(df["competencia"].median()))
    demanda = b3.slider("Demanda (10-100)", 10, 100, int(df["demanda"].median()))
    tendencia = b4.slider("Tendencia (20-100)", 20, 100, int(df["tendencia"].median()))
    margen_pct = b5.slider("Margen %", 0, 90, int(np.clip(df["margen_pct"].median(), 0, 90)))

    st.markdown("### Empaque + Claims (inputs)")
    p1, p2, p3 = st.columns(3)
    pack_legibility_score = p1.slider("Pack legibilidad (0-100)", 0, 100, 65)
    pack_shelf_pop_score = p2.slider("Pack shelf pop (0-100)", 0, 100, 70)
    pack_clarity_score = p3.slider("Pack claridad (0-100)", 0, 100, 65)

    # claims (selección rápida)
    recs = recommend_claims(segmento, canal, max_claims=6)
    claim_options = [c for c,_ in recs]
    selected_claims = st.multiselect("Selecciona claims (ideal 2-3)", claim_options, default=claim_options[:2])

    # tono del copy (manual / rápido)
    copy = st.text_input("Copy corto (opcional)", value="Energía y nutrición para tu día")
    # tono simple (no NLP pesado)
    pos_kw = ["energía","nutrición","saludable","delicioso","me encanta","premium","calidad","proteína","fibra"]
    neg_kw = ["caro","no","malo","rechazo","no me gusta","pésimo","horrible"]
    t = copy.lower()
    tone = 0
    if any(k in t for k in pos_kw): tone += 1
    if any(k in t for k in neg_kw): tone -= 1
    copy_tone = 1 if tone>0 else (-1 if tone<0 else 0)

    # Scores
    cscore = claims_score(selected_claims, canal)
    pack_emotion = pack_emotion_score(pack_legibility_score, pack_shelf_pop_score, pack_clarity_score, cscore, copy_tone)

    # Ajuste de conexión: base (rating+sentiment) + uplift empaque/claims
    # Uplift: empaque+claims pueden sumar hasta ~+18 puntos a conexión (controlado)
    uplift = clip((pack_emotion - 50)/50, -0.35, 0.35)  # -35% a +35% relativo
    base_rating = 6.5  # neutral
    rating_conexion = st.slider("Rating conexión producto (1-10)", 1, 10, 7)
    sentiment_score = st.select_slider("Sentimiento del producto (-1/0/1)", options=[-1,0,1], value=1)

    base_conexion = (rating_conexion/10)*70 + sentiment_score*15 + 5
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
        "canal": str(canal).lower()
    }])

    st.markdown("#### Scores calculados")
    s1, s2, s3 = st.columns(3)
    s1.metric("Claims Score", f"{cscore:.1f}/100")
    s2.metric("Emotion Pack Score", f"{pack_emotion:.1f}/100")
    s3.metric("Conexión final (con uplift)", f"{conexion_score:.1f}/100")

    if st.button("🚀 Simular"):
        p = float(success_model.predict_proba(entrada)[0][1])
        pred = int(success_model.predict(entrada)[0])

        ventas = max(0, round(float(sales_model.predict(entrada)[0])))
        ingresos = ventas * float(precio)
        utilidad = ventas * (float(precio) * (float(margen_pct)/100.0))

        r1, r2, r3 = st.columns(3)
        r1.metric("Prob. éxito", safe_percent(p))
        r2.metric("Predicción", "✅ Éxito" if pred else "⚠️ Riesgo")
        r3.metric("Ventas (unidades)", f"{ventas:,.0f}")

        r4, r5 = st.columns(2)
        r4.metric("Ingresos ($)", f"${ingresos:,.0f}")
        r5.metric("Utilidad ($)", f"${utilidad:,.0f}")

        st.caption("Nota: las recomendaciones de empaque/claims ajustan la conexión emocional y esto impacta los modelos.")
        st.dataframe(entrada, use_container_width=True)

# ============================================================
# 📦 Pack Lab
# ============================================================
with tab_pack:
    st.subheader("📦 Pack Lab (sube tu empaque y te doy recomendaciones)")
    st.write("Sube una imagen del empaque. Calculamos métricas visuales (sin OCR pesado) y devolvemos recomendaciones prácticas.")

    img_file = st.file_uploader("Sube imagen del empaque (PNG/JPG)", type=["png","jpg","jpeg"])
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
        recs = []
        if scores["pack_legibility_score"] < 60:
            recs.append("• Sube legibilidad: más contraste entre texto/fondo, tipografías más gruesas, menos elementos alrededor del claim principal.")
        if scores["pack_clarity_score"] < 60:
            recs.append("• Mejora claridad: reduce ruido visual (menos bloques/íconos), deja “aire”, y limita a 2–3 claims máximo.")
        if scores["pack_shelf_pop_score"] < 60:
            recs.append("• Sube shelf pop: usa un color acento, mejora contraste y evita que el pack quede demasiado oscuro o lavado.")
        if m["edge_density"] > 0.28:
            recs.append("• Hay saturación visual: demasiados bordes → parece “ruidoso”. Simplifica fondos, patrones y microtextos.")
        if not recs:
            recs.append("• Va bastante bien visualmente. Tu siguiente mejora está en jerarquía: Marca → beneficio → variedad/sabor → prueba/credencial.")

        st.write("\n".join(recs))

        st.markdown("### Cómo usar esto en tu simulador")
        st.code(
            f"Legibilidad={scores['pack_legibility_score']}, ShelfPop={scores['pack_shelf_pop_score']}, Claridad={scores['pack_clarity_score']}",
            language="text"
        )

# ============================================================
# 🏷️ Claims Lab
# ============================================================
with tab_claims:
    st.subheader("🏷️ Claims Lab (recomendación de claims ganadores)")
    st.write("Elige segmento + canal y te recomiendo claims; luego puedes simular su impacto (via conexión).")

    c1, c2 = st.columns(2)
    segmento = c1.selectbox("Segmento", ["fit", "kids", "premium", "value"], 0)
    canal = c2.selectbox("Canal", ["retail","marketplace"], 0)

    recs = recommend_claims(segmento, canal, max_claims=8)
    st.markdown("### Top claims recomendados")
    rec_df = pd.DataFrame(recs, columns=["claim", "score_base"])
    rec_df["score_base"] = (rec_df["score_base"]*100).round(1)
    st.dataframe(rec_df, use_container_width=True)

    selected = st.multiselect("Selecciona 2-3 claims para tu pack", rec_df["claim"].tolist(), default=rec_df["claim"].tolist()[:2])
    cscore = claims_score(selected, canal)
    st.metric("Claims Score (claridad + canal fit)", f"{cscore:.1f}/100")

    st.markdown("### Reglas prácticas")
    st.write(
        "- Ideal: **2–3 claims máximo** (claridad)\n"
        "- 1 claim funcional (ej. fibra/proteína) + 1 claim limpio (ej. sin azúcar añadida)\n"
        "- Marketplace tolera más detalle, retail exige más simplicidad\n"
    )

    st.warning("Nota: Los claims deben validarse con normativa/etiquetado aplicable. Esto es recomendación comercial, no legal.")

# ============================================================
# 🧪 Experimentos (A/B simple en memoria)
# ============================================================
with tab_exp:
    st.subheader("🧪 Experimentos (A/B) — aprende qué pack/claim gana")
    st.write("Aquí registras resultados rápidos (por ahora en memoria). Después lo conectamos a SQL para histórico real.")

    if "experiments" not in st.session_state:
        st.session_state.experiments = []

    c1, c2, c3 = st.columns(3)
    exp_name = c1.text_input("Nombre experimento", value="Test Pack v1 vs v2")
    variant = c2.selectbox("Variante", ["A", "B"], 0)
    metric = c3.selectbox("Métrica", ["intencion_compra", "conexion_pack", "ventas_piloto"], 0)

    v1, v2, v3 = st.columns(3)
    marca = v1.selectbox("Marca", sorted(df["marca"].unique().tolist()), 0)
    canal = v2.selectbox("Canal", sorted(df["canal"].unique().tolist()), 0)
    value = v3.number_input("Valor observado", value=7.0, step=0.1)

    if st.button("➕ Guardar medición"):
        st.session_state.experiments.append({
            "experimento": exp_name,
            "variante": variant,
            "marca": marca,
            "canal": canal,
            "metrica": metric,
            "valor": float(value)
        })
        st.success("Guardado.")

    if st.session_state.experiments:
        exp_df = pd.DataFrame(st.session_state.experiments)
        st.dataframe(exp_df, use_container_width=True)

        st.markdown("### Resumen rápido")
        try:
            pivot = exp_df.pivot_table(index=["experimento","metrica"], columns="variante", values="valor", aggfunc="mean")
            pivot["lift_B_vs_A"] = (pivot.get("B") - pivot.get("A"))
            st.dataframe(pivot.round(3), use_container_width=True)
        except Exception:
            st.info("Necesitas registros en A y B para calcular lift.")

        st.caption("Siguiente paso: guardar esto en SQL y entrenar un modelo que aprenda automáticamente el mejor claim/pack.")
    else:
        st.info("Aún no hay experimentos guardados.")

# ============================================================
# 📂 Datos
# ============================================================
with tab_data:
    st.subheader("📂 Datos + Descarga")
    st.download_button(
        label="📥 Descargar dataset (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="dataset_con_ventas.csv",
        mime="text/csv"
    )
    st.dataframe(df.head(300), use_container_width=True)

# ============================================================
# 🧠 Modelo
# ============================================================
with tab_model:
    st.subheader("🧠 Diagnóstico")
    st.markdown("**Matriz de confusión (éxito)**")
    st.dataframe(pd.DataFrame(cm, index=["Real 0","Real 1"], columns=["Pred 0","Pred 1"]), use_container_width=True)

    st.markdown("**MAE ventas**")
    st.write(f"Error absoluto medio: **{mae:,.0f}** unidades (mientras menor, mejor).")
