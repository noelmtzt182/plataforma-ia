import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

# ----------------------------
# Config
# ----------------------------
st.set_page_config(
    page_title="Plataforma IA | Desarrollo de Producto",
    layout="wide"
)

DATA_PATH_DEFAULT = "mercado_avanzado_emocional.csv"

# ----------------------------
# Carga de datos
# ----------------------------
@st.cache_data
def load_data(path_or_file) -> pd.DataFrame:
    df = pd.read_csv(path_or_file)
    # Limpieza mínima defensiva
    df = df.copy()
    # normalizar strings
    for c in ["tipo_producto", "canal", "estacionalidad"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.lower()
    return df

# ----------------------------
# Entrenamiento del modelo
# ----------------------------
@st.cache_resource
def train_model(df: pd.DataFrame):
    required_cols = [
        "precio","competencia","demanda","tendencia","margen_pct",
        "conexion_score","rating_conexion","sentiment_score",
        "tipo_producto","canal","exito"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en el CSV: {missing}")

    features = [
        "precio","competencia","demanda","tendencia","margen_pct",
        "conexion_score","rating_conexion","sentiment_score",
        "tipo_producto","canal"
    ]
    X = df[features]
    y = df["exito"].astype(int)

    num_cols = [
        "precio","competencia","demanda","tendencia","margen_pct",
        "conexion_score","rating_conexion","sentiment_score"
    ]
    cat_cols = ["tipo_producto","canal"]

    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced_subsample"
    )

    clf = Pipeline(steps=[("preprocessor", pre), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    proba = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, proba)
    cm = confusion_matrix(y_test, pred)

    return clf, acc, auc, cm, df

# ----------------------------
# Utilidades
# ----------------------------
def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def clip(v, a, b):
    return max(a, min(b, v))

# ----------------------------
# UI - Sidebar
# ----------------------------
st.sidebar.title("⚙️ Configuración")

uploaded = st.sidebar.file_uploader(
    "Sube tu CSV (mercado_avanzado_emocional.csv)",
    type=["csv"]
)

use_default = st.sidebar.checkbox(
    "Usar archivo local (mercado_avanzado_emocional.csv)",
    value=(uploaded is None)
)

if uploaded is not None:
    df = load_data(uploaded)
else:
    if use_default:
        df = load_data(DATA_PATH_DEFAULT)
    else:
        st.stop()

# ----------------------------
# Entrenar modelo
# ----------------------------
try:
    clf, acc, auc, cm, df = train_model(df)
except Exception as e:
    st.error(f"Error entrenando el modelo: {e}")
    st.stop()

# ----------------------------
# Header
# ----------------------------
st.title("🚀 Plataforma IA para Desarrollo de Producto (con Conexión Emocional)")
st.caption("Predicción de éxito + conexión emocional + simulación what-if + insights por tipo/canal")

# KPIs
k1, k2, k3, k4 = st.columns(4)
k1.metric("Registros", f"{len(df):,}")
k2.metric("Precisión (test)", f"{acc*100:.2f}%")
k3.metric("AUC (test)", f"{auc:.3f}")
k4.metric("Éxito (base)", f"{(df['exito'].mean()*100):.1f}%")

st.divider()

# ----------------------------
# Tabs
# ----------------------------
tab_sim, tab_ins, tab_data, tab_model = st.tabs(
    ["🧪 Simulador", "📊 Insights", "📂 Datos", "🧠 Modelo"]
)

# ============================================================
# TAB: Simulador
# ============================================================
with tab_sim:
    st.subheader("🧪 Simulador What-If (Producto / Digital / Servicio)")
    st.write("Ajusta variables y predice **probabilidad de éxito**. También puedes estimar la **conexión emocional**.")

    # Opciones desde el dataset (para evitar valores que el modelo no conozca)
    tipos = sorted(df["tipo_producto"].dropna().unique().tolist())
    canales = sorted(df["canal"].dropna().unique().tolist())

    c1, c2, c3 = st.columns(3)
    tipo_producto = c1.selectbox("Tipo de producto", tipos, index=0)
    canal = c2.selectbox("Canal", canales, index=0)
    estacionalidad = c3.selectbox(
        "Estacionalidad (solo informativa)",
        sorted(df["estacionalidad"].dropna().unique().tolist())
        if "estacionalidad" in df.columns else ["media"],
        index=0
    )

    st.markdown("### Variables de negocio")
    b1, b2, b3, b4, b5 = st.columns(5)
    precio = b1.number_input("Precio", value=float(df["precio"].median()), step=1.0)
    competencia = b2.slider("Competencia (1-10)", 1, 10, int(df["competencia"].median()))
    demanda = b3.slider("Demanda (10-100)", 10, 100, int(df["demanda"].median()))
    tendencia = b4.slider("Tendencia (20-100)", 20, 100, int(df["tendencia"].median()))
    margen_pct = b5.slider("Margen %", 0, 90, int(np.clip(df["margen_pct"].median(), 0, 90)))

    st.markdown("### Variables emocionales")
    e1, e2 = st.columns([1, 2])
    rating_conexion = e1.slider("Rating conexión (1-10)", 1, 10, int(np.clip(df["rating_conexion"].median(), 1, 10)))
    comentario = e2.text_input(
        "Comentario (opcional)",
        value="Me encanta este producto, lo volvería a elegir"
    )

    # Sentimiento (simple, mismo enfoque que dataset)
    pos_words = ["encanta", "muy bien", "positiva", "volvería", "identifico", "me identifico"]
    neg_words = ["no conecté", "no me convence", "no lo volvería", "no fue", "no me sentí", "no me convenció"]

    def sentimiento_simple(text: str) -> float:
        t = (text or "").lower()
        score = 0
        for w in pos_words:
            if w in t:
                score += 1
        for w in neg_words:
            if w in t:
                score -= 1
        return float(clip(score / 2, -1, 1))

    sent_score = sentimiento_simple(comentario)

    # conexion_score (0-100) compatible con tu dataset
    bonus_tipo = {"fisico": 5, "digital": 3, "servicio": 7}.get(tipo_producto, 0)
    rating_norm = (rating_conexion / 10) * 70
    sentiment_points = sent_score * 15
    conexion_score = clip(round(rating_norm + sentiment_points + bonus_tipo, 2), 0, 100)

    entrada = pd.DataFrame([{
        "precio": safe_float(precio),
        "competencia": safe_float(competencia),
        "demanda": safe_float(demanda),
        "tendencia": safe_float(tendencia),
        "margen_pct": safe_float(margen_pct),
        "conexion_score": safe_float(conexion_score),
        "rating_conexion": safe_float(rating_conexion),
        "sentiment_score": safe_float(sent_score),
        "tipo_producto": tipo_producto,
        "canal": canal
    }])

    if st.button("🚀 Simular"):
        proba = float(clf.predict_proba(entrada)[0][1])
        pred = int(clf.predict(entrada)[0])

        r1, r2, r3 = st.columns(3)
        r1.metric("Probabilidad de éxito", f"{proba*100:.2f}%")
        r2.metric("Predicción", "✅ Éxito" if pred == 1 else "⚠️ Riesgo")
        r3.metric("Conexión emocional", f"{conexion_score:.1f} / 100")

        st.caption(f"Sentimiento estimado: {sent_score:+.2f}  |  Estacionalidad (informativa): {estacionalidad}")

        st.markdown("#### Entrada usada por el modelo")
        st.dataframe(entrada, use_container_width=True)

# ============================================================
# TAB: Insights
# ============================================================
with tab_ins:
    st.subheader("📊 Insights (Conexión y Éxito)")

    a1, a2 = st.columns(2)

    with a1:
        st.markdown("**Conexión promedio por tipo de producto**")
        ins_tipo = df.groupby("tipo_producto")[["conexion_score"]].mean().sort_values("conexion_score", ascending=False).round(2)
        st.dataframe(ins_tipo, use_container_width=True)

        st.markdown("**Éxito promedio por tipo de producto**")
        ex_tipo = df.groupby("tipo_producto")[["exito"]].mean().sort_values("exito", ascending=False).round(3)
        ex_tipo["exito_%"] = (ex_tipo["exito"] * 100).round(1)
        st.dataframe(ex_tipo[["exito_%"]], use_container_width=True)

    with a2:
        st.markdown("**Conexión promedio por tipo + canal**")
        ins_tc = df.groupby(["tipo_producto", "canal"])[["conexion_score"]].mean().sort_values("conexion_score", ascending=False).round(2)
        st.dataframe(ins_tc.head(15), use_container_width=True)

        st.markdown("**Éxito por tipo + canal**")
        ex_tc = df.groupby(["tipo_producto", "canal"])[["exito"]].mean().sort_values("exito", ascending=False).round(3)
        ex_tc["exito_%"] = (ex_tc["exito"] * 100).round(1)
        st.dataframe(ex_tc.head(15)[["exito_%"]], use_container_width=True)

    st.divider()
    st.markdown("### Distribuciones")
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**Histograma: Conexión emocional**")
        st.bar_chart(df["conexion_score"].value_counts().sort_index())
    with d2:
        st.markdown("**Histograma: Precio**")
        st.bar_chart(df["precio"].round().value_counts().sort_index().head(60))

# ============================================================
# TAB: Datos
# ============================================================
with tab_data:
    st.subheader("📂 Dataset")
    st.write("Filtro rápido para explorar el dataset en la app.")

    f1, f2, f3 = st.columns(3)
    ftipo = f1.multiselect("Filtrar tipo_producto", sorted(df["tipo_producto"].unique().tolist()), default=[])
    fcanal = f2.multiselect("Filtrar canal", sorted(df["canal"].unique().tolist()), default=[])
    fex = f3.selectbox("Filtrar éxito", ["Todos", "Éxito (1)", "No éxito (0)"], index=0)

    dff = df.copy()
    if ftipo:
        dff = dff[dff["tipo_producto"].isin(ftipo)]
    if fcanal:
        dff = dff[dff["canal"].isin(fcanal)]
    if fex == "Éxito (1)":
        dff = dff[dff["exito"] == 1]
    elif fex == "No éxito (0)":
        dff = dff[dff["exito"] == 0]

    st.dataframe(dff.head(500), use_container_width=True)
    st.caption(f"Mostrando {min(len(dff), 500)} de {len(dff)} registros filtrados.")

# ============================================================
# TAB: Modelo
# ============================================================
with tab_model:
    st.subheader("🧠 Diagnóstico del modelo")

    st.markdown("**Matriz de confusión (test)**")
    cm_df = pd.DataFrame(cm, index=["Real 0", "Real 1"], columns=["Pred 0", "Pred 1"])
    st.dataframe(cm_df, use_container_width=True)

    st.markdown("**Importancias de features (aprox.)**")
    # Recuperar importancias del RandomForest dentro del Pipeline
    try:
        rf = clf.named_steps["model"]
        pre = clf.named_steps["preprocessor"]

        # Nombres de features después del OneHot
        ohe = pre.named_transformers_["cat"]
        cat_features = ohe.get_feature_names_out(["tipo_producto", "canal"]).tolist()
        feature_names = [
            "precio","competencia","demanda","tendencia","margen_pct",
            "conexion_score","rating_conexion","sentiment_score"
        ] + cat_features

        importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
        st.dataframe(importances.head(25).round(4), use_container_width=True)
    except Exception:
        st.info("No se pudieron mostrar importancias (depende de versión de scikit-learn).")
