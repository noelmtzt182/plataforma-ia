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
# Config Streamlit
# ----------------------------
st.set_page_config(page_title="IA Cereales | Éxito + Conexión", layout="wide")

DATA_PATH_DEFAULT = "mercado_cereales_5000.csv"  # <-- CSV de cereales

# ----------------------------
# Cargar datos
# ----------------------------
@st.cache_data
def load_data(path_or_file) -> pd.DataFrame:
    df = pd.read_csv(path_or_file).copy()

    # Limpieza defensiva
    for c in ["marca", "categoria", "canal", "estacionalidad"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.lower()

    # Validar columnas mínimas esperadas (para tu dataset de cereales)
    required = {
        "marca","canal","precio","costo","margen","margen_pct",
        "competencia","demanda","tendencia","estacionalidad",
        "rating_conexion","comentario","sentiment_score",
        "conexion_score","conexion_alta","score_latente","exito"
    }
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(f"Faltan columnas en el CSV: {missing}")

    # Asegurar tipos numéricos clave
    num_cols = ["precio","costo","margen","margen_pct","competencia","demanda","tendencia",
                "rating_conexion","sentiment_score","conexion_score","conexion_alta","score_latente","exito"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["marca","canal","precio","competencia","demanda","tendencia","margen_pct","conexion_score","rating_conexion","sentiment_score","exito"])
    df["exito"] = df["exito"].astype(int)

    return df

# ----------------------------
# Entrenar modelo
# ----------------------------
@st.cache_resource
def train_model(df: pd.DataFrame):
    # Features: negocio + emoción + categóricas (marca, canal)
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
    cat_cols = ["marca", "canal"]

    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=350,
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

    return clf, acc, auc, cm

def clip(v, a, b):
    return max(a, min(b, v))

# ----------------------------
# Sidebar: carga de CSV
# ----------------------------
st.sidebar.title("⚙️ Configuración")
uploaded = st.sidebar.file_uploader("Sube tu CSV (mercado_cereales_5000.csv)", type=["csv"])

use_default = st.sidebar.checkbox(
    "Usar archivo local (mercado_cereales_5000.csv)",
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
# Entrenar
# ----------------------------
try:
    clf, acc, auc, cm = train_model(df)
except Exception as e:
    st.error(f"Error entrenando el modelo: {e}")
    st.stop()

# ----------------------------
# Header
# ----------------------------
st.title("🥣 IA para Cereales | Éxito + Conexión Emocional")
st.caption("Predicción de éxito por marca/canal + simulación what-if + insights (retail vs marketplace)")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Registros", f"{len(df):,}")
k2.metric("Precisión (test)", f"{acc*100:.2f}%")
k3.metric("AUC (test)", f"{auc:.3f}")
k4.metric("Éxito (base)", f"{df['exito'].mean()*100:.1f}%")

st.divider()

tab_sim, tab_ins, tab_data, tab_model = st.tabs(
    ["🧪 Simulador", "📊 Insights", "📂 Datos", "🧠 Modelo"]
)

# ============================================================
# TAB: Simulador
# ============================================================
with tab_sim:
    st.subheader("🧪 Simulador What-If (Cereales)")
    st.write("Simula un escenario por **marca** y **canal** para obtener **probabilidad de éxito** y **conexión emocional**.")

    marcas = sorted(df["marca"].dropna().unique().tolist())
    canales = sorted(df["canal"].dropna().unique().tolist())
    estacionalidades = sorted(df["estacionalidad"].dropna().unique().tolist())

    c1, c2, c3 = st.columns(3)
    marca = c1.selectbox("Marca", marcas, index=0)
    canal = c2.selectbox("Canal", canales, index=0)
    estacionalidad = c3.selectbox("Estacionalidad (informativa)", estacionalidades, index=0)

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
    comentario = e2.text_input("Comentario (opcional)", value="Me encanta el sabor y la textura")

    # Sentimiento simple: -1, 0, 1 (coherente con el dataset)
    pos_words = ["encanta", "me gusta", "buena calidad", "me identifico", "excelente", "premium"]
    neg_words = ["no me gustó", "no me convence", "caro", "no conecté", "no lo volvería", "malo"]

    def sentimiento_simple(text: str) -> int:
        t = (text or "").lower()
        score = 0
        for w in pos_words:
            if w in t:
                score += 1
        for w in neg_words:
            if w in t:
                score -= 1
        if score > 0:
            return 1
        if score < 0:
            return -1
        return 0

    sentiment_score = sentimiento_simple(comentario)

    # Conexión emocional (0-100) — cereal físico: bonus 5
    conexion_score = clip(round((rating_conexion / 10) * 70 + sentiment_score * 15 + 5, 2), 0, 100)

    entrada = pd.DataFrame([{
        "precio": float(precio),
        "competencia": float(competencia),
        "demanda": float(demanda),
        "tendencia": float(tendencia),
        "margen_pct": float(margen_pct),
        "conexion_score": float(conexion_score),
        "rating_conexion": float(rating_conexion),
        "sentiment_score": float(sentiment_score),
        "marca": marca,
        "canal": canal
    }])

    if st.button("🚀 Simular"):
        proba = float(clf.predict_proba(entrada)[0][1])
        pred = int(clf.predict(entrada)[0])

        r1, r2, r3 = st.columns(3)
        r1.metric("Probabilidad de éxito", f"{proba*100:.2f}%")
        r2.metric("Predicción", "✅ Éxito" if pred == 1 else "⚠️ Riesgo")
        r3.metric("Conexión emocional", f"{conexion_score:.1f} / 100")

        st.caption(f"Sentimiento: {sentiment_score:+d}  |  Estacionalidad (informativa): {estacionalidad}")
        st.markdown("#### Entrada usada por el modelo")
        st.dataframe(entrada, use_container_width=True)

# ============================================================
# TAB: Insights
# ============================================================
with tab_ins:
    st.subheader("📊 Insights (Marca, Canal, Conexión, Éxito)")

    left, right = st.columns(2)

    with left:
        st.markdown("**Ranking por marca (Conexión promedio)**")
        ins_marca = (
            df.groupby("marca")[["conexion_score"]]
            .mean()
            .sort_values("conexion_score", ascending=False)
            .round(2)
        )
        st.dataframe(ins_marca, use_container_width=True)

        st.markdown("**Ranking por marca (Éxito %)**")
        ex_marca = (
            df.groupby("marca")[["exito"]]
            .mean()
            .sort_values("exito", ascending=False)
            .round(3)
        )
        ex_marca["exito_%"] = (ex_marca["exito"] * 100).round(1)
        st.dataframe(ex_marca[["exito_%"]], use_container_width=True)

    with right:
        st.markdown("**Marca + Canal (Conexión promedio)**")
        ins_mc = (
            df.groupby(["marca", "canal"])[["conexion_score"]]
            .mean()
            .sort_values("conexion_score", ascending=False)
            .round(2)
        )
        st.dataframe(ins_mc.head(20), use_container_width=True)

        st.markdown("**Marca + Canal (Éxito %)**")
        ex_mc = (
            df.groupby(["marca", "canal"])[["exito"]]
            .mean()
            .sort_values("exito", ascending=False)
            .round(3)
        )
        ex_mc["exito_%"] = (ex_mc["exito"] * 100).round(1)
        st.dataframe(ex_mc.head(20)[["exito_%"]], use_container_width=True)

    st.divider()
    st.markdown("### Distribuciones")
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**Histograma: Conexión emocional**")
        st.bar_chart(df["conexion_score"].value_counts().sort_index())
    with d2:
        st.markdown("**Histograma: Precio**")
        st.bar_chart(df["precio"].round().value_counts().sort_index().head(80))

# ============================================================
# TAB: Datos
# ============================================================
with tab_data:
    st.subheader("📂 Explorador del dataset")

    f1, f2, f3 = st.columns(3)
    fmarca = f1.multiselect("Filtrar marca", sorted(df["marca"].unique().tolist()), default=[])
    fcanal = f2.multiselect("Filtrar canal", sorted(df["canal"].unique().tolist()), default=[])
    fex = f3.selectbox("Filtrar éxito", ["Todos", "Éxito (1)", "No éxito (0)"], index=0)

    dff = df.copy()
    if fmarca:
        dff = dff[dff["marca"].isin(fmarca)]
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
    try:
        rf = clf.named_steps["model"]
        pre = clf.named_steps["preprocessor"]

        ohe = pre.named_transformers_["cat"]
        cat_features = ohe.get_feature_names_out(["marca", "canal"]).tolist()

        feature_names = [
            "precio","competencia","demanda","tendencia","margen_pct",
            "conexion_score","rating_conexion","sentiment_score"
        ] + cat_features

        importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
        st.dataframe(importances.head(30).round(4), use_container_width=True)
    except Exception:
        st.info("No se pudieron mostrar importancias (depende de versión de scikit-learn).")
