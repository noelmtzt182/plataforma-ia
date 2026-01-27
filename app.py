import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, mean_absolute_error

# ----------------------------
# Config Streamlit
# ----------------------------
st.set_page_config(page_title="IA Cereales | Éxito + Conexión + Ventas", layout="wide")
DATA_PATH_DEFAULT = "mercado_cereales_5000.csv"  # CSV de cereales en tu repo

# ----------------------------
# Utilidades
# ----------------------------
def clip(v, a, b):
    return max(a, min(b, v))

# ----------------------------
# Cargar datos
# ----------------------------
@st.cache_data
def load_data(path_or_file) -> pd.DataFrame:
    df = pd.read_csv(path_or_file).copy()

    # Limpieza defensiva de strings
    for c in ["marca", "categoria", "canal", "estacionalidad"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.lower()

    # Validar columnas mínimas esperadas (dataset cereales)
    required = {
        "marca","categoria","canal","precio","costo","margen","margen_pct",
        "competencia","demanda","tendencia","estacionalidad",
        "rating_conexion","comentario","sentiment_score",
        "conexion_score","conexion_alta","score_latente","exito"
    }
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(f"Faltan columnas en el CSV: {missing}")

    # Asegurar tipos numéricos clave
    num_cols = [
        "precio","costo","margen","margen_pct","competencia","demanda","tendencia",
        "rating_conexion","sentiment_score","conexion_score","conexion_alta",
        "score_latente","exito"
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=[
        "marca","canal","precio","competencia","demanda","tendencia","margen_pct",
        "conexion_score","rating_conexion","sentiment_score","exito"
    ])
    df["exito"] = df["exito"].astype(int)
    return df

# ----------------------------
# Agregar columnas de venta (si no existen)
# ----------------------------
def add_sales_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "ventas_unidades" in df.columns and "ventas_ingresos" in df.columns and "utilidad" in df.columns:
        # Ya existen; no recalcular
        return df

    # Factores por canal (ajusta a tu realidad)
    canal_factor = df["canal"].map({
        "retail": 1.00,
        "marketplace": 0.85
    }).fillna(1.0)

    # Factor emocional: 50 neutral; arriba sube; abajo baja
    emo_factor = (1 + (df["conexion_score"] - 50) / 120).clip(0.6, 1.5)

    # Factor marca (ejemplo; puedes calibrarlo con datos reales)
    marca_factor_map = {
        "goldengrain": 1.08, "fitmorning": 1.06, "fiberplus": 1.05, "vitalmix": 1.04,
        "kidsstar": 1.03, "chocoboom": 1.02,
        "cerealnova": 1.00, "crunchmax": 0.99, "honeyoats": 0.98, "corncrisp": 0.97
    }
    marca_factor = df["marca"].map(marca_factor_map).fillna(1.0)

    # Base de ventas (proxy): demanda * escala
    # Escala: 120 unidades por punto de demanda (mensual). Ajusta si quieres.
    base = df["demanda"] * 120

    # Ruido realista
    rng = np.random.default_rng(2026)
    ruido = rng.normal(0, 180, len(df))

    # Ventas mensuales estimadas (unidades)
    df["ventas_unidades"] = (base * canal_factor * emo_factor * marca_factor + ruido).clip(0).round(0).astype(int)

    # Ingresos y utilidad
    df["ventas_ingresos"] = (df["ventas_unidades"] * df["precio"]).round(2)
    df["utilidad"] = (df["ventas_unidades"] * df["margen"]).round(2)

    return df

# ----------------------------
# Entrenar modelo de ÉXITO (clasificación)
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
    cat_cols = ["marca", "canal"]

    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    clf_model = RandomForestClassifier(
        n_estimators=350,
        random_state=42,
        class_weight="balanced_subsample"
    )

    clf = Pipeline(steps=[("preprocessor", pre), ("model", clf_model)])

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

# ----------------------------
# Entrenar modelo de VENTAS (regresión)
# ----------------------------
@st.cache_resource
def train_sales_model(df: pd.DataFrame):
    if "ventas_unidades" not in df.columns:
        raise ValueError("No existe 'ventas_unidades' en el dataset. Asegúrate de ejecutar add_sales_columns(df).")

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
    cat_cols = ["marca", "canal"]

    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    reg_model = RandomForestRegressor(
        n_estimators=350,
        random_state=42
    )

    reg = Pipeline(steps=[("preprocessor", pre), ("model", reg_model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    reg.fit(X_train, y_train)

    pred = reg.predict(X_test)
    mae = mean_absolute_error(y_test, pred)

    return reg, mae

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("⚙️ Configuración")
uploaded = st.sidebar.file_uploader("Sube tu CSV (mercado_cereales_5000.csv)", type=["csv"])

use_default = st.sidebar.checkbox(
    "Usar archivo local del repo (mercado_cereales_5000.csv)",
    value=(uploaded is None)
)

# Cargar DF
if uploaded is not None:
    df = load_data(uploaded)
else:
    if use_default:
        df = load_data(DATA_PATH_DEFAULT)
    else:
        st.stop()

# Agregar ventas
df = add_sales_columns(df)

# Entrenar modelos
try:
    success_model, acc, auc, cm = train_success_model(df)
    sales_model, mae = train_sales_model(df)
except Exception as e:
    st.error(f"Error entrenando modelos: {e}")
    st.stop()

# ----------------------------
# Header
# ----------------------------
st.title("🥣 Cereales | Éxito + Conexión Emocional + Ventas")
st.caption("Predicción de éxito por marca/canal + simulación what-if + ventas estimadas + insights (retail vs marketplace)")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Registros", f"{len(df):,}")
k2.metric("Precisión (test)", f"{acc*100:.2f}%")
k3.metric("AUC (test)", f"{auc:.3f}")
k4.metric("Éxito (base)", f"{df['exito'].mean()*100:.1f}%")
k5.metric("MAE ventas", f"{mae:,.0f} u.")

st.divider()

tab_sim, tab_ins, tab_data, tab_model = st.tabs(
    ["🧪 Simulador", "📊 Insights", "📂 Datos", "🧠 Modelo"]
)

# ============================================================
# TAB: Simulador
# ============================================================
with tab_sim:
    st.subheader("🧪 Simulador What-If (Marca + Canal + Ventas)")
    st.write("Simula un escenario y obtiene **probabilidad de éxito**, **conexión emocional** y **ventas estimadas**.")

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

    # Calcular margen $ (aprox) desde margen_pct (si no tenemos costo real en simulación)
    margen_unitario_est = float(precio) * (float(margen_pct) / 100.0)

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
        proba = float(success_model.predict_proba(entrada)[0][1])
        pred = int(success_model.predict(entrada)[0])

        # Predicción ventas (unidades)
        ventas_pred = float(sales_model.predict(entrada)[0])
        ventas_pred = max(0, round(ventas_pred))

        ingresos_pred = ventas_pred * float(precio)
        utilidad_pred = ventas_pred * margen_unitario_est

        r1, r2, r3 = st.columns(3)
        r1.metric("Probabilidad de éxito", f"{proba*100:.2f}%")
        r2.metric("Predicción", "✅ Éxito" if pred == 1 else "⚠️ Riesgo")
        r3.metric("Conexión emocional", f"{conexion_score:.1f} / 100")

        v1, v2, v3 = st.columns(3)
        v1.metric("Ventas estimadas (unidades)", f"{ventas_pred:,.0f}")
        v2.metric("Ingresos estimados ($)", f"${ingresos_pred:,.0f}")
        v3.metric("Utilidad estimada ($)", f"${utilidad_pred:,.0f}")

        st.caption(f"Sentimiento: {sentiment_score:+d}  |  Estacionalidad (informativa): {estacionalidad}")

        st.markdown("#### Entrada usada por los modelos")
        st.dataframe(entrada, use_container_width=True)

# ============================================================
# TAB: Insights
# ============================================================
with tab_ins:
    st.subheader("📊 Insights (Marca, Canal, Conexión, Éxito, Ventas)")

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

        st.markdown("**Ranking por marca (Ventas unidades promedio)**")
        v_marca = (
            df.groupby("marca")[["ventas_unidades"]]
            .mean()
            .sort_values("ventas_unidades", ascending=False)
            .round(0)
        )
        st.dataframe(v_marca, use_container_width=True)

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

        st.markdown("**Marca + Canal (Ventas unidades promedio)**")
        v_mc = (
            df.groupby(["marca", "canal"])[["ventas_unidades"]]
            .mean()
            .sort_values("ventas_unidades", ascending=False)
            .round(0)
        )
        st.dataframe(v_mc.head(20), use_container_width=True)

    st.divider()
    st.markdown("### Distribuciones")
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**Histograma: Conexión emocional**")
        st.bar_chart(df["conexion_score"].value_counts().sort_index())
    with d2:
        st.markdown("**Histograma: Ventas (unidades)**")
        st.bar_chart(df["ventas_unidades"].clip(0, 20000).round(-1).value_counts().sort_index().head(120))

# ============================================================
# TAB: Datos
# ============================================================
with tab_data:
    st.subheader("📂 Explorador del dataset + Descarga CSV")

    # Botón de descarga
    st.download_button(
        label="📥 Descargar dataset completo (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="mercado_cereales_5000_con_ventas.csv",
        mime="text/csv"
    )

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

    st.markdown("**Matriz de confusión (test) — Éxito**")
    cm_df = pd.DataFrame(cm, index=["Real 0", "Real 1"], columns=["Pred 0", "Pred 1"])
    st.dataframe(cm_df, use_container_width=True)

    st.markdown("**Importancias de features (aprox.) — Modelo Éxito**")
    try:
        rf = success_model.named_steps["model"]
        pre = success_model.named_steps["preprocessor"]

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

    st.markdown("**MAE — Modelo Ventas**")
    st.write(f"Error absoluto medio (MAE): **{mae:,.0f} unidades** (mientras más bajo, mejor)")
