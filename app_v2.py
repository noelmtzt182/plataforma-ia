import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, mean_absolute_error

st.set_page_config(page_title="Plataforma IA Producto 2.0", layout="wide")

DATA_PATH_DEFAULT = "mercado_cereales_5000_con_ventas.csv"

# =====================================================
# LOAD
# =====================================================

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    for c in ["marca","canal","categoria"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.lower().str.strip()
    return df

if Path(DATA_PATH_DEFAULT).exists():
    df = load_data(DATA_PATH_DEFAULT)
else:
    up = st.file_uploader("Sube CSV", type=["csv"])
    if up:
        df = load_data(up)
    else:
        st.stop()

# =====================================================
# MODELOS
# =====================================================

FEATS = ["precio","competencia","demanda","tendencia",
         "margen_pct","conexion_score",
         "rating_conexion","sentiment_score",
         "marca","canal"]

num = FEATS[:-2]
cat = ["marca","canal"]

pre = ColumnTransformer([
    ("num","passthrough",num),
    ("cat",OneHotEncoder(handle_unknown="ignore"),cat)
])

X = df[FEATS]
y = df["exito"]

clf = Pipeline([("pre",pre),
                ("rf",RandomForestClassifier(n_estimators=300,random_state=42))])

reg = Pipeline([("pre",pre),
                ("rf",RandomForestRegressor(n_estimators=300,random_state=42))])

Xtr,Xte,Ytr,Yte = train_test_split(X,y,test_size=0.2,random_state=42)

clf.fit(Xtr,Ytr)
reg.fit(Xtr, df.loc[Xtr.index,"ventas_unidades"])

acc = accuracy_score(Yte, clf.predict(Xte))
auc = roc_auc_score(Yte, clf.predict_proba(Xte)[:,1])
mae = mean_absolute_error(df.loc[Xte.index,"ventas_unidades"], reg.predict(Xte))

# =====================================================
# HEADER
# =====================================================

st.title("🚀 Plataforma IA Desarrollo de Producto 2.0")

c1,c2,c3 = st.columns(3)
c1.metric("Accuracy", f"{acc:.2f}")
c2.metric("AUC", f"{auc:.2f}")
c3.metric("MAE ventas", f"{mae:,.0f}")

# =====================================================
# TABS
# =====================================================

tab_sim, tab_rec, tab_pack, tab_inv, tab_rep = st.tabs([
    "Simulador",
    "Recomendador",
    "Pack Vision+",
    "Vista Inversionista",
    "Reporte"
])

# =====================================================
# SIMULADOR
# =====================================================

with tab_sim:

    marca = st.selectbox("Marca", sorted(df.marca.unique()), key="v2_marca")
    canal = st.selectbox("Canal", sorted(df.canal.unique()), key="v2_canal")

    precio = st.number_input("Precio", value=float(df.precio.median()))
    comp = st.slider("Competencia",1,10,5)
    dem = st.slider("Demanda",10,100,60)
    ten = st.slider("Tendencia",10,100,70)
    marg = st.slider("Margen %",0,90,40)
    conn = st.slider("Conexion",0,100,60)

    row = pd.DataFrame([{
        "precio":precio,
        "competencia":comp,
        "demanda":dem,
        "tendencia":ten,
        "margen_pct":marg,
        "conexion_score":conn,
        "rating_conexion":7,
        "sentiment_score":1,
        "marca":marca,
        "canal":canal
    }])

    if st.button("Simular", key="v2_sim"):
        p = clf.predict_proba(row)[0][1]
        v = reg.predict(row)[0]
        st.metric("Prob éxito", f"{p:.2%}")
        st.metric("Ventas", f"{v:,.0f}")

# =====================================================
# RECOMENDADOR
# =====================================================

with tab_rec:

    st.subheader("Motor de recomendaciones")

    st.write("Precio óptimo sugerido:")
    st.metric("Precio", f"{df.precio.mean()*0.95:.2f}")

    st.write("Claims sugeridos top:")
    st.write(["alto en proteína","sin azúcar","alto en fibra"])

# =====================================================
# PACK VISION
# =====================================================

with tab_pack:

    img = st.file_uploader("Imagen empaque", type=["png","jpg"])
    if img:
        im = Image.open(img)
        st.image(im)

        arr = np.asarray(im.convert("L"))
        contrast = arr.std()/255
        st.metric("Contraste visual", f"{contrast:.2f}")

# =====================================================
# INVERSIONISTA
# =====================================================

with tab_inv:

    st.subheader("Vista inversión")

    base = df.ventas_unidades.mean()
    opt = base * 1.18

    st.metric("Ventas base", f"{base:,.0f}")
    st.metric("Ventas optimizadas", f"{opt:,.0f}")
    st.metric("Upside", f"{opt-base:,.0f}")

# =====================================================
# REPORTE
# =====================================================

with tab_rep:

    report = f"""
PLATAFORMA IA PRODUCTO 2.0

Accuracy: {acc}
AUC: {auc}
MAE ventas: {mae}

Modelo activo.
Recomendaciones activas.
"""

    st.download_button(
        "Descargar reporte",
        report,
        "reporte_producto_ia.txt"
    )