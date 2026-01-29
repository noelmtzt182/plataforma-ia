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

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Plataforma IA Producto", layout="wide")
DATA_PATH_DEFAULT = "mercado_cereales_5000_con_ventas.csv"

# ---------------- HELPERS ----------------
def clean_str(s):
    return s.astype(str).str.strip().str.lower()

def clip(v,a,b):
    return float(max(a,min(b,v)))

def bar_plot(vc, title):
    dfp = vc.reset_index()
    dfp = dfp.iloc[:, :2]
    dfp.columns = ["bucket","count"]
    dfp["bucket"] = dfp["bucket"].astype(str)
    dfp["count"] = pd.to_numeric(dfp["count"], errors="coerce").fillna(0)

    st.markdown(f"**{title}**")
    fig, ax = plt.subplots()
    ax.bar(dfp["bucket"], dfp["count"])
    ax.tick_params(axis="x", rotation=30)
    st.pyplot(fig, use_container_width=True)

# ---------------- DATA ----------------
@st.cache_data
def load_data(path):
    df = pd.read_csv(path).copy()
    for c in ["marca","canal","categoria"]:
        if c in df.columns:
            df[c] = clean_str(df[c])
    return df

# ---------------- MODELOS ----------------
@st.cache_resource
def train_models(df):

    feats = ["precio","competencia","demanda","tendencia",
             "margen_pct","conexion_score",
             "rating_conexion","sentiment_score",
             "marca","canal"]

    X = df[feats]
    y = df["exito"].astype(int)

    num = feats[:-2]
    cat = ["marca","canal"]

    pre = ColumnTransformer([
        ("num","passthrough",num),
        ("cat",OneHotEncoder(handle_unknown="ignore"),cat)
    ])

    clf = Pipeline([
        ("pre",pre),
        ("rf",RandomForestClassifier(n_estimators=300,random_state=42))
    ])

    reg = Pipeline([
        ("pre",pre),
        ("rf",RandomForestRegressor(n_estimators=300,random_state=42))
    ])

    Xtr,Xte,Ytr,Yte = train_test_split(X,y,test_size=0.2,random_state=42)
    clf.fit(Xtr,Ytr)

    reg.fit(Xtr, df.loc[Xtr.index,"ventas_unidades"])

    pred = clf.predict(Xte)
    proba = clf.predict_proba(Xte)[:,1]

    acc = accuracy_score(Yte,pred)
    auc = roc_auc_score(Yte,proba)
    cm = confusion_matrix(Yte,pred)

    mae = mean_absolute_error(
        df.loc[Xte.index,"ventas_unidades"],
        reg.predict(Xte)
    )

    return clf,reg,acc,auc,cm,mae

# ---------------- LOAD ----------------
st.sidebar.title("Datos")

up = st.sidebar.file_uploader("CSV", type=["csv"])

if up:
    df = load_data(up)
elif Path(DATA_PATH_DEFAULT).exists():
    df = load_data(DATA_PATH_DEFAULT)
else:
    st.error("Sube CSV")
    st.stop()

clf,reg,acc,auc,cm,mae = train_models(df)

# ---------------- HEADER ----------------
st.title("Plataforma IA Producto")
c1,c2,c3 = st.columns(3)
c1.metric("Registros", len(df))
c2.metric("Accuracy", f"{acc:.2f}")
c3.metric("MAE ventas", f"{mae:,.0f}")

# ---------------- TABS ----------------
tab_sim, tab_ins, tab_data = st.tabs(["Simulador","Insights","Datos"])

# ================= SIM =================
with tab_sim:

    marcas = sorted(df.marca.unique())
    canales = sorted(df.canal.unique())

    marca = st.selectbox("Marca", marcas, key="sim_marca")
    canal = st.selectbox("Canal", canales, key="sim_canal")

    precio = st.number_input("Precio", value=float(df.precio.median()))
    comp = st.slider("Competencia",1,10,5)
    dem = st.slider("Demanda",10,100,60)
    ten = st.slider("Tendencia",10,100,70)
    marg = st.slider("Margen %",0,90,40)
    conn = st.slider("Conexion",0,100,60)

    entrada = pd.DataFrame([{
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

    if st.button("Simular", key="sim_btn"):
        p = clf.predict_proba(entrada)[0][1]
        ventas = reg.predict(entrada)[0]

        st.metric("Prob éxito", f"{p:.2%}")
        st.metric("Ventas estimadas", f"{ventas:,.0f}")

# ================= INSIGHTS =================
with tab_ins:

    st.subheader("Rank marcas conexión")
    t1 = df.groupby("marca")[["conexion_score"]].mean().sort_values("conexion_score",ascending=False)
    st.dataframe(t1.reset_index(), use_container_width=True)

    st.subheader("Rank marcas ventas")
    t2 = df.groupby("marca")[["ventas_unidades"]].mean().sort_values("ventas_unidades",ascending=False)
    st.dataframe(t2.reset_index(), use_container_width=True)

    st.divider()

    d1,d2 = st.columns(2)

    with d1:
        bins = pd.cut(df.conexion_score,[0,20,40,60,80,100])
        bar_plot(bins.value_counts().sort_index(),"Distribución conexión")

    with d2:
        bins = pd.cut(df.ventas_unidades,[0,2000,5000,10000,20000,40000])
        bar_plot(bins.value_counts().sort_index(),"Distribución ventas")

# ================= DATA =================
with tab_data:
    st.dataframe(df.head(200), use_container_width=True)
    st.download_button(
        "Descargar CSV",
        df.to_csv(index=False),
        "data.csv",
        key="dl"
    )

# ================= MODEL =================
st.subheader("Matriz confusión")
st.dataframe(pd.DataFrame(cm,
    index=["Real 0","Real 1"],
    columns=["Pred 0","Pred 1"]))