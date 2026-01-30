# app_v2.py
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
# ✅ Market Intelligence
# ============================================================

# ============================================================
# 🧠 PRODUCT INTELLIGENCE PLATFORM — CORE ENGINE (FIXED)
# Base + Modelos + Pack Vision + Shelf & Emotion Engine
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
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
    page_title="AI Product Intelligence Platform",
    layout="wide"
)

DATA_PATH_DEFAULT = "mercado_cereales_5000_con_ventas.csv"

# ----------------------------
# Helpers
# ----------------------------
def clip(v,a,b):
    return float(np.clip(v,a,b))

def safe_percent(x):
    return f"{x*100:.2f}%"

def df_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")

# ============================================================
# 📦 PACK VISION ENGINE
# ============================================================

def image_metrics(img):
    im = img.convert("RGB")
    arr = np.asarray(im).astype(np.float32)

    gray = (0.299*arr[...,0] + 0.587*arr[...,1] + 0.114*arr[...,2])

    brightness = float(np.mean(gray)/255)
    contrast = float(np.std(gray)/255)

    rg = arr[...,0] - arr[...,1]
    yb = 0.5*(arr[...,0]+arr[...,1]) - arr[...,2]
    colorfulness = float((np.std(rg)+0.3*np.std(yb))/255)

    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:,1:-1] = gray[:,2:] - gray[:,:-2]
    gy[1:-1,:] = gray[2:,:] - gray[:-2,:]
    mag = np.sqrt(gx**2 + gy**2)

    edge_density = float(np.mean(mag > np.percentile(mag,85)))

    return {
        "brightness": brightness,
        "contrast": contrast,
        "colorfulness": colorfulness,
        "edge_density": edge_density
    }


def pack_scores_from_metrics(m):

    legibility = clip(70*m["contrast"] + 30*(1-abs(m["edge_density"]-0.18)/0.18),0,1)*100
    pop = clip(0.55*m["contrast"] + 0.45*m["colorfulness"],0,1)*100
    clarity = clip(0.6*m["contrast"] + 0.4*(1-m["edge_density"]),0,1)*100

    return {
        "legibility": round(legibility,1),
        "pop": round(pop,1),
        "clarity": round(clarity,1)
    }

# ============================================================
# 🧲 SHELF & EMOTION ENGINE
# ============================================================

def shelf_scores(pack_scores, metrics):

    attention = clip(0.6*pack_scores["pop"] + 0.4*(metrics["contrast"]*100),0,100)
    recall = clip(0.5*attention + 0.5*pack_scores["clarity"],0,100)
    choice = clip(0.45*attention + 0.35*pack_scores["clarity"] + 0.2*pack_scores["legibility"],0,100)

    emotion_energy = clip(pack_scores["pop"]/100,0,1)
    emotion_trust = clip(pack_scores["legibility"]/100,0,1)

    emotion = emotion_energy*0.6 + emotion_trust*0.4

    return {
        "attention": attention,
        "recall": recall,
        "choice": choice,
        "emotion": emotion*100
    }

# ============================================================
# 🔥 HEATMAP PROXY
# ============================================================

def simple_heatmap(img):
    im = img.convert("L")
    arr = np.asarray(im).astype(float)

    gx = np.zeros_like(arr)
    gy = np.zeros_like(arr)

    gx[:,1:-1] = arr[:,2:] - arr[:,:-2]
    gy[1:-1,:] = arr[2:,:] - arr[:-2,:]

    mag = np.sqrt(gx**2 + gy**2)
    mag = (mag - mag.min())/(mag.max()+1e-6)

    heat = np.stack([mag, mag*0.5, mag*0], axis=-1)
    heat = (heat*255).astype(np.uint8)

    return Image.fromarray(heat)

# ============================================================
# 🧮 MNL CHOICE SIMULATION
# ============================================================

def mnl_choice(df, beta):

    U = (
        beta["att"]*df["attention"] +
        beta["rec"]*df["recall"] +
        beta["emo"]*df["emotion"] +
        beta["price"]*df["price"]
    )

    U = U - U.max()
    expU = np.exp(U)
    probs = expU/expU.sum()

    df["choice_prob"] = probs
    return df.sort_values("choice_prob", ascending=False)

# ============================================================
# 📊 DATA LOADER
# ============================================================

@st.cache_data
def load_data(path):

    df = pd.read_csv(path)

    for c in ["marca","categoria","canal"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.lower().str.strip()

    # validación mínima
    needed = ["precio","competencia","demanda","tendencia","margen_pct",
              "conexion_score","rating_conexion","sentiment_score",
              "exito","ventas_unidades"]

    missing = [c for c in needed if c not in df.columns]
    if missing:
        st.error(f"Faltan columnas en dataset: {missing}")
        st.stop()

    return df

# ============================================================
# 🤖 MODELS — FIXED
# ============================================================

@st.cache_resource
def train_models(df):

    features = [
        "precio","competencia","demanda","tendencia",
        "margen_pct","conexion_score",
        "rating_conexion","sentiment_score",
        "marca","canal"
    ]

    X = df[features].copy()
    y_cls = df["exito"].astype(int).copy()
    y_reg = df["ventas_unidades"].astype(float).copy()

    num_cols = features[:-2]
    cat_cols = ["marca","canal"]

    pre = ColumnTransformer([
        ("num","passthrough",num_cols),
        ("cat",OneHotEncoder(handle_unknown="ignore"),cat_cols)
    ])

    # ---------- Clasificación ----------
    clf = Pipeline([
        ("pre", pre),
        ("model", RandomForestClassifier(n_estimators=300, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cls, test_size=0.2, random_state=42, stratify=y_cls
    )

    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    proba = clf.predict_proba(X_test)[:,1]

    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, proba)
    cm = confusion_matrix(y_test, pred)

    # ---------- Regresión ----------
    reg = Pipeline([
        ("pre", pre),
        ("model", RandomForestRegressor(n_estimators=300, random_state=42))
    ])

    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )

    reg.fit(X_train_s, y_train_s)

    pred_sales = reg.predict(X_test_s)
    mae = mean_absolute_error(y_test_s, pred_sales)

    return clf, reg, acc, auc, cm, mae

# ============================================================
# 🚀 BOOT
# ============================================================

if Path(DATA_PATH_DEFAULT).exists():
    df = load_data(DATA_PATH_DEFAULT)
else:
    st.error("No encuentro el dataset base")
    st.stop()

success_model, sales_model, ACC, AUC, CM, MAE = train_models(df)

# ============================================================
# 📡 MARKET INTELLIGENCE LAYER
# ============================================================

@st.cache_data
def load_mi_tables():
    tables = {}
    names = [
        "market_trends.csv",
        "market_claims.csv",
        "market_reviews.csv",
        "market_prices.csv"
    ]

    for n in names:
        if Path(n).exists():
            tables[n] = pd.read_csv(n)

    return tables


def mi_category_score(mi_tables, categoria):

    score = 0

    if "market_trends.csv" in mi_tables:
        t = mi_tables["market_trends.csv"]
        s = t[t["categoria"] == categoria]["trend_index"].mean()
        if pd.notna(s):
            score += 0.4 * s

    if "market_reviews.csv" in mi_tables:
        r = mi_tables["market_reviews.csv"]
        s = r[r["categoria"] == categoria]["sentiment"].mean()
        if pd.notna(s):
            score += 30 * s

    if "market_claims.csv" in mi_tables:
        c = mi_tables["market_claims.csv"]
        g = c[c["categoria"] == categoria]["growth_pct"].mean()
        if pd.notna(g):
            score += g * 0.3

    return clip(score, 0, 100)


def apply_mi_adjustment(row, mi_score):

    row = row.copy()

    row["demanda"] = clip(row["demanda"] * (1 + mi_score/200), 0, 100)
    row["tendencia"] = clip(row["tendencia"] * (1 + mi_score/250), 0, 100)

    return row


# ============================================================
# 🏷️ CLAIMS INTELLIGENCE ENGINE
# ============================================================

BASE_CLAIMS = {
    "fit": ["alto en proteína","sin azúcar añadida","alto en fibra","integral"],
    "kids": ["con vitaminas","sabor chocolate","energía diaria"],
    "premium": ["ingredientes seleccionados","calidad premium"],
    "value": ["rinde más","mejor precio"]
}


def get_claims_for_segment(segmento, mi_tables=None):

    base = BASE_CLAIMS.get(segmento, [])

    if mi_tables and "market_claims.csv" in mi_tables:
        mc = mi_tables["market_claims.csv"]
        extra = mc.sort_values("growth_pct", ascending=False)["claim"].head(5).tolist()
        base = list(dict.fromkeys(base + extra))

    return base


def claims_score(claims):

    if not claims:
        return 0

    return clip(60 + 8*len(claims), 0, 100)


# ============================================================
# 🧊 COLD START ENGINE (PRODUCTO NUEVO)
# ============================================================

def cold_start_predict(input_row, success_model, sales_model):

    df_row = pd.DataFrame([input_row])

    prob = float(success_model.predict_proba(df_row)[0][1])
    sales = float(sales_model.predict(df_row)[0])

    return prob, sales


# ============================================================
# 🚀 WHAT-IF RECOMMENDATION ENGINE
# ============================================================

def whatif_recommendations(base_row, success_model, sales_model):

    scenarios = []

    for dp in [-0.15,-0.1,0,0.1,0.15]:
        for dm in [-10,0,10]:
            r = base_row.copy()
            r["precio"] *= (1+dp)
            r["margen_pct"] = clip(r["margen_pct"]+dm,0,90)

            p,s = cold_start_predict(r, success_model, sales_model)

            scenarios.append({
                **r,
                "prob": p,
                "sales": s
            })

    out = pd.DataFrame(scenarios)
    out["score"] = out["prob"]*0.65 + (out["sales"]/out["sales"].max())*0.35

    return out.sort_values("score", ascending=False).head(10)


# ============================================================
# 💼 INVESTOR ENGINE
# ============================================================

def investor_metrics(price, cost, volume, cpa, retention):

    margin = price - cost
    ltv = margin * retention
    net = ltv - cpa

    return {
        "unit_margin": margin,
        "ltv": ltv,
        "ltv_net": net,
        "revenue": price * volume
    }


# ============================================================
# 📄 EXECUTIVE REPORT ENGINE
# ============================================================

def build_exec_report():

    lines = []
    lines.append("AI PRODUCT INTELLIGENCE REPORT")
    lines.append("="*40)

    if "last_sim" in st.session_state:
        lines.append("\nSIMULADOR:")
        for k,v in st.session_state.last_sim.items():
            lines.append(f"{k}: {v}")

    if "last_new" in st.session_state:
        lines.append("\nPRODUCTO NUEVO:")
        for k,v in st.session_state.last_new.items():
            lines.append(f"{k}: {v}")

    return "\n".join(lines).encode("utf-8")


# ============================================================
# 🔁 SHELF LEARNING LOG
# ============================================================

if "shelf_learning" not in st.session_state:
    st.session_state.shelf_learning = []


def log_shelf_learning(row):
    st.session_state.shelf_learning.append(row)

# ============================================================
# 🖥️ UI — TABS COMPLETAS (V2.3 + MI + SHELF + COLD START)
# ============================================================

st.title("🧠 AI Product Intelligence Platform")
st.caption("Predicción de éxito + ventas + pack/claims + Shelf & Emotion (3s) + Cold Start + Investor + Market Intelligence")

k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("Registros", f"{len(df):,}")
k2.metric("Precisión", f"{ACC*100:.2f}%")
k3.metric("AUC", f"{AUC:.3f}")
k4.metric("Éxito base", f"{df['exito'].mean()*100:.1f}%")
k5.metric("MAE ventas", f"{MAE:,.0f} u.")

st.divider()

# MI tables loaded once
mi_tables = load_mi_tables()

tabs = st.tabs([
    "🧪 Simulador",
    "📊 Insights",
    "📦 Pack Vision+",
    "🧲 Shelf & Emotion (3s)",
    "🧊 Producto Nuevo + Rec",
    "💼 Inversionista",
    "📡 Market Intelligence",
    "📄 Reporte",
    "📂 Datos",
    "🧠 Diagnóstico"
])

tab_sim, tab_ins, tab_pack, tab_shelf, tab_new, tab_inv, tab_mi, tab_rep, tab_data, tab_diag = tabs

# ============================================================
# 🧪 SIMULADOR
# ============================================================
with tab_sim:
    st.subheader("🧪 Simulador (éxito + ventas + pack + claims + conexión)")

    marcas = sorted(df["marca"].unique().tolist())
    canales = sorted(df["canal"].unique().tolist())
    categorias = sorted(df["categoria"].unique().tolist())

    c1,c2,c3,c4 = st.columns(4)
    marca = c1.selectbox("Marca", marcas, 0, key="sim_marca")
    canal = c2.selectbox("Canal", canales, 0, key="sim_canal")
    categoria = c3.selectbox("Categoría", categorias, 0, key="sim_cat")
    segmento = c4.selectbox("Segmento", ["fit","kids","premium","value"], 0, key="sim_seg")

    mi_score = mi_category_score(mi_tables, str(categoria).lower()) if mi_tables else 0
    use_mi = st.toggle("Aplicar Market Intelligence (ajusta demanda/tendencia)", value=True, key="sim_use_mi")
    if mi_tables:
        st.metric("MI Score (categoría)", f"{mi_score:.1f}/100")

    st.markdown("### Variables negocio")
    b1,b2,b3,b4,b5 = st.columns(5)
    precio = b1.number_input("Precio", 1.0, 9999.0, float(df["precio"].median()), step=1.0, key="sim_precio")
    margen = b2.slider("Margen %", 0, 90, int(np.clip(df["margen_pct"].median(),0,90)), key="sim_margen")
    competencia = b3.slider("Competencia (1-10)", 1, 10, int(df["competencia"].median()), key="sim_comp")
    demanda = b4.slider("Demanda (0-100)", 0, 100, int(df["demanda"].median()), key="sim_dem")
    tendencia = b5.slider("Tendencia (0-100)", 0, 100, int(df["tendencia"].median()), key="sim_tend")

    st.markdown("### Pack + Claims")
    img = st.file_uploader("Sube empaque (opcional)", type=["png","jpg","jpeg"], key="sim_pack_upl")

    if img:
        im = Image.open(img)
        st.image(im, caption="Empaque", use_container_width=True)
        m = image_metrics(im)
        ps = pack_scores_from_metrics(m)
        sh = shelf_scores(ps, m)
        st.write("**Scores Pack**", ps)
        st.write("**Scores Shelf (3s)**", sh)
        conexion_pack = sh["choice"]
    else:
        ps = {"legibility": 60.0, "pop": 60.0, "clarity": 60.0}
        sh = {"attention": 60.0, "recall": 60.0, "choice": 60.0, "emotion": 60.0}
        conexion_pack = 60.0

    claims_list = get_claims_for_segment(segmento, mi_tables if mi_tables else None)
    claims_sel = st.multiselect("Claims (elige 2-3)", claims_list, default=claims_list[:2], key="sim_claims")
    cscore = claims_score(claims_sel)
    st.metric("Claims Score", f"{cscore:.1f}/100")

    conexion_score = clip(0.45*demanda + 0.35*conexion_pack + 0.20*cscore, 0, 100)

    row = {
        "precio": float(precio),
        "competencia": float(competencia),
        "demanda": float(demanda),
        "tendencia": float(tendencia),
        "margen_pct": float(margen),
        "conexion_score": float(conexion_score),
        "rating_conexion": 7.0,
        "sentiment_score": 1.0,
        "marca": str(marca).lower(),
        "canal": str(canal).lower()
    }

    if use_mi and mi_tables:
        row = apply_mi_adjustment(row, mi_score)

    entrada = pd.DataFrame([row])

    if st.button("🚀 Simular", key="sim_btn"):
        prob = float(success_model.predict_proba(entrada)[0][1])
        ventas = float(sales_model.predict(entrada)[0])

        st.session_state.last_sim = {
            **row,
            "prob_exito": prob,
            "ventas_unidades": ventas,
            "claims": claims_sel
        }

        r1,r2,r3 = st.columns(3)
        r1.metric("Prob. Éxito", safe_percent(prob))
        r2.metric("Ventas (u.)", f"{ventas:,.0f}")
        r3.metric("Conexión final", f"{conexion_score:.1f}/100")

        st.dataframe(entrada, use_container_width=True)

# ============================================================
# 📊 INSIGHTS  ✅ (FIX Altair SchemaValidationError)
# ============================================================
with tab_ins:
    st.subheader("📊 Insights (rankings + distribuciones)")

    left, right = st.columns(2)

    with left:
        st.markdown("**Ranking por marca (Conexión promedio)**")
        ins = df.groupby("marca")["conexion_score"].mean().sort_values(ascending=False).round(2)
        st.dataframe(ins.to_frame("conexion_score"), use_container_width=True)

        st.markdown("**Ranking por marca (Éxito %)**")
        ex = (df.groupby("marca")["exito"].mean() * 100).sort_values(ascending=False).round(1)
        st.dataframe(ex.to_frame("exito_%"), use_container_width=True)

    with right:
        st.markdown("**Ranking marca + canal (Ventas promedio)**")
        vm = df.groupby(["marca", "canal"])["ventas_unidades"].mean().sort_values(ascending=False).round(0)
        st.dataframe(vm.head(25).to_frame("ventas_unidades"), use_container_width=True)

    st.divider()
    d1, d2 = st.columns(2)

    def _bar_from_bins(bin_counts: pd.Series, title: str):
        """Evita Altair SchemaValidationError convirtiendo a DF simple."""
        st.markdown(f"**{title}**")
        dfp = bin_counts.reset_index()
        dfp = dfp.iloc[:, :2].copy()
        dfp.columns = ["bucket", "count"]
        dfp["bucket"] = dfp["bucket"].astype(str)
        st.bar_chart(dfp.set_index("bucket"), use_container_width=True)

    with d1:
        bins = pd.cut(df["conexion_score"], [0, 20, 40, 60, 80, 100], include_lowest=True)
        dist = bins.value_counts().sort_index()
        _bar_from_bins(dist, "Distribución: Conexión emocional (bucket)")

    with d2:
        bins2 = pd.cut(df["ventas_unidades"].clip(0, 40000), [0, 2000, 5000, 10000, 20000, 40000], include_lowest=True)
        dist2 = bins2.value_counts().sort_index()
        _bar_from_bins(dist2, "Distribución: Ventas unidades (bucket)")

# ============================================================
# 📦 PACK VISION+
# ============================================================
with tab_pack:
    st.subheader("📦 Pack Vision+ (pack suelto + heatmap + quick wins)")

    img = st.file_uploader("Sube imagen del empaque", type=["png","jpg","jpeg"], key="pack_upl")

    if not img:
        st.info("Sube un pack para ver scores + heatmap + quick wins.")
    else:
        im = Image.open(img)
        st.image(im, caption="Empaque", use_container_width=True)

        m = image_metrics(im)
        ps = pack_scores_from_metrics(m)
        sh = shelf_scores(ps, m)

        a1,a2,a3,a4 = st.columns(4)
        a1.metric("Legibilidad", f"{ps['legibility']}/100")
        a2.metric("Shelf Pop", f"{ps['pop']}/100")
        a3.metric("Claridad", f"{ps['clarity']}/100")
        a4.metric("Elección (3s)", f"{sh['choice']:.1f}/100")

        st.markdown("### Heatmap (proxy visual)")
        hm = simple_heatmap(im)
        st.image(hm, caption="Heatmap proxy", use_container_width=True)

        st.markdown("### Quick wins")
        wins = []
        if ps["legibility"] < 60: wins.append("Sube contraste texto/fondo y tipografía más gruesa.")
        if ps["clarity"] < 60: wins.append("Reduce ruido visual y deja aire; 2–3 claims máximo.")
        if ps["pop"] < 60: wins.append("Agrega color acento / jerarquía fuerte del beneficio principal.")
        if not wins: wins.append("Está sólido. Ajusta jerarquía: Marca → Beneficio → Variedad → Credencial.")
        for w in wins:
            st.write("•", w)

# ============================================================
# 🧲 SHELF & EMOTION (3s)
# ============================================================
with tab_shelf:
    st.subheader("🧲 Shelf & Emotion (3s)")
    st.caption("Comparación pack vs competidores + foto de anaquel con ROIs + ranking + MNL + learning log.")

    mode = st.radio("Modo", ["Pack suelto vs competidores", "Foto de anaquel + ROI recortes"], horizontal=True, key="shelf_mode")

    beta_att = st.slider("Peso Atención", 0.0, 2.0, 1.0, 0.05, key="b_att")
    beta_rec = st.slider("Peso Recordación", 0.0, 2.0, 0.8, 0.05, key="b_rec")
    beta_emo = st.slider("Peso Emoción", 0.0, 2.0, 0.7, 0.05, key="b_emo")
    beta_price = st.slider("Penalización Precio", -0.02, 0.0, -0.005, 0.0005, key="b_price")

    beta = {"att": beta_att, "rec": beta_rec, "emo": beta_emo, "price": beta_price}

    if mode == "Pack suelto vs competidores":
        st.markdown("### Sube tu pack + competidores")
        your_pack = st.file_uploader("Tu pack", type=["png","jpg","jpeg"], key="your_pack")
        comp_packs = st.file_uploader("Competidores (2–6)", type=["png","jpg","jpeg"], accept_multiple_files=True, key="comp_packs")

        if your_pack and comp_packs:
            items = []

            # tu pack
            im = Image.open(your_pack)
            m = image_metrics(im); ps = pack_scores_from_metrics(m); sh = shelf_scores(ps,m)
            items.append({"name":"TU_PACK","attention":sh["attention"],"recall":sh["recall"],"emotion":sh["emotion"],"choice":sh["choice"],"price":0})

            # comps
            for i,f in enumerate(comp_packs[:6], start=1):
                cim = Image.open(f)
                cm = image_metrics(cim); cps = pack_scores_from_metrics(cm); csh = shelf_scores(cps,cm)
                items.append({"name":f"COMP_{i}","attention":csh["attention"],"recall":csh["recall"],"emotion":csh["emotion"],"choice":csh["choice"],"price":0})

            d = pd.DataFrame(items)

            st.markdown("### Ranking (scores)")
            st.dataframe(d.sort_values("choice", ascending=False), use_container_width=True)

            st.markdown("### Simulación de elección MNL")
            # si no hay precio real, el price=0 no penaliza; puedes meterlo manual
            d["price"] = st.number_input("Precio tu pack (para MNL)", 0.0, 9999.0, 0.0, key="mnl_price_you")
            for i in range(1, len(d)):
                d.loc[i,"price"] = st.number_input(f"Precio {d.loc[i,'name']}", 0.0, 9999.0, 0.0, key=f"mnl_price_{i}")

            ranked = mnl_choice(d, beta)
            st.dataframe(ranked[["name","choice_prob","attention","recall","emotion","price"]], use_container_width=True)

            if st.button("🧠 Log resultado (ganó/perdió)", key="log_pack_mode"):
                win_name = st.selectbox("Cuál ganó en la prueba real?", ranked["name"].tolist(), key="win_sel_1")
                log_shelf_learning({
                    "ts": datetime.utcnow().isoformat(),
                    "mode": "pack",
                    "winner": win_name,
                    "beta": beta,
                    "rows": ranked.to_dict(orient="records")
                })
                st.success("Log guardado.")

        else:
            st.info("Sube tu pack y al menos 2 competidores.")

    else:
        st.markdown("### Foto de anaquel + ROI recortes (tu pack + hasta 3 competidores)")
        shelf_img = st.file_uploader("Foto de anaquel", type=["png","jpg","jpeg"], key="shelf_photo")

        if shelf_img:
            shelf_im = Image.open(shelf_img)
            st.image(shelf_im, caption="Anaquel", use_container_width=True)

            st.markdown("#### Define ROIs (x1,y1,x2,y2) en % (0-100)")
            def crop_roi(im, roi):
                W,H = im.size
                x1 = int(W*roi[0]/100); y1 = int(H*roi[1]/100)
                x2 = int(W*roi[2]/100); y2 = int(H*roi[3]/100)
                x1 = max(0,min(W-1,x1)); x2 = max(1,min(W,x2))
                y1 = max(0,min(H-1,y1)); y2 = max(1,min(H,y2))
                if x2 <= x1+2 or y2 <= y1+2:
                    return None
                return im.crop((x1,y1,x2,y2))

            cols = st.columns(4)
            rois = []
            labels = ["TU_PACK","COMP_1","COMP_2","COMP_3"]
            for i in range(4):
                with cols[i]:
                    st.markdown(f"**{labels[i]} ROI**")
                    x1 = st.number_input("x1%", 0.0, 100.0, 5.0, key=f"roi_{i}_x1")
                    y1 = st.number_input("y1%", 0.0, 100.0, 5.0, key=f"roi_{i}_y1")
                    x2 = st.number_input("x2%", 0.0, 100.0, 25.0, key=f"roi_{i}_x2")
                    y2 = st.number_input("y2%", 0.0, 100.0, 40.0, key=f"roi_{i}_y2")
                    rois.append((x1,y1,x2,y2))

            items = []
            imgs = []
            for i,roi in enumerate(rois):
                crop = crop_roi(shelf_im, roi)
                if crop is None:
                    continue
                imgs.append((labels[i], crop))
                m = image_metrics(crop); ps = pack_scores_from_metrics(m); sh = shelf_scores(ps,m)
                items.append({"name":labels[i],"attention":sh["attention"],"recall":sh["recall"],"emotion":sh["emotion"],"choice":sh["choice"],"price":0})

            if items:
                st.markdown("### ROIs recortados")
                cols2 = st.columns(min(4,len(imgs)))
                for j,(lab,imgc) in enumerate(imgs):
                    cols2[j].image(imgc, caption=lab, use_container_width=True)

                d = pd.DataFrame(items)

                st.markdown("### Ranking (scores)")
                st.dataframe(d.sort_values("choice", ascending=False), use_container_width=True)

                st.markdown("### Simulación de elección MNL")
                for i in range(len(d)):
                    d.loc[i,"price"] = st.number_input(f"Precio {d.loc[i,'name']}", 0.0, 9999.0, 0.0, key=f"shelf_price_{i}")

                ranked = mnl_choice(d, beta)
                st.dataframe(ranked[["name","choice_prob","attention","recall","emotion","price"]], use_container_width=True)

                if st.button("🧠 Log resultado (anaquel)", key="log_shelf_mode"):
                    win_name = st.selectbox("Cuál ganó en la prueba real?", ranked["name"].tolist(), key="win_sel_2")
                    log_shelf_learning({
                        "ts": datetime.utcnow().isoformat(),
                        "mode": "shelf",
                        "winner": win_name,
                        "beta": beta,
                        "rows": ranked.to_dict(orient="records")
                    })
                    st.success("Log guardado.")
            else:
                st.warning("No pude recortar ROIs válidos. Ajusta coordenadas.")

        else:
            st.info("Sube una foto de anaquel.")

    st.divider()
    st.markdown("### Learning log (descarga)")
    if st.session_state.shelf_learning:
        log_df = pd.json_normalize(st.session_state.shelf_learning)
        st.dataframe(log_df.head(50), use_container_width=True)
        st.download_button(
            "📥 Descargar learning log (CSV)",
            data=df_to_csv_bytes(log_df),
            file_name="shelf_learning_log.csv",
            mime="text/csv",
            key="dl_learning"
        )
    else:
        st.info("Aún no hay logs guardados.")

# ============================================================
# 🧊 PRODUCTO NUEVO + RECOMENDACIONES
# ============================================================
with tab_new:
    st.subheader("🧊 Producto Nuevo (Cold Start) + Recomendaciones (what-if)")

    categorias = sorted(df["categoria"].unique().tolist())
    canales = sorted(df["canal"].unique().tolist())

    c1,c2,c3 = st.columns(3)
    categoria = c1.selectbox("Categoría comparable", categorias, key="new_cat")
    canal = c2.selectbox("Canal", canales, key="new_canal")
    segmento = c3.selectbox("Segmento", ["fit","kids","premium","value"], key="new_seg")

    mi_score = mi_category_score(mi_tables, str(categoria).lower()) if mi_tables else 0
    use_mi = st.toggle("Aplicar MI en cold start", value=True, key="new_use_mi")
    if mi_tables:
        st.metric("MI Score (categoría)", f"{mi_score:.1f}/100")

    b1,b2,b3,b4,b5 = st.columns(5)
    precio = b1.number_input("Precio", 1.0, 9999.0, float(df["precio"].median()), step=1.0, key="new_precio")
    margen = b2.slider("Margen %", 0, 90, int(np.clip(df["margen_pct"].median(),0,90)), key="new_margen")
    competencia = b3.slider("Competencia (1-10)", 1, 10, int(df["competencia"].median()), key="new_comp")
    demanda = b4.slider("Demanda (0-100)", 0, 100, int(df["demanda"].median()), key="new_dem")
    tendencia = b5.slider("Tendencia (0-100)", 0, 100, int(df["tendencia"].median()), key="new_tend")

    st.markdown("### Claims + Pack (opcional)")
    claims_list = get_claims_for_segment(segmento, mi_tables if mi_tables else None)
    claims_sel = st.multiselect("Claims (2-3)", claims_list, default=claims_list[:2], key="new_claims")
    cscore = claims_score(claims_sel)

    img = st.file_uploader("Sube empaque (opcional)", type=["png","jpg","jpeg"], key="new_pack")

    if img:
        im = Image.open(img)
        st.image(im, caption="Empaque nuevo", use_container_width=True)
        m = image_metrics(im)
        ps = pack_scores_from_metrics(m)
        sh = shelf_scores(ps, m)
        pack_choice = sh["choice"]
        pack_emotion = sh["emotion"]
    else:
        pack_choice = 60.0
        pack_emotion = 60.0

    conexion_score = clip(0.45*demanda + 0.35*pack_choice + 0.20*cscore, 0, 100)

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
        "canal": str(canal).lower()
    }

    if use_mi and mi_tables:
        base_row = apply_mi_adjustment(base_row, mi_score)

    prob, ventas = cold_start_predict(base_row, success_model, sales_model)

    comp = df[df["categoria"]==str(categoria).lower()].copy()
    if comp.empty:
        comp = df.copy()

    comp["dist"] = (
        (comp["precio"] - base_row["precio"]).abs() / max(base_row["precio"],1e-6) +
        (comp["margen_pct"] - base_row["margen_pct"]).abs()/100 +
        (comp["demanda"] - base_row["demanda"]).abs()/100 +
        (comp["tendencia"] - base_row["tendencia"]).abs()/100
    )
    top = comp.sort_values("dist").head(20)

    p25 = float(np.percentile(top["ventas_unidades"], 25))
    p50 = float(np.percentile(top["ventas_unidades"], 50))
    p75 = float(np.percentile(top["ventas_unidades"], 75))

    launch_score = (
        0.45*(prob*100) +
        0.25*pack_choice +
        0.15*cscore +
        0.15*pack_emotion
    )

    st.markdown("## 🎯 Resultado (Producto Nuevo)")
    o1,o2,o3,o4 = st.columns(4)
    o1.metric("Prob. éxito", f"{prob*100:.1f}%")
    o2.metric("Ventas (punto)", f"{ventas:,.0f} u.")
    o3.metric("Rango comparables (p25–p75)", f"{p25:,.0f} — {p75:,.0f} u.")
    o4.metric("Launch Score", f"{launch_score:.1f}/100")

    st.session_state.last_new = {
        **base_row,
        "categoria": categoria,
        "segmento": segmento,
        "claims": claims_sel,
        "claims_score": float(cscore),
        "pack_choice": float(pack_choice),
        "pack_emotion": float(pack_emotion),
        "prob_exito": float(prob),
        "ventas_point": float(ventas),
        "ventas_p25": p25,
        "ventas_p50": p50,
        "ventas_p75": p75,
        "launch_score": float(launch_score),
        "mi_score": float(mi_score) if mi_tables else 0.0
    }

    st.markdown("### 🔍 Top comparables (20)")
    st.dataframe(top[["marca","precio","margen_pct","demanda","tendencia","ventas_unidades","exito"]], use_container_width=True)

    st.divider()
    st.markdown("## 🧠 Recomendaciones para subir probabilidad (what-if)")

    if st.button("🚀 Generar recomendaciones (what-if)", key="new_recos_btn"):
        out = whatif_recommendations(base_row, success_model, sales_model)

        best = out.iloc[0]
        st.metric("Mejor Prob. Éxito", safe_percent(best["prob"]))
        st.metric("Mejor Ventas", f"{best['sales']:,.0f} u.")

        st.dataframe(
            out[["prob","sales","precio","margen_pct","demanda","tendencia","conexion_score"]],
            use_container_width=True
        )

# ============================================================
# 💼 INVERSIONISTA
# ============================================================
with tab_inv:
    st.subheader("💼 Vista Inversionista (TAM + escenarios + unit economics + launch score)")

    st.markdown("### Inputs (ajustables)")
    a1,a2,a3,a4,a5 = st.columns(5)
    tam = a1.number_input("TAM ($)", 0.0, 1e12, 5e9, step=1e8, key="inv_tam")
    sam = a2.number_input("SAM ($)", 0.0, 1e12, 1e9, step=1e8, key="inv_sam")
    som = a3.number_input("SOM ($)", 0.0, 1e12, 1e8, step=1e7, key="inv_som")
    cpa = a4.number_input("CPA ($)", 0.0, 1e6, 25.0, step=1.0, key="inv_cpa")
    retention = a5.number_input("Retención (compras)", 1.0, 50.0, 6.0, step=1.0, key="inv_ret")

    st.markdown("### Supuestos unit economics")
    b1,b2,b3 = st.columns(3)
    price = b1.number_input("Precio unitario ($)", 0.0, 1e6, 40.0, step=1.0, key="inv_price")
    cost = b2.number_input("Costo unitario ($)", 0.0, 1e6, 20.0, step=1.0, key="inv_cost")
    volume = b3.number_input("Volumen anual (u.)", 0.0, 1e9, 500000.0, step=10000.0, key="inv_vol")

    met = investor_metrics(price, cost, volume, cpa, retention)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Margen unitario", f"${met['unit_margin']:.2f}")
    c2.metric("LTV", f"${met['ltv']:.2f}")
    c3.metric("LTV neto (LTV-CPA)", f"${met['ltv_net']:.2f}")
    c4.metric("Revenue anual", f"${met['revenue']:,.0f}")

    if "last_new" in st.session_state:
        st.markdown("### Launch Score (último producto nuevo)")
        st.metric("Launch Score", f"{st.session_state.last_new.get('launch_score',0):.1f}/100")
        if st.session_state.last_new.get("mi_score",0) > 0:
            st.metric("Market Tailwind (MI)", f"{st.session_state.last_new.get('mi_score',0):.1f}/100")

# ============================================================
# 📡 MARKET INTELLIGENCE
# ============================================================
with tab_mi:
    st.subheader("📡 Market Intelligence (tendencias + claims + reviews + precios)")

    if not mi_tables:
        st.info("No encontré market_*.csv en el repo. Puedes agregar: market_trends, market_claims, market_reviews, market_prices.")
    else:
        st.write("Tablas detectadas:", list(mi_tables.keys()))

        categorias = sorted(df["categoria"].unique().tolist())
        cat = st.selectbox("Categoría", categorias, key="mi_cat")
        mi_s = mi_category_score(mi_tables, str(cat).lower())
        st.metric("MI Category Score", f"{mi_s:.1f}/100")

        if "market_claims.csv" in mi_tables:
            mc = mi_tables["market_claims.csv"].copy()
            mc["categoria"] = mc["categoria"].astype(str).str.lower().str.strip()
            show = mc[mc["categoria"]==str(cat).lower()].sort_values("growth_pct", ascending=False).head(15)
            st.markdown("### Claims en crecimiento")
            st.dataframe(show, use_container_width=True)

        if "market_reviews.csv" in mi_tables:
            mr = mi_tables["market_reviews.csv"].copy()
            mr["categoria"] = mr["categoria"].astype(str).str.lower().str.strip()
            show = mr[mr["categoria"]==str(cat).lower()].head(30)
            st.markdown("### Reviews (muestra)")
            st.dataframe(show, use_container_width=True)

        if "market_prices.csv" in mi_tables:
            mp = mi_tables["market_prices.csv"].copy()
            if "categoria" in mp.columns:
                mp["categoria"] = mp["categoria"].astype(str).str.lower().str.strip()
                show = mp[mp["categoria"]==str(cat).lower()].head(50)
                st.markdown("### Precios competencia (muestra)")
                st.dataframe(show, use_container_width=True)

# ============================================================
# 📄 REPORTE EJECUTIVO
# ============================================================
with tab_rep:
    st.subheader("📄 Reporte Ejecutivo (TXT + CSV inputs)")

    rep = build_exec_report()
    st.download_button(
        "📥 Descargar Reporte (TXT)",
        data=rep,
        file_name="reporte_product_intelligence.txt",
        mime="text/plain",
        key="dl_rep_txt"
    )

    # CSV inputs
    rows = []
    if "last_sim" in st.session_state:
        rows.append({"type":"simulador", **st.session_state.last_sim})
    if "last_new" in st.session_state:
        rows.append({"type":"producto_nuevo", **st.session_state.last_new})

    if rows:
        outdf = pd.DataFrame(rows)
        st.download_button(
            "📥 Descargar Inputs (CSV)",
            data=df_to_csv_bytes(outdf),
            file_name="reporte_inputs.csv",
            mime="text/csv",
            key="dl_rep_csv"
        )
        st.dataframe(outdf, use_container_width=True)
    else:
        st.info("Aún no hay simulaciones guardadas.")

# ============================================================
# 📂 DATOS
# ============================================================
with tab_data:
    st.subheader("📂 Datos (download CSV)")

    st.download_button(
        "📥 Descargar dataset",
        data=df_to_csv_bytes(df),
        file_name="dataset_con_ventas.csv",
        mime="text/csv",
        key="dl_data"
    )

    st.dataframe(df.head(300), use_container_width=True)

# ============================================================
# 🧠 DIAGNÓSTICO
# ============================================================
with tab_diag:
    st.subheader("🧠 Diagnóstico de modelo")

    cm_df = pd.DataFrame(CM, index=["Real 0","Real 1"], columns=["Pred 0","Pred 1"])
    st.dataframe(cm_df, use_container_width=True)
    st.write(f"MAE ventas: **{MAE:,.0f}** unidades.")

