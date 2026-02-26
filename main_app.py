import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px

from streamlit_drawable_canvas import st_canvas

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf
from tensorflow.keras import layers, models

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="MNIST Digit Classifier", layout="wide")

st.title("✍️ Clasificador de Dígitos (MNIST) — Streamlit")
st.caption("Modelos: Logistic Regression, Random Forest, CNN (Keras) + canvas para dibujar y validar.")

# -----------------------------
# Data
# -----------------------------
@st.cache_data(show_spinner=False)
def load_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Normalizar 0..1 para CNN
    x_train_n = x_train.astype("float32") / 255.0
    x_test_n  = x_test.astype("float32") / 255.0

    # Aplanar para modelos sklearn
    x_train_flat = x_train.reshape(-1, 28 * 28).astype("float32") / 255.0
    x_test_flat  = x_test.reshape(-1, 28 * 28).astype("float32") / 255.0

    return (x_train, y_train, x_test, y_test, x_train_n, x_test_n, x_train_flat, x_test_flat)

x_train, y_train, x_test, y_test, x_train_n, x_test_n, x_train_flat, x_test_flat = load_mnist()

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("⚙️ Configuración")

model_name = st.sidebar.selectbox(
    "Modelo",
    ["Logistic Regression", "Random Forest", "CNN (Keras)"]
)

use_subset = st.sidebar.checkbox("Modo rápido (entrenar con subconjunto)", value=True)
subset_size = st.sidebar.slider("Tamaño subconjunto (si modo rápido)", 2000, 60000, 12000, 1000)

seed = st.sidebar.number_input("Seed", 0, 9999, 42, 1)

st.sidebar.subheader("🔧 Hiperparámetros")

params = {"seed": int(seed)}
if model_name == "Logistic Regression":
    params["C"] = st.sidebar.slider("C", 0.1, 5.0, 1.0, 0.1)
    params["max_iter"] = st.sidebar.slider("max_iter", 100, 2000, 400, 50)

elif model_name == "Random Forest":
    params["n_estimators"] = st.sidebar.slider("n_estimators", 50, 500, 200, 10)
    params["max_depth"] = st.sidebar.selectbox("max_depth", [None, 10, 15, 20, 30])
    params["min_samples_split"] = st.sidebar.slider("min_samples_split", 2, 10, 2, 1)

elif model_name == "CNN (Keras)":
    params["epochs"] = st.sidebar.slider("epochs", 1, 8, 2, 1)
    params["batch_size"] = st.sidebar.selectbox("batch_size", [32, 64, 128])
    params["filters"] = st.sidebar.selectbox("conv filters", [16, 32, 48])

# -----------------------------
# Training helpers
# -----------------------------
def get_train_data_subset():
    rng = np.random.default_rng(params["seed"])
    n = x_train.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    if use_subset:
        idx = idx[:subset_size]
    return idx

def build_cnn(filters=32):
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(filters, (3,3), activation="relu"),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(filters*2, (3,3), activation="relu"),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

@st.cache_resource(show_spinner=False)
def train_model(model_name: str, use_subset: bool, subset_size: int, params: dict):
    idx = get_train_data_subset()

    if model_name == "Logistic Regression":
        clf = LogisticRegression(
            C=float(params["C"]),
            max_iter=int(params["max_iter"]),
            solver="saga",
            n_jobs=-1,
            random_state=int(params["seed"]),
        )
        clf.fit(x_train_flat[idx], y_train[idx])
        return ("sklearn", clf)

    if model_name == "Random Forest":
        clf = RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=params["max_depth"],
            min_samples_split=int(params["min_samples_split"]),
            random_state=int(params["seed"]),
            n_jobs=-1,
        )
        clf.fit(x_train_flat[idx], y_train[idx])
        return ("sklearn", clf)

    if model_name == "CNN (Keras)":
        model = build_cnn(filters=int(params["filters"]))
        x_tr = x_train_n[idx][..., np.newaxis]
        y_tr = y_train[idx]
        # usar un pequeño validation split interno
        model.fit(
            x_tr, y_tr,
            epochs=int(params["epochs"]),
            batch_size=int(params["batch_size"]),
            validation_split=0.1,
            verbose=0
        )
        return ("keras", model)

    raise ValueError("Modelo no soportado")

# -----------------------------
# Train + evaluate
# -----------------------------
with st.spinner("Entrenando modelo (la primera vez puede tardar)..."):
    model_type, model = train_model(model_name, use_subset, subset_size, params)

# predicciones test
if model_type == "sklearn":
    y_pred = model.predict(x_test_flat)
    proba = model.predict_proba(x_test_flat) if hasattr(model, "predict_proba") else None
else:
    x_te = x_test_n[..., np.newaxis]
    proba = model.predict(x_te, verbose=0)
    y_pred = np.argmax(proba, axis=1)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# -----------------------------
# Layout: métricas + gráficas
# -----------------------------
left, right = st.columns([1.1, 0.9], gap="large")

with left:
    st.subheader("📊 Métricas de desempeño (Test)")
    st.metric("Accuracy", f"{acc:.4f}")

    fig = px.imshow(
        cm,
        text_auto=True,
        title="Matriz de confusión",
        labels=dict(x="Predicción", y="Real", color="Conteo"),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📄 Classification report (detalle)", expanded=False):
        st.text(classification_report(y_test, y_pred))

with right:
    st.subheader("🖼️ Foto de validación + predicción")
    if "val_idx" not in st.session_state:
        st.session_state.val_idx = int(np.random.randint(0, len(x_test)))

    colA, colB = st.columns([0.55, 0.45])
    with colA:
        if st.button("🔀 Cambiar imagen de validación"):
            st.session_state.val_idx = int(np.random.randint(0, len(x_test)))

        idx = st.session_state.val_idx
        img = x_test[idx]
        true_label = int(y_test[idx])

        fig2, ax = plt.subplots(figsize=(3.2, 3.2))
        ax.imshow(img, cmap="gray")
        ax.set_title(f"Etiqueta real: {true_label}")
        ax.axis("off")
        st.pyplot(fig2, clear_figure=True)

    with colB:
        pred_label = int(y_pred[st.session_state.val_idx])
        st.success(f"✅ Predicción: **{pred_label}**")

        if proba is not None:
            # sklearn: proba is Nx10
            p = proba[st.session_state.val_idx]
            dfp = pd.DataFrame({"dígito": list(range(10)), "probabilidad": p}).sort_values("probabilidad", ascending=False)
            st.plotly_chart(px.bar(dfp, x="dígito", y="probabilidad", title="Probabilidades"), use_container_width=True)
        elif model_type == "keras":
            p = proba[st.session_state.val_idx]
            dfp = pd.DataFrame({"dígito": list(range(10)), "probabilidad": p}).sort_values("probabilidad", ascending=False)
            st.plotly_chart(px.bar(dfp, x="dígito", y="probabilidad", title="Probabilidades"), use_container_width=True)

# -----------------------------
# User testing: canvas drawing
# -----------------------------
st.divider()
st.subheader("🧪 Pruebas del usuario: dibuja un dígito y clasifícalo")

c1, c2 = st.columns([0.9, 1.1], gap="large")

with c1:
    st.markdown("Dibuja en el recuadro (fondo negro, trazo blanco).")
    canvas = st_canvas(
        fill_color="rgba(255, 255, 255, 0.0)",
        stroke_width=18,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="mnist_canvas",
    )

with c2:
    st.markdown("Vista previa 28×28 (preprocesada) + predicción:")

    if canvas.image_data is None:
        st.info("Dibuja un número para ver la predicción.")
    else:
        # image_data: RGBA 280x280
        img_rgba = canvas.image_data.astype(np.uint8)
        # Convertir a escala de grises (usar canal R, dado que es blanco sobre negro)
        img_gray = img_rgba[..., 0].astype(np.float32) / 255.0  # 0..1

        # Downsample a 28x28
        img_28 = tf.image.resize(img_gray[..., np.newaxis], (28, 28), method="area").numpy().squeeze()

        # MNIST es dígito claro sobre fondo oscuro? En dataset original es fondo negro y trazo blanco.
        # Aquí ya es igual (fondo 0, trazo ~1), perfecto. Solo normalizamos y aplanamos.
        fig3, ax3 = plt.subplots(figsize=(3.2, 3.2))
        ax3.imshow(img_28, cmap="gray")
        ax3.set_title("Input 28×28")
        ax3.axis("off")
        st.pyplot(fig3, clear_figure=True)

        if st.button("🔮 Predecir dibujo", type="primary"):
            if model_type == "sklearn":
                x_in = img_28.reshape(1, -1).astype("float32")
                pred = int(model.predict(x_in)[0])
                st.success(f"✅ Predicción del dibujo: **{pred}**")
                if hasattr(model, "predict_proba"):
                    p = model.predict_proba(x_in)[0]
                    dfp = pd.DataFrame({"dígito": list(range(10)), "probabilidad": p}).sort_values("probabilidad", ascending=False)
                    st.plotly_chart(px.bar(dfp, x="dígito", y="probabilidad", title="Probabilidades"), use_container_width=True)
            else:
                x_in = img_28.reshape(1, 28, 28, 1).astype("float32")
                p = model.predict(x_in, verbose=0)[0]
                pred = int(np.argmax(p))
                st.success(f"✅ Predicción del dibujo: **{pred}**")
                dfp = pd.DataFrame({"dígito": list(range(10)), "probabilidad": p}).sort_values("probabilidad", ascending=False)
                st.plotly_chart(px.bar(dfp, x="dígito", y="probabilidad", title="Probabilidades"), use_container_width=True)

st.caption("Tip: si el CNN tarda, usa 'Modo rápido' o Logistic Regression.")
