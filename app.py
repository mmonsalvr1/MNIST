import time
import numpy as np
import pandas as pd
import streamlit as st

from PIL import Image, ImageOps
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from streamlit_drawable_canvas import st_canvas

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="MNIST - Clasificador", layout="wide")

st.title("🧠 Clasificador de dígitos MNIST (tipo Iris)")
st.caption("Selecciona un modelo, mira métricas, y prueba con imágenes de validación o dibujando un dígito.")

# -----------------------------
# Carga MNIST (sin depender de tensorflow)
# Usamos OpenML (pero sin web aquí). Para evitar fallos, usamos un fallback:
# - Intento: keras.datasets (si el entorno lo tiene)
# - Fallback: dataset embebido vía sklearn (digits 8x8) NO sirve para MNIST
#
# RECOMENDADO: incluir tensorflow/keras si quieres 100% MNIST.
# En esta versión, usamos un loader MNIST por Keras si está disponible.
# -----------------------------
@st.cache_resource
def load_mnist():
    """
    Carga MNIST 28x28. Requiere tensorflow/keras instalado.
    Si no está, lanza error con instrucciones.
    """
    try:
        from tensorflow import keras
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        # normalizar a [0,1]
        x_train = x_train.astype(np.float32) / 255.0
        x_test = x_test.astype(np.float32) / 255.0
        return (x_train, y_train), (x_test, y_test)
    except Exception as e:
        raise RuntimeError(
            "No pude cargar MNIST. Para usar MNIST real, agrega 'tensorflow' al requirements.txt.\n"
            "Ejemplo: tensorflow==2.16.1 (o similar).\n"
            f"Detalle: {e}"
        )

def flatten(x):
    return x.reshape(x.shape[0], -1)

# -----------------------------
# Modelos (varias opciones)
# -----------------------------
def build_model(name: str):
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=200, n_jobs=None)
    if name == "SVM (RBF)":
        return SVC(kernel="rbf", gamma="scale")
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    if name == "MLP (Red neuronal)":
        return MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=25, random_state=42)
    raise ValueError("Modelo no soportado")

@st.cache_resource
def train_and_evaluate(model_name: str, train_size: int):
    (x_train, y_train), (x_test, y_test) = load_mnist()

    # Submuestreo para que sea ágil en despliegue (puedes subirlo)
    n = min(train_size, x_train.shape[0])
    idx = np.random.RandomState(42).choice(x_train.shape[0], size=n, replace=False)

    Xtr = flatten(x_train[idx])
    ytr = y_train[idx]
    Xte = flatten(x_test)
    yte = y_test

    model = build_model(model_name)

    t0 = time.time()
    model.fit(Xtr, ytr)
    train_time = time.time() - t0

    t1 = time.time()
    pred = model.predict(Xte)
    pred_time = time.time() - t1

    acc = accuracy_score(yte, pred)
    cm = confusion_matrix(yte, pred)
    report = classification_report(yte, pred, output_dict=True)

    return {
        "model": model,
        "x_test": x_test,
        "y_test": yte,
        "pred_test": pred,
        "accuracy": acc,
        "cm": cm,
        "report": report,
        "train_time": train_time,
        "pred_time": pred_time,
    }

# -----------------------------
# Sidebar (tipo Iris)
# -----------------------------
st.sidebar.header("⚙️ Configuración")
model_name = st.sidebar.selectbox(
    "Modelo de clasificación",
    ["Logistic Regression", "SVM (RBF)", "Random Forest", "MLP (Red neuronal)"]
)

train_size = st.sidebar.slider(
    "Tamaño de entrenamiento (para rapidez en la nube)",
    min_value=5000, max_value=60000, value=20000, step=5000
)

st.sidebar.info(
    "Tip: si quieres máxima precisión, sube el train_size.\n"
    "Para despliegue rápido, 20k funciona bien."
)

# -----------------------------
# Entrenar + métricas
# -----------------------------
with st.spinner("Entrenando y evaluando..."):
    out = train_and_evaluate(model_name, train_size)

colA, colB, colC, colD = st.columns(4)
colA.metric("Accuracy (test)", f"{out['accuracy']:.4f}")
colB.metric("Tiempo de entrenamiento", f"{out['train_time']:.2f}s")
colC.metric("Tiempo predicción test", f"{out['pred_time']:.2f}s")
colD.metric("Modelo", model_name)

st.subheader("📊 Métricas de desempeño")
cm = out["cm"]
fig, ax = plt.subplots()
ax.imshow(cm)
ax.set_title("Matriz de confusión (test)")
ax.set_xlabel("Predicción")
ax.set_ylabel("Real")
st.pyplot(fig)

report_df = pd.DataFrame(out["report"]).T
st.write("**Classification report**")
st.dataframe(report_df)

# -----------------------------
# Zona de pruebas: ver imagen + predicción
# -----------------------------
st.subheader("🧪 Pruebas con imagen de validación (se ve el dígito antes de clasificar)")
x_test = out["x_test"]
y_test = out["y_test"]
model = out["model"]

c1, c2 = st.columns([1, 1])

with c1:
    idx = st.number_input("Índice en test (0..9999)", min_value=0, max_value=int(x_test.shape[0]-1), value=0, step=1)
    img = x_test[int(idx)]
    st.image(img, caption=f"Etiqueta real: {y_test[int(idx)]}", width=220)

    if st.button("Clasificar esta imagen"):
        pred = model.predict(flatten(img[None, ...]))[0]
        st.success(f"✅ Predicción: **{pred}**")

with c2:
    st.write("### ✍️ Dibujar un dígito (canvas) y clasificar")
    st.caption("Dibuja en negro sobre fondo blanco. El sistema lo reescala a 28x28 y predice.")

    canvas = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=18,
        stroke_color="#000000",
        background_color="#FFFFFF",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    def preprocess_canvas(img_rgba: np.ndarray) -> np.ndarray:
        # RGBA -> PIL -> grayscale
        pil = Image.fromarray(img_rgba.astype("uint8"), mode="RGBA").convert("L")
        # Invertir para que el trazo negro se parezca a MNIST (blanco = fondo)
        pil = ImageOps.invert(pil)
        # Recortar borde y centrar un poco (simple)
        pil = pil.resize((28, 28))
        arr = np.array(pil).astype(np.float32) / 255.0
        return arr

    if canvas.image_data is not None:
        proc = preprocess_canvas(canvas.image_data)
        st.image(proc, caption="Preprocesado 28x28", width=150)

        if st.button("Clasificar dibujo"):
            pred = model.predict(flatten(proc[None, ...]))[0]
            st.success(f"🧠 Predicción del dibujo: **{pred}**")
