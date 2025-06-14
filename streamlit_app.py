import streamlit as st
import numpy as np
import keras
import pickle
from PIL import Image
from keras.applications.resnet50 import preprocess_input

# ==== CONFIGURACIÓN STREAMLIT ====
st.set_page_config(page_title="Clasificador de Tumores Cerebrales", layout="centered")
st.title("🧠 Clasificador de Tumores Cerebrales con Deep Learning")
st.markdown("Sube una imagen de resonancia magnética (MRI) para predecir si está **sana** o presenta un **tumor** cerebral.")

# ==== CARGA DE MODELO Y ENCODER ====
@st.cache_resource
def load_model_and_encoder():
    model = keras.models.load_model("results/brain_tumor_phase1_streamlit.h5")
    with open("results/label_encoder_phase1.pkl", "rb") as f:
        encoder = pickle.load(f)
    return model, encoder

model, label_encoder = load_model_and_encoder()

# ==== SUBIDA DE IMAGEN ====
uploaded_file = st.file_uploader("📁 Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼 Imagen cargada", use_container_width=True)

    # ==== PREPROCESAMIENTO ====
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_array)

    # ==== PREDICCIÓN ====
    prediction = model.predict(img_preprocessed)[0][0]
    class_index = int(prediction >= 0.5)
    class_label = label_encoder.inverse_transform([class_index])[0]
    confidence = round(float(prediction) * 100 if class_index == 1 else (1 - prediction) * 100, 2)

    # ==== RESULTADO ====
    st.markdown("## 🧾 Resultado del Análisis")
    st.markdown(f"### 🔍 Predicción: **{class_label}**")
    st.markdown(f"### 📊 Confianza: **{confidence}%**")

