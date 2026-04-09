import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import pandas as pd
import os

MODELO_PATH = 'modelos/mejor_modelo_fase2.keras'
CLASS_NAMES = ['no', 'yes']
INPUT_SHAPE = (150, 150)

st.set_page_config(page_title="Detector de Tumores - UD05", page_icon="🧠")

st.title("Clasificador de Imágenes Médicas")
st.write("Sube una imagen de resonancia magnética para obtener un diagnóstico asistido por IA.")

@st.cache_resource
def cargar_mi_modelo():
    if not os.path.exists(MODELO_PATH):
        alt_path = 'mejor_modelo_fase2.keras'
        if os.path.exists(alt_path):
            return keras.models.load_model(alt_path)
        st.error(f"❌ No se encontró el modelo en: {MODELO_PATH}")
        return None
    return keras.models.load_model(MODELO_PATH)

modelo = cargar_mi_modelo()

archivo = st.file_uploader("Seleccionar imagen de resonancia (JPG, PNG)", type=['jpg', 'jpeg', 'png'])

if archivo is not None and modelo is not None:
    imagen = Image.open(archivo)
    
    imagen_rgb = imagen.convert('RGB')
    img_resized = imagen_rgb.resize(INPUT_SHAPE)
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner('Analizando...'):
        preds = modelo.predict(img_array, verbose=0)[0]
        
        if len(preds) == 1:
            prob_yes = float(preds[0])
            prob_no = 1.0 - prob_yes
            probabilidades = [prob_no, prob_yes]
        else:
            probabilidades = preds

        idx_predicho = np.argmax(probabilidades)
        clase_detectada = CLASS_NAMES[idx_predicho]
        confianza = probabilidades[idx_predicho]

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(imagen, caption="Imagen cargada", use_container_width=True)

    with col2:
        st.subheader("Resultado")
        if clase_detectada == 'yes':
            st.error(f"**TUMOR DETECTADO**")
        else:
            st.success(f"**SIN TUMOR**")
        
        st.metric("Confianza", f"{confianza*100:.2f}%")

    st.divider()
    st.subheader("📊 Desglose de probabilidades por clase")
    
    df_stats = pd.DataFrame({
        'Estado': ['Sin Tumor (no)', 'Tumor Detectado (yes)'],
        'Probabilidad': [p * 100 for p in probabilidades]
    })
    
    st.bar_chart(df_stats.set_index('Estado'))
    st.table(df_stats)

elif modelo is None:
    st.warning("El sistema no puede funcionar sin el archivo del modelo.")