import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import os

MODELO_PATH = 'modelos/mejor_modelo_fase2.keras' 
CLASS_NAMES = ['no', 'yes']              
INPUT_SHAPE = (150, 150)                 

def cargar_modelo(ruta_modelo: str) -> keras.Model:
    print(f"Cargando modelos desde: {ruta_modelo}")
    return keras.models.load_model(ruta_modelo)

def preprocesar_imagen(ruta_imagen: str, input_shape: tuple) -> np.ndarray:
    img = tf.keras.utils.load_img(ruta_imagen, target_size=input_shape)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = img_array / 255.0   
    img_array = np.expand_dims(img_array, axis=0)   
    return img_array

def predecir(modelo: keras.Model, img_array: np.ndarray) -> tuple:
    preds = modelo.predict(img_array, verbose=0)[0]
    if len(preds) == 1:
        prob_yes = float(preds[0])
        prob_no = 1.0 - prob_yes
        probabilidades = np.array([prob_no, prob_yes])
    else:
        probabilidades = preds
    idx_predicho = np.argmax(probabilidades)
    clase_predicha = CLASS_NAMES[idx_predicho]
    confianza = probabilidades[idx_predicho]
    return clase_predicha, confianza, probabilidades

def visualizar_prediccion(ruta_imagen: str, clase_predicha: str,
                         confianza: float, probabilidades: np.ndarray) -> None:
    
    carpeta_salida = "Resultados_tumor"
    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    img = plt.imread(ruta_imagen)
    axes[0].imshow(img)
    color = 'green' if confianza > 0.7 else 'orange' if confianza > 0.4 else 'red'
    
    nombre_archivo = os.path.basename(ruta_imagen)
    axes[0].set_title(
        f'IMAGEN: {nombre_archivo}\nPREDICCIÓN: {clase_predicha.upper()}\nConfianza: {confianza*100:.1f}%',
        color=color, fontsize=12, fontweight='bold'
    )
    axes[0].axis('off')

    y_pos = np.arange(len(CLASS_NAMES))
    barras = axes[1].barh(y_pos, probabilidades * 100, color='steelblue', alpha=0.8)
    idx_ganador = np.argmax(probabilidades)
    barras[idx_ganador].set_color('green')
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(CLASS_NAMES)
    axes[1].set_xlabel('Confianza (%)')
    axes[1].set_title('Análisis de Probabilidad')
    axes[1].set_xlim(0, 100)

    for i, prob in enumerate(probabilidades):
        axes[1].text(prob * 100 + 0.5, i, f'{prob*100:.1f}%', va='center')

    plt.tight_layout()
    
    nombre_salida = os.path.join(carpeta_salida, f"resultado_{nombre_archivo}.png")
    plt.savefig(nombre_salida)
    print(f" Gráfica guardada en: {nombre_salida}")
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imagen', type=str, default=None, help="Ruta de imagen específica")
    parser.add_argument('--modelo', type=str, default=MODELO_PATH)
    args = parser.parse_args()

    modelo = cargar_modelo(args.modelo)

    if args.imagen:
        if not Path(args.imagen).exists():
            print(f" No encuentro la imagen: {args.imagen}")
            return
        rutas_a_procesar = [args.imagen]
    else:
        print("\n--- Ejecutando demostración automática de carpeta 'demo_images' ---")
        rutas_a_procesar = [
            'demo_images/tumorN.jpg',
            'demo_images/tumorn2.jpg',
            'demo_images/tumorN5.jpg',
            'demo_images/tumorS3.jpg',
            'demo_images/tumorS4.jpg'
        ]

    for ruta in rutas_a_procesar:
        if Path(ruta).exists():
            print(f"\nProcesando: {ruta}")
            img_ready = preprocesar_imagen(ruta, INPUT_SHAPE)
            clase, conf, probs = predecir(modelo, img_ready)
            print(f"Resultado: {clase.upper()} ({conf*100:.2f}%)")
            visualizar_prediccion(ruta, clase, conf, probs)
        else:
            print(f" El archivo {ruta} no existe. Saltando...")

if __name__ == '__main__':
    main()