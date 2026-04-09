# 🧠 Detector de Tumores Cerebrales con Deep Learning

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-3.x-red?style=flat-square&logo=keras)
![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b?style=flat-square&logo=streamlit)
![Accuracy](https://img.shields.io/badge/Accuracy-98%25-brightgreen?style=flat-square)
![Status](https://img.shields.io/badge/Estado-Completado-green?style=flat-square)

> Sistema de clasificación de tumores cerebrales en imágenes de resonancia magnética (MRI) mediante redes neuronales convolucionales (CNN), con experimentación sistemática de arquitecturas y aplicación web de inferencia en tiempo real.

---

## 📋 Descripción

Este proyecto desarrolla un **detector de tumores cerebrales** entrenado sobre imágenes de resonancia magnética del dataset público `brain-tumor-detection-mri` (Kaggle). El sistema implementa el ciclo completo de Machine Learning: análisis exploratorio, preprocesamiento, diseño y comparación de múltiples arquitecturas CNN, evaluación profunda de errores y despliegue en una aplicación web funcional con Streamlit.

El proyecto está estructurado en **3 fases progresivas** más un bonus de despliegue:

| Fase | Contenido | Accuracy obtenido |
|---|---|---|
| Fase 1 | CNN base, sanity checks, evaluación | 82% |
| Fase 2 | Experimentación con 3 arquitecturas + regularización | **98%** |
| Fase 3 | Optimización avanzada | En desarrollo |
| Bonus | App web Streamlit de inferencia | ✅ |

---

## 🎯 Motivación

El diagnóstico temprano de tumores cerebrales es crítico para la supervivencia del paciente. La visión artificial permite identificar patrones sutiles en resonancias magnéticas que apoyan y agilizan la precisión del diagnóstico médico, reduciendo la dependencia de la revisión manual exhaustiva.

---

## 📊 Dataset

| Parámetro | Valor |
|---|---|
| **Fuente** | Kaggle — `brain-tumor-detection-mri` |
| **Total imágenes** | 3.000 |
| **Clase "yes" (con tumor)** | 1.500 |
| **Clase "no" (sin tumor)** | 1.500 |
| **Balanceo** | 50% / 50% ✅ |
| **Baseline accuracy** | 50% |
| **Split entrenamiento/validación** | 80% / 20% (2.400 / 600) |
| **Tamaño de entrada** | 150×150 px, RGB |

---

## 🧠 Pipeline Técnico

```
Dataset MRI (Kaggle)
       │
       ▼
┌──────────────────────┐
│  Análisis Exploratorio│  → Distribución de clases, estadísticas
│                       │    de píxeles, detección de corruptas
└──────────────────────┘
       │
       ▼
┌──────────────────────┐
│   Preprocesamiento    │  → Normalización [0,1], resize 150x150
│                       │    Reshape para Conv2D, Data Augmentation
└──────────────────────┘
       │
       ▼
┌──────────────────────┐
│   Sanity Checks       │  → Verificación loss inicial (~0.69)
│                       │    Overfit en batch pequeño (100% acc)
└──────────────────────┘
       │
       ▼
┌──────────────────────┐
│  Diseño y Comparación │  → 3 arquitecturas CNN distintas
│  de Arquitecturas     │    Regularización: Dropout, BN, L2
└──────────────────────┘
       │
       ▼
┌──────────────────────┐
│  Evaluación y Análisis│  → Matriz de confusión, F1, Recall
│  de Errores           │    Análisis de falsos negativos
└──────────────────────┘
       │
       ▼
┌──────────────────────┐
│  App Streamlit        │  → Inferencia en tiempo real
│  (Bonus)              │    Confianza + desglose por clase
└──────────────────────┘
```

---

## 🏗️ Arquitecturas CNN Comparadas

Se diseñaron y entrenaron **3 arquitecturas distintas** para experimentación sistemática:

| Arquitectura | Bloques | Filtros | Kernel | Pooling | BatchNorm | Capas Densas | Val Acc |
|---|---|---|---|---|---|---|---|
| `CNN_2B_Ligera_K5` | 2 | 16→32 | **5×5** | Max | No | 1 | **~98%** ✅ |
| `CNN_3B_Estandar_BN` | 3 | 32→128 | 3×3 | Max | **Sí** | 2 | ~40% |
| `CNN_4B_Pesada_GAP` | 4 | 64→512 | 3×3 | **GAP** | No | 1 | ~80% |

**Conclusión:** La arquitectura ligera con kernel 5×5 obtuvo los mejores resultados. El campo de visión más amplio del kernel captura la masa tumoral de forma íntegra, y la baja profundidad evita la dispersión del gradiente.

---

## 🛡️ Técnicas de Regularización Aplicadas

| Técnica | Configuración | Efecto observado |
|---|---|---|
| **Dropout** | Tasas 0.2, 0.35, 0.5 comparadas | Reducción de overfitting |
| **BatchNormalization** | Con/sin comparado | Convergencia más estable |
| **L2 Weight Decay** | Penalización en Conv2D y Dense | Pesos más distribuidos |
| **Data Augmentation** | 5 transformaciones justificadas | Mejor generalización |

### Data Augmentation — Transformaciones y Justificación Médica

| Transformación | Justificación clínica |
|---|---|
| Rotación ±20° | El paciente puede inclinar ligeramente la cabeza en el escáner |
| Desplazamiento H/V 15% | El cerebro no siempre queda centrado en el sensor MRI |
| Zoom 15% | Adapta el modelo a diferentes tamaños de tumor |
| Cizallamiento 10% | Corrige distorsiones de perspectiva del escáner |
| Variación de brillo 0.8–1.2 | Simula diferencias en la intensidad de la resonancia |

---

## 📈 Resultados y Evaluación

### Métricas del mejor modelo (Fase 2)

| Métrica | Valor |
|---|---|
| **Val Accuracy** | **98%** |
| **Sensibilidad (recall tumores)** | 91% |
| **Falsos Negativos** | 9% |
| **Falsos Positivos** | 28% |

### Análisis de Errores

La matriz de confusión reveló que el modelo genera **más falsos negativos que positivos** — clasifica tumores reales como tejido sano. Las causas identificadas:

- **Tumores isointensos**: textura similar al tejido cerebral sano
- **Efecto del brillo**: augmentation oscuro reduce el contraste tumor/cerebro
- **Tumores periféricos**: confusión con sombras óseas del cráneo

> En contexto médico, un falso negativo es más peligroso que un falso positivo. Esta métrica fue prioritaria en la evaluación del modelo.

---

## 🖥️ Aplicación Web — Streamlit

Se desarrolló una **app de inferencia en tiempo real** que permite:

- Subir cualquier imagen de resonancia magnética (JPG, PNG)
- Obtener la predicción: `TUMOR DETECTADO` / `SIN TUMOR`
- Ver el porcentaje de confianza del modelo
- Visualizar el desglose de probabilidades por clase en gráfico de barras

```bash
# Ejecutar la app
streamlit run bonus/app.py
```

---

## 🛠️ Stack Tecnológico

| Librería | Uso |
|---|---|
| `TensorFlow 2.x` + `Keras 3.x` | Construcción y entrenamiento de modelos |
| `NumPy` | Operaciones matriciales y preprocesamiento |
| `Pandas` | Análisis y estructuración de datos |
| `Matplotlib` | Visualización de curvas de entrenamiento |
| `scikit-learn` | Métricas: classification report, matriz de confusión |
| `OpenCV` | Procesamiento de imágenes |
| `Streamlit` | Aplicación web de inferencia |
| `Pillow` | Carga y manipulación de imágenes |

---

## 🚀 Cómo ejecutar

### Requisitos
```bash
pip install -r requirements.txt
```

### Entorno recomendado
- Python 3.x
- Google Colab o Kaggle Notebooks (para entrenamiento con GPU)

### Pasos
```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/detector-tumores-cerebrales.git
cd detector-tumores-cerebrales

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar notebooks en orden
# Fase 1 → Fase 2 → Fase 3

# 4. Lanzar la app de inferencia
streamlit run bonus/app.py
```

> ⚠️ El dataset debe descargarse desde Kaggle. Ejecuta la celda de descarga del notebook Fase 1.

---

## 📁 Estructura del Proyecto

```
detector-tumores-cerebrales/
│
├── UDO5_Proyecto_Rodriguez_Fase1.ipynb   # CNN base + sanity checks
├── UDO5_Proyecto_Rodriguez_Fase2.ipynb   # Experimentación sistemática
├── UDO5_Proyecto_Rodriguez_Fase3.ipynb   # Optimización avanzada
│
├── bonus/
│   └── app.py                            # Aplicación Streamlit
│
├── modelos/
│   ├── mejor_modelo_fase1.keras
│   ├── mejor_modelo_fase2.keras          # Modelo principal (98% acc)
│   └── mejor_modelo_fase3.keras
│
├── demo_images/                          # Imágenes de prueba
│   ├── tumorS3.jpg
│   ├── tumorS4.jpg
│   ├── tumorN.jpg
│   └── tumorN5.jpg
│
├── Resultados_tumor/                     # Capturas de resultados
├── requirements.txt
└── README.md
```

---

## 🔗 Aplicaciones y Transferencia

Los conceptos aplicados en este proyecto son directamente transferibles a:

- **Control de calidad industrial**: clasificación de producto por imagen
- **Inspección agrícola**: detección de defectos en plantas o frutos
- **Cualquier dominio de clasificación binaria en imagen**

> El dominio cambia, la metodología es la misma.

---

## 👤 Autor

**Carlos Rodríguez Monzó**  
Curso de Especialización en Inteligencia Artificial y Big Data  
DAW — Desarrollo de Aplicaciones Web

---

## ⚠️ Aviso

Este sistema es una herramienta académica de apoyo al aprendizaje. No sustituye el diagnóstico médico profesional.

---

## 📚 Referencias

- Dataset: [Brain Tumor Detection MRI — Kaggle](https://www.kaggle.com/datasets/abhranta/brain-tumor-detection-mri)
- TensorFlow/Keras Documentation
- Goodfellow et al. — *Deep Learning* (MIT Press)
