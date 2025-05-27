# %% Imports
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

# Librerías de Scikit-learn para preprocesamiento y métricas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# Imbalanced-learn para manejar desbalance de clases
from imblearn.over_sampling import RandomOverSampler

# Componentes de TensorFlow y Keras
import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50 # El modelo base que usaremos
from tensorflow.keras.applications.resnet import preprocess_input # Función de preprocesamiento específica para ResNet
from tensorflow.keras.metrics import AUC # Métrica de Área Bajo la Curva ROC

# Configuración para la visualización y logs de TensorFlow
sns.set_style("whitegrid") # Estilo para los gráficos de Seaborn
tf.get_logger().setLevel('ERROR') # Suprime mensajes informativos de TensorFlow, mostrando solo errores


# %% Carga de datos
base_path = "../images"  # Directorio raíz donde se encuentran las carpetas de categorías
categories = ["Healthy", "Tumor"] # Nombres de las subcarpetas y nuestras clases

image_paths = []  # Lista para almacenar las rutas a cada imagen
labels = []       # Lista para almacenar la etiqueta de cada imagen

# Iterar sobre cada categoría definida
for category in categories:
    category_path = os.path.join(base_path, category) # Construir la ruta a la carpeta de la categoría
    if os.path.isdir(category_path):
        # Iterar sobre cada archivo de imagen dentro de la carpeta de la categoría
        for image_name in os.listdir(category_path):
            image_path = os.path.join(category_path, image_name) # Ruta completa a la imagen
            image_paths.append(image_path) # Añadir la ruta de la imagen a la lista
            labels.append(category)        # Añadir la etiqueta (nombre de la categoría) a la lista
    else:
        print(f"Advertencia: El directorio para la categoría '{category}' no fue encontrado en '{category_path}'")

# Crear un DataFrame de Pandas para almacenar las rutas de las imágenes y sus etiquetas
df = pd.DataFrame({"image_path": image_paths, "label": labels})

# Mostrar las primeras filas del DataFrame y la distribución de clases
print("DataFrame inicial con rutas de imágenes y etiquetas:")
print(df.head())
print("\nDistribución de clases inicial:")
print(df['label'].value_counts())