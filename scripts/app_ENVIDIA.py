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

from tqdm import tqdm
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


# Configuración para la visualización y logs de TensorFlow
sns.set_style("whitegrid") # Estilo para los gráficos de Seaborn
tf.get_logger().setLevel('ERROR') # Suprime mensajes informativos de TensorFlow, mostrando solo errores


# %% Carga de datos
base_path = "../images"  # Directorio raíz donde se encuentran las carpetas de categorías
categories = ["Healthy", "Tumor"] # Nombres de las subcarpetas y nuestras clases

image_paths = []  # Lista para almacenar las rutas a cada imagen
labels = []       # Lista para almacenar la etiqueta de cada imagen

valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')  # Opcional, para filtrar imágenes

# Iterar sobre cada categoría definida
for category in categories:
    category_path = os.path.join(base_path, category)
    if os.path.isdir(category_path):
        for image_name in tqdm(os.listdir(category_path), desc=f"Cargando {category}"):
            if image_name.lower().endswith(valid_extensions):  # Filtra solo imágenes
                image_path = os.path.join(category_path, image_name)
                image_paths.append(image_path)
                labels.append(category)
    else:
        print(f"Advertencia: El directorio para la categoría '{category}' no fue encontrado en '{category_path}'")

# Crear un DataFrame de Pandas para almacenar las rutas de las imágenes y sus etiquetas
df = pd.DataFrame({"image_path": image_paths, "label": labels})

# Mostrar las primeras filas del DataFrame y la distribución de clases
print("DataFrame inicial con rutas de imágenes y etiquetas:")
print(df.head())
print("\nDistribución de clases inicial:")
print(df['label'].value_counts())

# %% Preprocesamiento
# Codificación de etiquetas
label_encoder = LabelEncoder()
# Se crea una nueva columna 'category_encoded' con las etiquetas numéricas (ej. 0 para Healthy, 1 para Tumor)
df['category_encoded'] = label_encoder.fit_transform(df['label'])

print("DataFrame después de la codificación de etiquetas:")
print(df.head())
print(f"Clases codificadas: {label_encoder.classes_} -> {label_encoder.transform(label_encoder.classes_)}")


# Primero, dividir en entrenamiento (80%) y un conjunto temporal (20% para validación + prueba)
X_train_original, X_temp, y_train_original, y_temp = train_test_split(
    df[['image_path']],  # Características (rutas de imagen como DataFrame)
    df['category_encoded'], # Etiquetas numéricas
    train_size=0.8,         # 80% para entrenamiento
    shuffle=True,           # Mezclar los datos antes de dividir
    random_state=42,        # Para reproducibilidad
    stratify=df['category_encoded'] # Asegurar proporciones de clase similares en la división
)

# Luego, dividir el conjunto temporal en validación (50% de temp -> 10% del total)
X_valid, X_test, y_valid, y_test = train_test_split(
    X_temp,                 # DataFrame con rutas de imagen del conjunto temporal
    y_temp,                 # Etiquetas del conjunto temporal
    test_size=0.5,          # 50% de X_temp para el conjunto de prueba (el resto para validación)
    shuffle=True,
    random_state=42,
    stratify=y_temp         # Estratificar sobre las etiquetas del conjunto temporal
)

ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train_original, y_train_original)

# train_df utilizará los datos de entrenamiento sobremuestreados.
train_df = pd.DataFrame(X_train_resampled, columns=['image_path'])
train_df['category_encoded'] = y_train_resampled.astype(str)

# valid_df y test_df utilizan los datos originales de validación y prueba, sin sobremuestreo.
valid_df = pd.DataFrame(X_valid, columns=['image_path'])
valid_df['category_encoded'] = y_valid.astype(str)

test_df = pd.DataFrame(X_test, columns=['image_path'])
test_df['category_encoded'] = y_test.astype(str)

# %% Generadores de datos para ResNet
batch_size = 32
img_size = (224, 224)  # ResNet requiere 224x224

# Data augmentation para entrenamiento
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # Preprocesamiento específico de ResNet
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Solo preprocesamiento para validación y prueba
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)


# Generador para el conjunto de entrenamiento
train_gen = train_datagen.flow_from_dataframe(
    train_df,                   # DataFrame con datos de entrenamiento (sobremuestreados)
    x_col='image_path',         # Columna con las rutas de las imágenes
    y_col='category_encoded',   # Columna con las etiquetas codificadas (como string)
    target_size=img_size,       # Tamaño al que se redimensionarán las imágenes
    class_mode='binary',        # Para clasificación binaria
    color_mode='rgb',           # Cargar imágenes en color
    shuffle=True,               # Mezclar los datos de entrenamiento en cada época
    batch_size=batch_size
)

# Generador para el conjunto de validación
valid_gen = test_datagen.flow_from_dataframe(
    valid_df,                   # DataFrame con datos de validación (originales)
    x_col='image_path',
    y_col='category_encoded',
    target_size=img_size,
    class_mode='binary',
    color_mode='rgb',
    shuffle=True,               # Mezclar datos de validación (opcional, pero True en el script)
    batch_size=batch_size
)

# Generador para el conjunto de prueba
test_gen = test_datagen.flow_from_dataframe(
    test_df,                    # DataFrame con datos de prueba (originales)
    x_col='image_path',
    y_col='category_encoded',
    target_size=img_size,
    class_mode='binary',
    color_mode='rgb',
    shuffle=False,              # IMPORTANTE: No mezclar el conjunto de prueba
    batch_size=batch_size
)

print("\nGeneradores creados:")
print(f"Entrenamiento: {train_gen.samples} imágenes")
print(train_gen)
print(f"Validación: {valid_gen.samples} imágenes")
print(valid_gen)
print(f"Prueba: {test_gen.samples} imágenes")
print(test_gen)


# %% Construcción del modelo ResNet50
def build_resnet_model(input_shape=(224, 224, 3)):
    # Cargar ResNet50 preentrenada (sin las capas superiores de clasificación de ImageNet)
    base_model = ResNet50(
        include_top=False,      # No incluir la capa densa final de ResNet50
        weights='imagenet',     # Usar pesos pre-entrenados en ImageNet
        input_shape=input_shape # Definir la forma de entrada de las imágenes
    )

    # Congelar las capas del modelo base inicialmente.
    # Sus pesos no se actualizarán durante la primera fase de entrenamiento.
    base_model.trainable = False

    # Construir el modelo completo añadiendo nuestras propias capas encima de ResNet50
    inputs = layers.Input(shape=input_shape) # Capa de entrada
    # Pasar las entradas a través del modelo base.
    x = base_model(inputs, training=False)
    # Reducir la dimensionalidad espacial a un vector por cada mapa de características.
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)  # Capa de Dropout para regularización
    # Capa densa final para la clasificación binaria, con activación sigmoide.
    # Se añade regularización L2 al kernel para prevenir el sobreajuste.
    outputs = layers.Dense(1, activation='sigmoid',
                          kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)

    # Crear el modelo final especificando las entradas y salidas.
    model = Model(inputs, outputs)
    return model

# Instanciar el modelo
model = build_resnet_model()



# Compilación del modelo
model.compile(
    optimizer=Adam(learning_rate=1e-3), # Optimizador Adam con una tasa de aprendizaje inicial
    loss='binary_crossentropy',         # Función de pérdida para clasificación binaria
    metrics=['accuracy', AUC(name='auc')] # Métricas a monitorear
)


# Callbacks para el entrenamiento
callbacks_list = [
    callbacks.EarlyStopping(
        monitor='val_auc',        # Métrica a monitorear (AUC en el conjunto de validación)
        patience=5,               # Número de épocas a esperar sin mejora antes de detener
        mode='max',               # Indica que buscamos maximizar la métrica (AUC)
        restore_best_weights=True,# Restaura los pesos del modelo de la mejor época al finalizar
        verbose=1                 # Muestra mensajes cuando el callback se activa
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',       # Métrica a monitorear (pérdida en el conjunto de validación)
        factor=0.1,               # Factor por el cual se reduce la tasa de aprendizaje (new_lr = lr * factor)
        patience=3,               # Número de épocas a esperar sin mejora antes de reducir LR
        verbose=1
    ),
    callbacks.ModelCheckpoint(
        'best_resnet_model.h5',   # Nombre del archivo para guardar el mejor modelo
        monitor='val_auc',        # Métrica que determina si el modelo es "mejor"
        save_best_only=True,      # Guarda solo el modelo si la métrica monitoreada ha mejorado
        mode='max',               # El objetivo es maximizar val_auc
        verbose=1
    )
]


# Fase 1: Entrenar solo las capas nuevas
print("\nEntrenando capas nuevas...")
initial_epochs = 10 # Número de épocas para la primera fase de entrenamiento
history = model.fit(
    train_gen,
    validation_data=valid_gen,
    epochs=initial_epochs,
    callbacks=callbacks_list, # Utilizar la lista de callbacks definida
    verbose=1                 # Mostrar barra de progreso e información por época
)


# Fase 2: Fine-tuning de todo el modelo
print("\nFine-tuning de todo el modelo...")
# Acceder a la capa base ResNet50 dentro del modelo 'model'
base_model_layer_from_model = model.layers[1] # Obtenemos la referencia a la capa ResNet50
base_model_layer_from_model.trainable = True   # Hacemos que la capa ResNet50 sea entrenable

# Recompilar el modelo con una tasa de aprendizaje mucho más baja para el fine-tuning
# Esto es esencial para no destruir los pesos pre-entrenados de ResNet50.
model.compile(
    optimizer=Adam(learning_rate=1e-5), # Tasa de aprendizaje significativamente menor
    loss='binary_crossentropy',
    metrics=['accuracy', AUC(name='auc')]
)

fine_tune_epochs = 10 # Número de épocas adicionales para el fine-tuning
total_epochs = initial_epochs + fine_tune_epochs # Número total de épocas de entrenamiento

# Continuar el entrenamiento (fine-tuning)
history_fine = model.fit(
    train_gen,
    validation_data=valid_gen,
    initial_epoch=history.epoch[-1]+1, # Comenzar el conteo de épocas desde el final de la fase anterior
    epochs=total_epochs,            # Entrenar hasta alcanzar el número total de épocas
    callbacks=callbacks_list,
    verbose=1
)


# %% Evaluación del modelo
def evaluate_model(model, test_gen):
    # Reiniciar el generador de prueba para asegurar que empieza desde el principio
    test_gen.reset()
    # Evaluar el modelo en el conjunto de prueba
    loss, accuracy, auc_score = model.evaluate(test_gen, verbose=0)

    print(f"\nResultados en conjunto de prueba:")
    print(f"Pérdida (Loss): {loss:.4f}")
    print(f"Exactitud (Accuracy): {accuracy:.4f}")
    print(f"AUC: {auc_score:.4f}")

    # Obtener las probabilidades predichas por el modelo para el conjunto de prueba
    y_pred_probs = model.predict(test_gen)
    # Convertir las probabilidades a predicciones de clase (0 o 1) usando un umbral de 0.5
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    # Obtener las etiquetas verdaderas del generador de prueba
    y_true = test_gen.classes

    # Obtener los nombres originales de las clases usando el label_encoder ajustado previamente
    class_names = list(label_encoder.inverse_transform([0, 1]))

    # --- Matriz de Confusión ---
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusión - ResNet50')
    plt.ylabel('Etiqueta Verdadera')
    plt.xlabel('Etiqueta Predicha')
    plt.show()

    # --- Reporte de Clasificación ---
    print("\nReporte de Clasificación:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # --- Curva ROC ---
    # Calcular la tasa de falsos positivos (fpr) y la tasa de verdaderos positivos (tpr)
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs) # Usar y_pred_probs para la curva ROC
    # Calcular el Área Bajo la Curva ROC
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'Curva ROC (área = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Línea de no discriminación
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.title('Curva ROC - ResNet50')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

# Llamar a la función de evaluación con el modelo entrenado y el generador de prueba
evaluate_model(model, test_gen)


# %% Visualización del historial de entrenamiento
def plot_combined_history(initial_hist, fine_tune_hist):
    # Extraer métricas de la fase inicial
    acc = initial_hist.history['accuracy']
    val_acc = initial_hist.history['val_accuracy']
    loss = initial_hist.history['loss']
    val_loss = initial_hist.history['val_loss']

    # Añadir métricas de la fase de fine-tuning
    acc += fine_tune_hist.history['accuracy']
    val_acc += fine_tune_hist.history['val_accuracy']
    loss += fine_tune_hist.history['loss']
    val_loss += fine_tune_hist.history['val_loss']

    plt.figure(figsize=(12, 6))

    # Subgráfico para la Exactitud
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Exactitud de Entrenamiento')
    plt.plot(val_acc, label='Exactitud de Validación')
    # Línea vertical para marcar el inicio del fine-tuning
    # Asumiendo que initial_epochs fue definido como en el script original
    plt.plot([initial_epochs-1, initial_epochs-1],
             plt.ylim(), label='Inicio Fine-Tuning', linestyle='--')
    plt.title('Exactitud durante Entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Exactitud')
    plt.legend()
    plt.grid(True)

    # Subgráfico para la Pérdida
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Pérdida de Entrenamiento')
    plt.plot(val_loss, label='Pérdida de Validación')
    # Línea vertical para marcar el inicio del fine-tuning
    plt.plot([initial_epochs-1, initial_epochs-1],
             plt.ylim(), label='Inicio Fine-Tuning', linestyle='--')
    plt.title('Pérdida durante Entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.grid(True)

    plt.tight_layout() # Ajustar el layout para evitar superposiciones
    plt.show()

# Llamar a la función para graficar el historial combinado
plot_combined_history(history, history_fine)