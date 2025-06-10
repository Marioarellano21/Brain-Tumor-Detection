# %% Imports
# Arreglo de compatibilidad Python 3.10 - DEBE IR PRIMERO
import sys
import collections.abc
collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableMapping = collections.abc.MutableMapping

# Arreglo para NumPy 2.0+ con Keras 2.2.4
import numpy as np
if not hasattr(np, 'Inf'):
    np.Inf = np.inf
print(f"NumPy version: {np.__version__} (arreglo np.Inf aplicado)")

# Configurar PlaidML ANTES de cualquier import de Keras
import plaidml.keras
plaidml.keras.install_backend()

# Imports básicos
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from datetime import datetime

# Librerías de Scikit-learn para preprocesamiento y métricas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# Imbalanced-learn para manejar desbalance de clases
from imblearn.over_sampling import RandomOverSampler

# CAMBIO PRINCIPAL: Usar keras en lugar de tensorflow.keras
import keras
from keras import layers, Model, callbacks
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
# AUC no está disponible en Keras 2.2.4, la calcularemos manualmente

from tqdm import tqdm

# Configuración para la visualización
sns.set_style("whitegrid")

print("🚀 PlaidML configurado para GPU AMD gfx90c")
print(f"✅ Keras version: {keras.__version__}")
print(f"✅ Backend: {keras.backend.backend()}")

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
batch_size = 32  # CONFIGURACIÓN REAL - Aprovechando GPU completa
img_size = (224, 224)  # ResNet requiere 224x224

print(f"🎯 CONFIGURACIÓN PARA ENTRENAMIENTO REAL:")
print(f"   - Batch size: {batch_size}")
print(f"   - Image size: {img_size}")
print(f"   - Épocas: 10 + 10 (total 20)")
print(f"   - GPU Memory: 7GB disponibles")
print(f"   - Learning rates: 1e-3 → 1e-6")

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
print(f"Validación: {valid_gen.samples} imágenes")
print(f"Prueba: {test_gen.samples} imágenes")

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
    x = base_model(inputs)  # Quitar training=False para Keras 2.2.4
    # Reducir la dimensionalidad espacial a un vector por cada mapa de características.
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)  # Capa de Dropout para regularización
    # Capa densa final para la clasificación binaria, con activación sigmoide.
    # Se añade regularización L2 al kernel para prevenir el sobreajuste.
    outputs = layers.Dense(1, activation='sigmoid',
                          kernel_regularizer=keras.regularizers.l2(0.01))(x)

    # Crear el modelo final especificando las entradas y salidas.
    model = Model(inputs, outputs)
    return model

# Instanciar el modelo
model = build_resnet_model()

# Compilación del modelo
model.compile(
    optimizer=Adam(lr=1e-3),  # En Keras 2.2.4 es 'lr' no 'learning_rate'
    loss='binary_crossentropy',         # Función de pérdida para clasificación binaria
    metrics=['accuracy']  # AUC se calculará manualmente
)

# Callbacks para el entrenamiento
callbacks_list = [
    callbacks.EarlyStopping(
        monitor='val_loss',       # Cambiar a val_loss ya que no tenemos val_auc
        patience=5,               # Número de épocas a esperar sin mejora antes de detener
        mode='min',               # Minimizar la pérdida
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
        'best_resnet_plaidml_final.h5', # Nombre del archivo para guardar el mejor modelo
        monitor='val_loss',       # Monitorear pérdida de validación
        save_best_only=True,      # Guarda solo el modelo si la métrica monitoreada ha mejorado
        mode='min',               # El objetivo es minimizar val_loss
        verbose=1
    )
]

# Fase 1: Entrenar solo las capas nuevas
print("\n" + "="*70)
print("🚀 INICIANDO ENTRENAMIENTO COMPLETO EN GPU AMD gfx90c")
print("="*70)

training_start_time = datetime.now()
print(f"⏰ Inicio del entrenamiento: {training_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"🎯 Configuración: ResNet50 + PlaidML + 7GB GPU")
print(f"📊 Dataset: {train_gen.samples} train, {valid_gen.samples} val, {test_gen.samples} test")

print(f"\n📊 FASE 1: Entrenamiento de capas nuevas (10 épocas)")
phase1_start = datetime.now()
print(f"⏰ Inicio Fase 1: {phase1_start.strftime('%H:%M:%S')}")
print(f"🔧 Learning rate: 1e-3")

initial_epochs = 10  # CONFIGURACIÓN REAL
history = model.fit_generator(  # En Keras 2.2.4 es fit_generator
    train_gen,
    validation_data=valid_gen,
    epochs=initial_epochs,
    callbacks=callbacks_list, # Utilizar la lista de callbacks definida
    verbose=1                 # Mostrar barra de progreso e información por época
)

phase1_end = datetime.now()
phase1_duration = phase1_end - phase1_start
print(f"\n✅ FASE 1 COMPLETADA!")
print(f"⏰ Fin Fase 1: {phase1_end.strftime('%H:%M:%S')}")
print(f"⏱️ Duración Fase 1: {phase1_duration}")
print(f"📊 Mejor val_acc Fase 1: {max(history.history['val_acc']):.4f}")
print(f"📊 Mejor val_loss Fase 1: {min(history.history['val_loss']):.4f}")

# Calcular AUC rápido de validación
print(f"\n📈 Calculando AUC intermedio...")
try:
    valid_gen.reset()
    y_pred_probs_val = model.predict_generator(valid_gen, verbose=0)
    y_true_val = valid_gen.classes
    from sklearn.metrics import roc_auc_score
    auc_val_phase1 = roc_auc_score(y_true_val, y_pred_probs_val)
    print(f"🎯 AUC Validación Fase 1: {auc_val_phase1:.4f}")
except Exception as e:
    print(f"⚠️ No se pudo calcular AUC intermedio: {e}")

# Fase 2: Fine-tuning de todo el modelo
print(f"\n📊 FASE 2: Fine-tuning del modelo completo (10 épocas)")
phase2_start = datetime.now()
print(f"⏰ Inicio Fase 2: {phase2_start.strftime('%H:%M:%S')}")
print(f"🔧 Learning rate: 1e-6 (fine-tuning seguro)")

# Acceder a la capa base ResNet50 dentro del modelo 'model'
base_model_layer_from_model = model.layers[1] # Obtenemos la referencia a la capa ResNet50
base_model_layer_from_model.trainable = True   # Hacemos que la capa ResNet50 sea entrenable

# Recompilar el modelo con una tasa de aprendizaje mucho más baja para el fine-tuning
# Esto es esencial para no destruir los pesos pre-entrenados de ResNet50.
model.compile(
    optimizer=Adam(lr=1e-6),  # Learning rate MÁS BAJO para fine-tuning seguro
    loss='binary_crossentropy',
    metrics=['accuracy']  # AUC se calculará manualmente
)

fine_tune_epochs = 10  # CONFIGURACIÓN REAL
total_epochs = initial_epochs + fine_tune_epochs # Número total de épocas de entrenamiento

# Continuar el entrenamiento (fine-tuning)
history_fine = model.fit_generator(  # En Keras 2.2.4 es fit_generator
    train_gen,
    validation_data=valid_gen,
    initial_epoch=history.epoch[-1]+1, # Comenzar el conteo de épocas desde el final de la fase anterior
    epochs=total_epochs,            # Entrenar hasta alcanzar el número total de épocas
    callbacks=callbacks_list,
    verbose=1
)

phase2_end = datetime.now()
phase2_duration = phase2_end - phase2_start
print(f"\n✅ FASE 2 COMPLETADA!")
print(f"⏰ Fin Fase 2: {phase2_end.strftime('%H:%M:%S')}")
print(f"⏱️ Duración Fase 2: {phase2_duration}")
print(f"📊 Mejor val_acc Fase 2: {max(history_fine.history['val_acc']):.4f}")
print(f"📊 Mejor val_loss Fase 2: {min(history_fine.history['val_loss']):.4f}")

# Resumen total del entrenamiento
training_end_time = datetime.now()
total_duration = training_end_time - training_start_time
all_val_acc = history.history['val_acc'] + history_fine.history['val_acc']
all_val_loss = history.history['val_loss'] + history_fine.history['val_loss']

print(f"\n" + "="*70)
print("🎉 ENTRENAMIENTO COMPLETO FINALIZADO")
print("="*70)
print(f"⏰ Inicio total: {training_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"⏰ Fin total: {training_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"⏱️ Duración total: {total_duration}")
print(f"⏱️ Duración Fase 1: {phase1_duration}")
print(f"⏱️ Duración Fase 2: {phase2_duration}")
print(f"📊 Mejor val_acc global: {max(all_val_acc):.4f}")
print(f"📊 Mejor val_loss global: {min(all_val_loss):.4f}")

# Cargar el mejor modelo antes de evaluar
print(f"\n🔄 Cargando mejor modelo guardado...")
try:
    model.load_weights('best_resnet_plaidml_final.h5')
    print(f"✅ Modelo óptimo restaurado")
except Exception as e:
    print(f"⚠️ Error cargando pesos: {e}")
    print(f"✅ Usando modelo actual (callbacks ya restauraron el mejor)")

# Guardar modelo en formatos compatibles para Streamlit
print(f"\n💾 Guardando modelo para uso en Streamlit...")

# Método 1: Intentar guardar modelo completo
try:
    model.save('brain_tumor_model_for_streamlit.h5')
    print(f"✅ Modelo completo guardado: brain_tumor_model_for_streamlit.h5")
    
    # Test de carga inmediata
    test_model = keras.models.load_model('brain_tumor_model_for_streamlit.h5')
    print(f"✅ Test de carga exitoso - modelo funcional")
    del test_model
    
except Exception as e:
    print(f"⚠️ Error con modelo completo: {e}")
    
    # Método 2: Guardar solo pesos (más compatible)
    try:
        print(f"📦 Guardando en formato alternativo...")
        
        # Guardar pesos sin metadata problemática
        model.save_weights('brain_tumor_weights.h5', overwrite=True)
        
        # Guardar arquitectura
        import json
        model_config = model.to_json()
        with open('brain_tumor_architecture.json', 'w') as f:
            json.dump(json.loads(model_config), f, indent=2)
        
        print(f"✅ Pesos guardados: brain_tumor_weights.h5")
        print(f"✅ Arquitectura guardada: brain_tumor_architecture.json")
        
        # Test de reconstrucción
        test_model = keras.models.model_from_json(model_config)
        test_model.load_weights('brain_tumor_weights.h5')
        print(f"✅ Test de reconstrucción exitoso")
        del test_model
        
    except Exception as e2:
        print(f"❌ Error guardando alternativo: {e2}")

# Guardar información esencial para Streamlit
try:
    import pickle
    
    # Guardar label encoder
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Guardar configuración del modelo
    model_info = {
        'input_shape': (224, 224, 3),
        'classes': list(label_encoder.classes_),
        'preprocessing': 'resnet50_preprocess_input',
        'threshold': 0.5,
        'model_type': 'ResNet50_binary_classifier'
    }
    
    with open('model_config.pkl', 'wb') as f:
        pickle.dump(model_info, f)
    
    print(f"✅ Configuración guardada para Streamlit")
    
except Exception as e:
    print(f"⚠️ Error guardando configuración: {e}")

# Guardar modelo completo para uso futuro
print(f"\n💾 Guardando modelo completo para uso futuro...")
try:
    model.save('resnet_tumor_detector_complete.h5')
    print(f"✅ Modelo completo guardado: resnet_tumor_detector_complete.h5")
except Exception as e:
    print(f"⚠️ Error guardando modelo completo: {e}")
    # Alternativa: Guardar arquitectura + pesos por separado
    try:
        # Guardar arquitectura
        model_json = model.to_json()
        with open("resnet_tumor_detector_architecture.json", "w") as json_file:
            json_file.write(model_json)
        # Guardar pesos en formato compatible
        model.save_weights("resnet_tumor_detector_weights.h5", overwrite=True)
        print(f"✅ Modelo guardado por separado:")
        print(f"   - Arquitectura: resnet_tumor_detector_architecture.json")
        print(f"   - Pesos: resnet_tumor_detector_weights.h5")
    except Exception as e2:
        print(f"❌ Error guardando por separado: {e2}")
        print(f"💡 El modelo sigue funcionando en memoria")

# Calcular AUC final de validación
print(f"\n📈 Calculando AUC final de validación...")
try:
    valid_gen.reset()
    y_pred_probs_val_final = model.predict_generator(valid_gen, verbose=0)
    y_true_val_final = valid_gen.classes
    auc_val_final = roc_auc_score(y_true_val_final, y_pred_probs_val_final)
    print(f"🎯 AUC Validación Final: {auc_val_final:.4f}")
except Exception as e:
    print(f"⚠️ No se pudo calcular AUC final de validación: {e}")

print("="*70)

# %% Evaluación del modelo
def evaluate_model(model, test_gen):
    print(f"\n{'='*70}")
    print("📊 EVALUACIÓN FINAL DEL MODELO EN CONJUNTO DE PRUEBA")
    print("="*70)
    
    eval_start = datetime.now()
    print(f"⏰ Inicio evaluación: {eval_start.strftime('%H:%M:%S')}")
    
    # Reiniciar el generador de prueba para asegurar que empieza desde el principio
    test_gen.reset()
    # Evaluar el modelo en el conjunto de prueba
    loss, accuracy = model.evaluate_generator(test_gen, verbose=1)  # Keras 2.2.4 usa evaluate_generator

    print(f"\n📈 RESULTADOS EN CONJUNTO DE PRUEBA:")
    print(f"📊 Pérdida (Loss): {loss:.4f}")
    print(f"📊 Exactitud (Accuracy): {accuracy:.4f}")

    # Obtener las probabilidades predichas por el modelo para el conjunto de prueba
    test_gen.reset()
    y_pred_probs = model.predict_generator(test_gen, verbose=0)  # En Keras 2.2.4 es predict_generator
    # Convertir las probabilidades a predicciones de clase (0 o 1) usando un umbral de 0.5
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    # Obtener las etiquetas verdaderas del generador de prueba
    y_true = test_gen.classes

    # Calcular AUC manualmente
    from sklearn.metrics import roc_auc_score
    auc_score = roc_auc_score(y_true, y_pred_probs)
    
    eval_end = datetime.now()
    eval_duration = eval_end - eval_start
    print(f"📊 AUC: {auc_score:.4f}")
    print(f"⏰ Fin evaluación: {eval_end.strftime('%H:%M:%S')}")
    print(f"⏱️ Duración evaluación: {eval_duration}")

    # Obtener los nombres originales de las clases usando el label_encoder ajustado previamente
    class_names = list(label_encoder.inverse_transform([0, 1]))

    # --- Matriz de Confusión ---
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusión - ResNet50 PlaidML (Final)')
    plt.ylabel('Etiqueta Verdadera')
    plt.xlabel('Etiqueta Predicha')
    plt.show()

    # --- Reporte de Clasificación ---
    print("\n📋 REPORTE DE CLASIFICACIÓN:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # --- Curva ROC ---
    # Calcular la tasa de falsos positivos (fpr) y la tasa de verdaderos positivos (tpr)
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs) # Usar y_pred_probs para la curva ROC
    # Calcular el Área Bajo la Curva ROC
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=3,
             label=f'Curva ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Línea base') # Línea de no discriminación
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.title('Curva ROC - ResNet50 PlaidML (Final)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"{'='*70}")
    return auc_score

# Llamar a la función de evaluación con el modelo entrenado y el generador de prueba
final_auc = evaluate_model(model, test_gen)

print(f"\n🏆 RESUMEN FINAL DEL PROYECTO:")
print(f"🎯 AUC Final: {final_auc:.4f}")
print(f"⏱️ Tiempo total entrenamiento: {total_duration}")
print(f"🚀 Entrenamiento completado exitosamente en GPU AMD gfx90c!")
print(f"💾 Modelo guardado en: best_resnet_plaidml_final.h5")

# %% Visualización del historial de entrenamiento
def plot_combined_history(initial_hist, fine_tune_hist):
    # Extraer métricas de la fase inicial
    acc = initial_hist.history['acc']
    val_acc = initial_hist.history['val_acc']
    loss = initial_hist.history['loss']
    val_loss = initial_hist.history['val_loss']

    # Añadir métricas de la fase de fine-tuning
    acc += fine_tune_hist.history['acc']
    val_acc += fine_tune_hist.history['val_acc']
    loss += fine_tune_hist.history['loss']
    val_loss += fine_tune_hist.history['val_loss']

    plt.figure(figsize=(15, 6))

    # Subgráfico para la Exactitud
    plt.subplot(1, 2, 1)
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo-', label='Exactitud de Entrenamiento', markersize=4)
    plt.plot(epochs, val_acc, 'ro-', label='Exactitud de Validación', markersize=4)
    # Línea vertical para marcar el inicio del fine-tuning
    plt.axvline(x=initial_epochs, color='green', linestyle='--', linewidth=2, label='Inicio Fine-Tuning')
    plt.title('Exactitud durante Entrenamiento - ResNet50 PlaidML', fontsize=14)
    plt.xlabel('Época')
    plt.ylabel('Exactitud')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subgráfico para la Pérdida
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Pérdida de Entrenamiento', markersize=4)
    plt.plot(epochs, val_loss, 'ro-', label='Pérdida de Validación', markersize=4)
    # Línea vertical para marcar el inicio del fine-tuning
    plt.axvline(x=initial_epochs, color='green', linestyle='--', linewidth=2, label='Inicio Fine-Tuning')
    plt.title('Pérdida durante Entrenamiento - ResNet50 PlaidML', fontsize=14)
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout() # Ajustar el layout para evitar superposiciones
    plt.show()

# Llamar a la función para graficar el historial combinado
print(f"\n📈 Generando gráficas del historial de entrenamiento...")
plot_combined_history(history, history_fine)

print(f"\n🎉 ¡PROYECTO COMPLETADO EXITOSAMENTE!")
print(f"📊 Resumen de rendimiento de GPU AMD gfx90c:")
print(f"   - Total épocas: {len(all_val_acc)}")
print(f"   - Mejor accuracy: {max(all_val_acc):.4f}")
print(f"   - AUC final: {final_auc:.4f}")
print(f"   - Tiempo promedio/época: {total_duration.total_seconds()/(len(all_val_acc)*60):.1f} min")
print(f"🚀 GPU AMD funcionando perfectamente para Deep Learning!")