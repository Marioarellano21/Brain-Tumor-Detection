# Test final para verificar que PlaidML funciona
import sys
import collections.abc

# Aplicar parche de compatibilidad
collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableMapping = collections.abc.MutableMapping

print("=== TEST PLAIDML FINAL ===")

try:
    import plaidml.keras
    plaidml.keras.install_backend()
    import keras
    
    print("✅ PlaidML importado correctamente")
    print(f"✅ Keras version: {keras.__version__}")
    print(f"✅ Backend: {keras.backend.backend()}")
    
    # Test simple de operación
    from keras import layers
    import numpy as np
    
    print("\n🧪 Test de operación en GPU...")
    
    # Crear modelo simple
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(10,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Datos de prueba
    x = np.random.random((100, 10))
    y = np.random.randint(2, size=(100, 1))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    import time
    start = time.time()
    model.fit(x, y, epochs=2, verbose=0)
    gpu_time = time.time() - start
    
    print(f"✅ Entrenamiento en GPU exitoso: {gpu_time:.4f}s")
    print("🚀 PlaidML listo para entrenar tu modelo ResNet50!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Revisa la instalación")