# Reconocimiento de Gestos con MobileNetV2

Este proyecto implementa un sistema de clasificación de imágenes para reconocimiento de gestos utilizando aprendizaje por transferencia (Transfer Learning) con la arquitectura **MobileNetV2** preentrenada sobre ImageNet.

## 📁 Estructura del proyecto


Cada una de las carpetas `train`, `valid` y `test` debe contener subcarpetas con las clases (por ejemplo, `puño`, `mano_abierta`, etc.), y dentro de ellas las imágenes.

---

## 🧠 Modelo

Se utilizó **MobileNetV2** como base, congelando sus capas y añadiendo una cabeza densa para clasificación:

- Capa base: `MobileNetV2(weights='imagenet', include_top=False)`
- Capas añadidas:
  - GlobalAveragePooling2D
  - Dense(128, ReLU)
  - Dropout(0.3)
  - Dense(num_clases, Softmax)

---

## ⚙️ Entrenamiento

Parámetros principales:

- Tamaño de imagen: `128x128`
- Épocas: `50`
- Batch size: `64`
- Optimizador: `Adam`
- Función de pérdida: `categorical_crossentropy`
- Métrica: `accuracy`

También se usaron técnicas como:

- **Aumento de datos** con `ImageDataGenerator`
- **EarlyStopping** para evitar sobreentrenamiento
- **ReduceLROnPlateau** para ajustar la tasa de aprendizaje
- **ModelCheckpoint** para guardar el mejor modelo automáticamente

---

## 🧪 Evaluación

Al finalizar el entrenamiento, el modelo se evalúa automáticamente en el conjunto de prueba (`test/`) y se guarda:



---

## ▶️ Cómo usar

### Entrenamiento:

```bash
python main.py <ruta_al_directorio_data>
