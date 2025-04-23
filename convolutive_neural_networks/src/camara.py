from keras.models import load_model
import cv2
import numpy as np
from pathlib import Path

# Obtener la ruta del archivo actual y construir la ruta al modelo
base_path = Path(__file__).resolve().parent  # Carpeta donde está el script
model_path = base_path / '..' / 'trained_model_parameters' / 'final_model.h5'
model_path = model_path.resolve()  # Obtener la ruta absoluta final

# Cargar el modelo
model = load_model(str(model_path))

# Inicializar la cámara
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo obtener la imagen de la cámara")
        break

    # Preprocesar la imagen
    img = cv2.resize(frame, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Realizar la predicción
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)

    label = f'Predicción: {predicted_class[0]}'
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Clasificación en tiempo real", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
