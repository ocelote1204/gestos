from keras.models import load_model
import cv2
import numpy as np

# Cargar el modelo desde la ruta correcta
model = load_model('D:/vision artificial/artificial_vision/artificial_vision-main (2)/artificial_vision-main/convolutive_neural_networks/trained_model_parameters/final_model.h5')

# Inicializar la cámara
cap = cv2.VideoCapture(0)

while True:
    # Capturar un fotograma de la cámara
    ret, frame = cap.read()

    if not ret:
        print("No se pudo obtener la imagen de la cámara")
        break

    # Preprocesar la imagen
    img = cv2.resize(frame, (128, 128))  # Cambiar el tamaño a 32x32
    img = img / 255.0  # Normalizar
    img = np.expand_dims(img, axis=0)  # Agregar la dimensión del batch

    # Realizar la predicción
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)  # Obtener la clase predicha

    # Mostrar la clase predicha en el fotograma
    label = f'Predicción: {predicted_class[0]}'
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Mostrar el fotograma con la predicción
    cv2.imshow("Clasificación en tiempo real", frame)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
