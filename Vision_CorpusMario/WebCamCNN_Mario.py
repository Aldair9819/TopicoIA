#Liberias para usar cv2, numpy y tensorflow para cargar modelo
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Cargar modelo mediante tensorflow, en h5
modelo_emociones = load_model("Abril/modelo_cnn.h5")

#Las etiquetas del modelo, dado que está en y_oneHot
labels = ['bored', 'engaged', 'excited', 'focused', 'interested', 'relaxed']

#Funcion que detecta y predice las emociones
def detectar_y_predecir_emociones(frame, modelo):
    # Convertir el frame a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    
    # Detectar caras en el frame
    caras = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Para cada cara detectada
    for (x, y, w, h) in caras:
        # Extraer la región de interés (ROI) correspondiente a la cara
        roi = frame[y:y+h, x:x+w]
        
        # Redimensionar la ROI a 150x150 (para que coincida con el input_shape del modelo)
        roi_resized = cv2.resize(roi, (150, 150))
        
        # Normalizar la imagen (dividiendo por 255)
        roi_normalized = roi_resized / 255.0
        
        # Realizar la predicción de emociones en la ROI
        roi_predicted = modelo.predict(np.expand_dims(roi_normalized, axis=0))#[0]
        
        # Obtener el índice de la etiqueta con mayor probabilidad
        idx_etiqueta = np.argmax(roi_predicted)
        
        # Obtener la etiqueta correspondiente
        etiqueta = labels[idx_etiqueta]
        
        # Mostrar la etiqueta en la pantalla
        cv2.putText(frame, etiqueta, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Dibujar un rectángulo alrededor de la cara detectada
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Iniciar la captura de video desde la cámara web
video_capture = cv2.VideoCapture(0)

# Cargar el clasificador de Haar para la detección de caras
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    # Capturar un frame de la cámara
    ret, frame = video_capture.read()

    # Detectar y predecir emociones en el frame
    detectar_y_predecir_emociones(frame, modelo_emociones)

    # Mostrar el frame resultante
    cv2.imshow('Emociones en tiempo real', frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura de video y cerrar las ventanas
video_capture.release()
cv2.destroyAllWindows()
