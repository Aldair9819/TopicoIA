#Listo. No hay nada que revisar

#Liberias para usar cv2, numpy y tensorflow para cargar modelo
import cv2
import numpy as np
import face_recognition
from tensorflow.keras.models import load_model
from PIL import Image

# Cargar modelo mediante tensorflow, en h5
modelo1 = 'modelo_mlp_Estratificado.h5' #SI PASA
modelo2 = 'modelo_mlp_NoEstratificado.h5' #SI PASA
modelo_emociones = load_model('/home/waldos/Documents/2doCodigo/TopicoIA/Abril/Camera/Final_Aldair/Emotions_3/'+modelo1)

#Las etiquetas del modelo, dado que está en y_oneHot
#labels = ['bored', 'engaged', 'excited' ,'focused', 'interested', 'relaxed']
labels = ['bored', 'engaged' ,'excited']





def extract_face_from_image(frame_cara):
    # Cambiar el tamaño de la imagen
    #gray = cv2.cvtColor(frame_cara, cv2.COLOR_RGB2GRAY)
    
    # Obtener los landmarks faciales
    caracteristicas_faciales_lista = face_recognition.face_landmarks(frame_cara)[0]
    
    landmarks = []
    for facial_feature in caracteristicas_faciales_lista.keys():
        landmarks.extend(caracteristicas_faciales_lista[facial_feature])
    
    caracteristicas_faciales_numpy = np.array(landmarks)

    puntosX_YRostro = face_recognition.face_locations(frame_cara)
        
    #Recorta la cara
    puntos_ubicacion_cara = puntosX_YRostro[0]
    arriba, derecha, abajo, izquierda = puntos_ubicacion_cara

    return caracteristicas_faciales_numpy, arriba, derecha, abajo, izquierda 

        
def impresionCara(frame_carita, arriba, derecha, abajo, izquierda):
    cara_recortada = frame_carita[arriba:abajo, izquierda:derecha]    
    cara_numpy = np.array(cara_recortada)

    cara_redimensionada = np.array(Image.fromarray(cara_numpy).resize((150,150)))
    cv2.imshow('Cara', cara_redimensionada)      

#Funcion que detecta y predice las emociones
def detectar_y_predecir_emociones(frame, modelo):
    # Convertir el frame a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    try:
        extraccion_caracteristicas, arriba, derecha, abajo, izquierda = extract_face_from_image(gray)
        
        #impresionCara(gray, arriba, derecha, abajo, izquierda)
        # Detectar los rostros en la imagen

        # Normalizar el frame
        normalized_frame = extraccion_caracteristicas
        
        # Agregar una dimensión adicional para la compatibilidad con la red neuronal
        normalized_frame = np.expand_dims(normalized_frame, axis=0)

        #cara_normalizada = cara_redimensionada / 255.0
        cara_predicion = modelo.predict(normalized_frame)
        idx_etiqueta = np.argmax(cara_predicion)
        etiqueta = labels[idx_etiqueta]
        print(etiqueta)
        print(idx_etiqueta)
        cv2.putText(frame, etiqueta, (izquierda, arriba - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.rectangle(frame, (izquierda, arriba), (derecha, abajo), (0, 255, 0), 2)


    except Exception as e:
        if str(e) == 'list index out of range':
            pass
        else:
            print(e)


# Iniciar la captura de video desde la cámara web
video_capture = cv2.VideoCapture(0)


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
