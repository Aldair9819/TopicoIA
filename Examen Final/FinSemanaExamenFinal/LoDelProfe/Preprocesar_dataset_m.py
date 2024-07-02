import os
import pandas as pd
import face_recognition
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
from concurrent.futures import ThreadPoolExecutor

# Definir la ruta de la carpeta que contiene las imágenes
ruta_carpeta = 'C:\\Users\\Waldosir\\Documents\\2doCodigo\\TopicoIA\\Marzo\\BD\\corpus_images'

# Definir el nombre del archivo de la base de datos .pickle
archivo_pickle = 'datos_imagenes.pickle'

# Función para procesar una imagen y obtener los datos
def procesar_imagen(archivo_ruta):
    # Cargar la imagen utilizando PIL (Pillow)
    imagen = face_recognition.load_image_file(archivo_ruta)

    # Redimensionar la imagen a 150x150
    imagen_redimensionada = Image.fromarray(imagen).resize((150, 150))
    imagen_redimensionada_np = np.array(imagen_redimensionada)

    # Detectar los rostros en la imagen
    rostros = face_recognition.face_locations(imagen_redimensionada_np)

    # Obtener los landmarks faciales
    landmarks = face_recognition.face_landmarks(imagen_redimensionada_np)

    return imagen_redimensionada, rostros, landmarks

# Comprobar si el archivo de la base de datos ya existe
if os.path.exists(archivo_pickle):
    # Cargar la base de datos desde el archivo .pickle
    with open(archivo_pickle, 'rb') as f:
        df_imagenes = pickle.load(f)
else:
    # Crear una lista para almacenar los datos de las imágenes
    datos_imagenes = []

    # Utilizar ThreadPoolExecutor para procesar imágenes en paralelo
    with ThreadPoolExecutor() as executor:
        for carpeta_nombre in os.listdir(ruta_carpeta):
            carpeta_ruta = os.path.join(ruta_carpeta, carpeta_nombre)
            if os.path.isdir(carpeta_ruta):
                for archivo_nombre in os.listdir(carpeta_ruta):
                    if archivo_nombre.endswith('.jpeg'):
                        archivo_ruta = os.path.join(carpeta_ruta, archivo_nombre)

                        # Procesar la imagen en paralelo
                        future = executor.submit(procesar_imagen, archivo_ruta)
                        imagen_redimensionada, rostros, landmarks = future.result()

                        # Agregar la información de la imagen al DataFrame
                        for rostro, landmark in zip(rostros, landmarks):
                            datos_imagenes.append([imagen_redimensionada, carpeta_nombre, rostro, landmark])

    # Crear un DataFrame con los datos de las imágenes
    columnas = ['Imagen', 'Etiqueta', 'Rostro', 'Landmarks']
    df_imagenes = pd.DataFrame(datos_imagenes, columns=columnas)

    # Guardar la base de datos en un archivo .pickle
    with open(archivo_pickle, 'wb') as f:
        pickle.dump(df_imagenes, f)

# Elegir aleatoriamente 5 ejemplos del DataFrame
df_ejemplos = df_imagenes.sample(n=5)

# Visualizar las imágenes seleccionadas con los rostros detectados y los landmarks faciales
fig, axes = plt.subplots(nrows=len(df_ejemplos), ncols=4, figsize=(16, 16))

for i, (index, row) in enumerate(df_ejemplos.iterrows()):
    # Subgráfico para la imagen original
    ax0 = axes[i, 0]
    ax0.imshow(row['Imagen'])
    ax0.set_title("Imagen Original")
    ax0.axis('off')

    # Subgráfico para la imagen recortada con el rostro detectado
    ax1 = axes[i, 1]
    # Convertir la imagen a un array numpy para poder acceder a sus elementos
    imagen_np = np.array(row['Imagen'])
    imagen_recortada = imagen_np[row['Rostro'][0]:row['Rostro'][2], row['Rostro'][3]:row['Rostro'][1]]
    ax1.imshow(imagen_recortada)
    ax1.set_title("Imagen Recortada")
    ax1.axis('off')
    
    # Subgráfico para los landmarks faciales
    ax2 = axes[i, 2]
    ax2.imshow(row['Imagen'])
    for landmark_tipo, landmark_puntos in row['Landmarks'].items():
        for punto in landmark_puntos:
            ax2.plot(punto[0], punto[1], marker='o', markersize=6, color='blue')
    ax2.set_title("Landmarks Faciales")
    ax2.axis('off')

    # Subgráfico para la emoción (si está disponible)
    ax3 = axes[i, 3]
    # Aquí puedes agregar código para mostrar la emoción en lugar de un gráfico vacío
    ax3.text(0.5, 0.5, row['Etiqueta'], horizontalalignment='center', verticalalignment='center', fontsize=12)
    ax3.set_title("Emoción")
    ax3.axis('off')

plt.tight_layout()
plt.show()
