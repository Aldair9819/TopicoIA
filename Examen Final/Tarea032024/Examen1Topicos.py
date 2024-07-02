import os
import pandas as pd
import face_recognition
from PIL import Image
import matplotlib.pyplot as plt
import random
import numpy as np

# Definir la ruta de la carpeta que contiene las imágenes
ruta_carpeta = 'C:\\Users\\Waldosir\\Documents\\2doCodigo\\TopicoIA\\Marzo\\BD\\corpus_images'

# Crear una lista para almacenar los datos de las imágenes
datos_imagen_recortada = []
# Recorrer la carpeta para buscar archivos jpg
for carpeta_nombre in os.listdir(ruta_carpeta):
    carpeta_ruta = os.path.join(ruta_carpeta, carpeta_nombre)
    if os.path.isdir(carpeta_ruta):
        for archivo_nombre in os.listdir(carpeta_ruta):
            try:
                archivo_ruta = os.path.join(carpeta_ruta, archivo_nombre)
                
                # Cargar la imagen utilizando PIL (Pillow)
                imagen = face_recognition.load_image_file(archivo_ruta)
                
                # Redimensionar la imagen a 150x150
                imagen_redimensionada = np.array(Image.fromarray(imagen).resize((150, 150)))
                # Obtener los landmarks faciales
                landmarks = face_recognition.face_landmarks(imagen_redimensionada)
                
                caracteristicas = []
                for clave,valor in landmarks[0].items():
                    caracteristicas.append(valor)

                # Detectar los rostros en la imagen
                rostros = face_recognition.face_locations(imagen_redimensionada)

                #Recorta la cara
                cara = rostros[0]
                top, right, bottom, left = cara
                cara_recortada = imagen_redimensionada[top:bottom, left:right]
                
                # Agregar la información de la imagen al DataFrame
                datos_imagen_recortada.append([imagen_redimensionada, caracteristicas,cara_recortada,carpeta_nombre])
            except Exception as e:
                print("Error en archivo:"+str(archivo_nombre))
print("Proceso terminado")
columnas = ['Imagen', 'Caracteristicas', 'Rostro', 'Etiqueta']
df_imagenes_recortadas = pd.DataFrame(datos_imagen_recortada, columns=columnas)

df_ejemploHDF = df_imagenes_recortadas.sample(n=5)

# Visualizar las imágenes seleccionadas con los rostros detectados y los landmarks faciales
fig, axes = plt.subplots(nrows=len(df_ejemploHDF), ncols=4, figsize=(16, 16))

for i, (index, row) in enumerate(df_ejemploHDF.iterrows()):
    # Subgráfico para la imagen original
    ax0 = axes[i, 0]
    ax0.imshow(row['Imagen'])
    ax0.set_title("Imagen Original")
    ax0.axis('off')
    
    # Subgráfico para la imagen recortada con el rostro detectado
    ax1 = axes[i, 1]
    imagen_recortada = row['Rostro']
    ax1.imshow(imagen_recortada)
    ax1.set_title("Imagen Recortada")
    ax1.axis('off')
    
    # Subgráfico para los landmarks faciales
    ax2 = axes[i, 2]
    ax2.imshow(row['Imagen'])

    for landmark in row['Caracteristicas']:
        for punto in landmark:
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


df_imagenes_recortadas.to_hdf('testHDF.h5', key='dataframe', mode='w')
df_HDF = pd.read_hdf('testHDF.h5')

df_ejemploHDF = df_HDF.sample(n=5)

# Visualizar las imágenes seleccionadas con los rostros detectados y los landmarks faciales
fig, axes = plt.subplots(nrows=len(df_ejemploHDF), ncols=4, figsize=(16, 16))

for i, (index, row) in enumerate(df_ejemploHDF.iterrows()):
    # Subgráfico para la imagen original
    ax0 = axes[i, 0]
    ax0.imshow(row['Imagen'])
    ax0.set_title("Imagen Original")
    ax0.axis('off')
    
    # Subgráfico para la imagen recortada con el rostro detectado
    ax1 = axes[i, 1]
    imagen_recortada = row['Rostro']
    ax1.imshow(imagen_recortada)
    ax1.set_title("Imagen Recortada")
    ax1.axis('off')
    
    # Subgráfico para los landmarks faciales
    ax2 = axes[i, 2]
    ax2.imshow(row['Imagen'])

    for landmark in row['Caracteristicas']:
        for punto in landmark:
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