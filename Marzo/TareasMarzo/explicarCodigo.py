import face_recognition
from PIL import Image
from PIL import ImageDraw
import PIL.Image as Image
import os
import base64
from PIL import Image
from io import BytesIO
import numpy as np

def imprimirCara(imagen):
    pil_image = Image.fromarray(imagen)
    pil_image.show()
    
def imprimirTodo(imagen,cara):
    imprimirCara(imagen)
    pil_image = Image.fromarray(recortarCara(imagen,cara))
    pil_image.show()

def recortarCara(imagen, cara):
    top, right, bottom, left = cara
    cara_recortada = imagen[top:bottom, left:right]
    return cara_recortada

def hasFace(face_locations):
    if len(face_locations) > 0:
        return True
    else:
        return False

def info_foto(ruta, imagen):
    #Carga imagen
    imagen = face_recognition.load_image_file(ruta)
    #Imagen en String
    
    #Busca la cara
    face_locations = face_recognition.face_locations(imagen)
    if not hasFace(face_locations):
        return
    
    #Recorta la cara
    cara = face_locations[0]

    #Patrones de la cara
    face_landmarks_list = face_recognition.face_landmarks(imagen)
    caracteristicas = []
    for clave,valor in face_landmarks_list[0].items():
        caracteristicas.append(valor)
    
    folder_name = os.path.basename(os.path.dirname(ruta))
    
    return [imagen, caracteristicas,cara, folder_name]

def sacar_datos_imagen(carpeta_raiz):
    # Recorrer todas las carpetas, subcarpetas y archivos de la carpeta raíz
    for carpeta_actual, subcarpetas, archivos in os.walk(carpeta_raiz):
        # Recorrer todos los archivos de la carpeta actual
        for archivo in archivos:
            # Obtener la ruta completa del archivo
            ruta_completa = os.path.join(carpeta_actual, archivo)
            try:
                with Image.open(ruta_completa):
                    # Imprimir el nombre del archivo y el directorio
                    folder_name = os.path.basename(os.path.dirname(ruta_completa))
                    print(f"Imagen: {archivo} - Directorio: {folder_name}")
                    dato = info_foto(ruta_completa, archivo)
                    if dato != None:
                        procesoCapturaDato(dato)
            except (IOError, OSError):
                # Ignorar archivos que no son imágenes
                pass
    print("Terminado")

def checandoUnaFoto():
    print("Hola")
    rutaE = "Marzo\\BD\\corpus_images\\bored\\227.jpeg"
    imagen = face_recognition.load_image_file(rutaE)
    return info_foto(rutaE,imagen)


def todoEnUnString(matriz):
    if(len(matriz) == 0):
        print("No hay datos que guardar")
        return
    string = ""
    for item in matriz:
        string += str(item)+"|"
    return string

def agregar_linea_a_archivo(archivo, texto):
    # Abrir el archivo en modo de escritura (append)
    with open(archivo, 'a') as archivo_salida:
        # Escribir el texto en una nueva línea
        archivo_salida.write(texto + '\n')

def procesoCapturaDato(matriz):
    texto = todoEnUnString(matriz)
    if texto != None:
        agregar_linea_a_archivo("datos.txt", texto)

print("Inicio")
ruta_de_tu_carpeta = 'Marzo\BD\corpus_images'
sacar_datos_imagen(ruta_de_tu_carpeta)


