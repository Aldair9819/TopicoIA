import face_recognition
from PIL import ImageDraw
import PIL.Image as Image
import os
from PIL import Image
from io import BytesIO


def leer_primera_linea(nombre_archivo):
    try:
        with open(nombre_archivo, 'r') as archivo:
            primera_linea = archivo.readline().strip()
            return primera_linea
    except FileNotFoundError:
        return "El archivo no fue encontrado."
    except Exception as e:
        return f"Ocurrió un error: {str(e)}"

def imprimir_imagen_desde_string(imagen_string):
    try:
        # Convierte el string en bytes
        imagen_bytes = bytes(imagen_string, 'utf-8')

        # Crea un objeto BytesIO para manejar los bytes como un archivo
        imagen_io = BytesIO(imagen_bytes)

        # Carga la imagen desde BytesIO utilizando face_recognition
        imagen = face_recognition.load_image_file(imagen_io)

        # Crea un objeto de la clase Image de PIL
        imagen_pil = Image.fromarray(imagen)

        # Muestra la imagen
        imagen_pil.show()
    except Exception as e:
        print(f"Ocurrió un error: {str(e)}")

def imprimirCara(imagen):
    pil_image = Image.fromarray(imagen)
    pil_image.show()

# Ejemplo de uso:
nombre_archivo = 'datos.txt'
primera_linea = leer_primera_linea(nombre_archivo)
separar = primera_linea.split("|")

print(f"Primera línea del archivo: {separar[1]}")
imprimirCara(separar[1])
