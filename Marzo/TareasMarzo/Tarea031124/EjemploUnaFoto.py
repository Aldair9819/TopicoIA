import face_recognition
from PIL import ImageDraw
import PIL.Image as Image
import os

def recortarCara(imagen, cara):
    top, right, bottom, left = cara
    cara_recortada = imagen[top:bottom, left:right]
    return cara_recortada

def imprimirCara(imagen):
    pil_image = Image.fromarray(imagen)
    pil_image.show()

ruta = "Marzo\image.png"
imagen = face_recognition.load_image_file(ruta)
face_locations = face_recognition.face_locations(imagen)
cara = face_locations[0]
top, right, bottom, left = cara
cara_recortada = imagen[top:bottom, left:right]

face_landmarks_list = face_recognition.face_landmarks(imagen)

caracteristicas = []
print("La imagen original es: ", imagen)
imprimirCara(imagen)


for clave,valor in face_landmarks_list[0].items():
    caracteristicas.append(valor)
print("Estas son las caracteristicas de la cara:")
print(caracteristicas)
print("Esta es la cara recortada:")
print(cara)

folder_name = os.path.basename(os.path.dirname(ruta))
print("La ruta es: ",folder_name)
pil_image = Image.fromarray(recortarCara(imagen,cara))
pil_image.show()
