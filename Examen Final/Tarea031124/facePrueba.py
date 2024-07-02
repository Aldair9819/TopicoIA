import face_recognition
import matplotlib.pyplot as plt
from PIL import Image

def mostrar_cara_recortada(imagen_path):
    # Cargar la imagen con face_recognition
    imagen = face_recognition.load_image_file(imagen_path)

    # Encontrar todas las caras en la imagen
    caras = face_recognition.face_locations(imagen)

    # Verificar si se encontraron caras
    if not caras:
        print("No se encontraron caras en la imagen.")
        return

    # Tomar la primera cara (puedes ajustar esto según tus necesidades)
    cara = caras[0]

    # Obtener las coordenadas de la cara
    top, right, bottom, left = cara

    # Abrir la imagen con Pillow (PIL)
    imagen_pil = Image.open(imagen_path)

    # Recortar la cara de la imagen original
    cara_recortada = imagen_pil.crop((left, top, right, bottom))

    # Mostrar la cara recortada
    plt.imshow(cara_recortada)
    plt.axis('off')
    plt.show()

# Ruta de la imagen de entrada
imagen_path = "Marzo\image.png"

# Llamar a la función para mostrar la cara recortada
mostrar_cara_recortada(imagen_path)
