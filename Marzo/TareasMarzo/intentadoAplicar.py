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
    


def imprimirTodo(imagen,cara):
    imprimirCara(imagen)
    pil_image = Image.fromarray(recortarCara(imagen,cara))
    pil_image.show()

def hasFace(face_locations):
    if len(face_locations) > 0:
        return True
    else:
        return False

def info_foto(ruta, imagen):
    #ruta = "Marzo\image.png"
    imagen = face_recognition.load_image_file(ruta)
    face_locations = face_recognition.face_locations(imagen)
    if not hasFace(face_locations):
        return
    cara = face_locations[0]
    top, right, bottom, left = cara
    cara_recortada = imagen[top:bottom, left:right]
    face_landmarks_list = face_recognition.face_landmarks(imagen)
    caracteristicas = []
    folder_name = os.path.basename(os.path.dirname(ruta))
    #print("La imagen original es: ", imagen)
    #imprimirCara(imagen)
    for clave,valor in face_landmarks_list[0].items():
        caracteristicas.append(valor)
    #print("Estas son las caracteristicas de la cara:")
    #print(caracteristicas)
    #print("Esta es la cara recortada:")
    #print(cara)
        

    
    #print("La ruta es: ",folder_name)
    #pil_image = Image.fromarray(recortarCara(imagen,cara))
    #pil_image.show()
    #imprimirTodo(imagen,cara)
    imagen_aplanada = imagen.flatten()
    print("La imagen aplanada es: ", imagen_aplanada)
    return [imagen_aplanada, caracteristicas,cara, folder_name]

import os
from PIL import Image

def sacar_datos_imagen(carpeta_raiz):
    matriz_principal = []
    for carpeta_actual, subcarpetas, archivos in os.walk(carpeta_raiz):
        for archivo in archivos:
            ruta_completa = os.path.join(carpeta_actual, archivo)
            try:
                with Image.open(ruta_completa):
                    # Imprimir el nombre del archivo y el directorio
                    folder_name = os.path.basename(os.path.dirname(ruta_completa))
                    print(f"Imagen: {archivo} - Directorio: {folder_name}")
                    dato = info_foto(ruta_completa, archivo)
                    if dato != None:
                        #matriz_principal.append(dato)
                        procesoCapturaDato(dato)
            except (IOError, OSError):
                # Ignorar archivos que no son imágenes
                pass
    print("Terminado")
    return matriz_principal

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

# Reemplaza 'ruta_de_tu_carpeta' con la ruta de la carpeta que deseas explorar
ruta_de_tu_carpeta = 'Marzo\BD\corpus_images'

#matriz = checandoUnaFoto()
#procesoCapturaDato(matriz)


#guardarDatos(matriz)


sacar_datos_imagen(ruta_de_tu_carpeta)
print("Estos son los datos finales:")
#print(datos_total)


