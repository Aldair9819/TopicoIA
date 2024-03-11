import os
from PIL import Image

def encontrar_imagenes(carpeta_raiz):
    for carpeta_actual, subcarpetas, archivos in os.walk(carpeta_raiz):
        for archivo in archivos:
            ruta_completa = os.path.join(carpeta_actual, archivo)
            try:
                with Image.open(ruta_completa):
                    # Imprimir el nombre del archivo y el directorio
                    folder_name = os.path.basename(os.path.dirname(ruta_completa))
                    print(f"Imagen: {archivo} - Directorio: {folder_name}")
            except (IOError, OSError):
                # Ignorar archivos que no son imágenes
                pass

# Reemplaza 'ruta_de_tu_carpeta' con la ruta de la carpeta que deseas explorar
ruta_de_tu_carpeta = 'Marzo\BD\corpus_images'
encontrar_imagenes(ruta_de_tu_carpeta)
