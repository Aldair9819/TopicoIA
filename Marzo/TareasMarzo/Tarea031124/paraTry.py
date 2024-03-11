
mi_arreglo = []
segundo_arreglo = [[1,2,3],[4,5,6],[7,8,9]]
try:
    # Tu código que accede al primer elemento del arreglo
    primer_elemento = mi_arreglo[0]
    print(primer_elemento)
    segundo_arreglo.append(primer_elemento)

except IndexError:
    # Manejar la excepción si el índice está fuera de rango
    print("El arreglo está vacío. No hay elementos para acceder.")
except Exception as e:
    # Manejar otras excepciones
    print(f"Se produjo una excepción: {e}")