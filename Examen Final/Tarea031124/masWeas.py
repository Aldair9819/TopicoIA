import numpy as np

# Tu matriz
matriz = np.array([[[165, 165, 163], [165, 165, 163], [165, 165, 163], [171, 170, 168], [170, 169, 167], [169, 168, 166]],
                   [[165, 165, 163], [165, 165, 163], [165, 165, 163], [171, 170, 168], [170, 169, 167], [169, 168, 166]],
                   [[165, 165, 163], [165, 165, 163], [165, 165, 163], [171, 170, 168], [170, 169, 167], [170, 169, 167]],
                   [[138, 139, 134], [138, 139, 134], [137, 138, 133], [151, 151, 149], [151, 151, 149], [151, 151, 149]],
                   [[138, 139, 134], [138, 139, 134], [137, 138, 133], [151, 151, 149], [150, 152, 149], [150, 152, 149]],
                   [[138, 139, 134], [138, 139, 134], [137, 138, 133], [151, 151, 149], [150, 152, 149], [150, 152, 149]]])

# Convertir la matriz a una cadena de texto
cadena_texto = str(matriz.tolist())

print(cadena_texto)
