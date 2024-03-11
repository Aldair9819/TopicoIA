import pandas as pd

# Crear una matriz de ejemplo
data = {'Nombre': ['Juan', 'María', 'Pedro'],
        'Edad': [25, 30, 35],
        'Ciudad': ['Madrid', 'Barcelona', 'Valencia']}

# Convertir la matriz en un DataFrame de pandas
df = pd.DataFrame(data)

# Especificar el nombre del archivo Excel
nombre_archivo = 'ejemplo_excel.xlsx'

# Guardar el DataFrame en un archivo Excel
df.to_excel(nombre_archivo, index=False)

print(f'DataFrame guardado en {nombre_archivo}')
