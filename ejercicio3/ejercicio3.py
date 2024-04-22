# Importar las bibliotecas necesarias
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el conjunto de datos
ruta_archivo = 'ObesityDataSet_raw_and_data_sinthetic.csv'
datos = pd.read_csv(ruta_archivo)

# Renombrar la columna
datos.rename(columns={'family_history_with_overweight': 'historial_familiar_sobrepeso'}, inplace=True)

# Mostrar las primeras filas del DataFrame y los tipos de datos de las columnas
print(datos.head())
print(datos.dtypes)

# Inicializar LabelEncoder
codificador_etiquetas = LabelEncoder()

# Copiar los datos para no alterar el DataFrame original
datos_codificados = datos.copy()

# Codificar las columnas categóricas
columnas_categoricas = datos_codificados.select_dtypes(include=['object']).columns
for columna in columnas_categoricas:
    datos_codificados[columna] = codificador_etiquetas.fit_transform(datos_codificados[columna])

# Mostrar las primeras filas del DataFrame codificado y los nuevos tipos de datos
print(datos_codificados.head())
print(datos_codificados.dtypes)

# Inicializar MinMaxScaler
escalador = MinMaxScaler()

# Escalar las columnas numéricas
datos_normalizados = pd.DataFrame(escalador.fit_transform(datos_codificados), columns=datos_codificados.columns)

# Mostrar las primeras filas del DataFrame normalizado
print(datos_normalizados.head())

# Establecer la estética para los gráficos
sns.set(style="whitegrid")

# Crear una figura para los gráficos
fig, ejes = plt.subplots(figsize=(12, 6))

# Gráfico de violín de todas las variables
sns.violinplot(data=datos_normalizados, ax=ejes)
ejes.set_title('Gráfico de Violín de Todas las Variables')

# Rotar las etiquetas del eje x
ejes.set_xticklabels(ejes.get_xticklabels(), rotation=30)

# Ajustar el diseño para prevenir contenido superpuesto
plt.tight_layout()

# Guardar la figura en un archivo
fig.savefig("visualizacion_datos_violin.png")

plt.tight_layout()
plt.show()
