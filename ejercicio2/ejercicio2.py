import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Carga los datos
file_path = 'ObesityDataSet_raw_and_data_sinthetic.csv'  
data = pd.read_csv(file_path)


# convierte 'NObeyesdad' a numérico usando Label Encoding
label_encoder = LabelEncoder()
data['NObeyesdad'] = label_encoder.fit_transform(data['NObeyesdad'])

# Ajustamos el codificador a los datos y transformamos la columna 'Gender'
data['Gender'] = label_encoder.fit_transform(data['Gender'])

# Mapeo inverso para referencia
nobeyesdad_mapping = {index: label for index, label in enumerate(label_encoder.classes_)}

# Lista de variables seleccionadas para los pairplots
selected_variables = ['Gender','Age', 'Height', 'Weight', 'CH2O', 'FAF']

# Filtrar el conjunto de datos para incluir solo las filas donde 'NObeyesdad' tenga valores de 0 a 6
data_filtered = data[data['NObeyesdad'].isin(range(7))]

# Crear pairplots para cada valor de 'NObeyesdad' (de 0 a 6)
for value in range(7):
    subset = data_filtered[data_filtered['NObeyesdad'] == value]
    if not subset.empty:
        # Seleccionar solo las variables de interés para el pairplot
        subset_selected_variables = subset[selected_variables]
        
        # Crear el pairplot sin 'NObeyesdad' como una de las variables de los ejes
        pairplot = sns.pairplot(subset_selected_variables, plot_kws={'alpha': 0.5})
        pairplot.figure.suptitle(f'Pairplot para NObeyesdad = {value}', y=1.02)
        plt.show()
        # Descomenta la siguiente línea si quieres guardar la imagen
        pairplot.savefig(f'pairplot_nobeyesdad_{value}.png')
        plt.close()

    else:
        print(f'No hay datos para NObeyesdad = {value}')

print('Pairplots creados con éxito.')

# Crea el pairplot para el conjunto de datos completo
pairplot = sns.pairplot(data[selected_variables], plot_kws={'alpha': 0.5})
pairplot.fig.suptitle('Pairplot para todo el conjunto de datos', y=1.02)

# Muestra el gráfico
plt.show()
pairplot.savefig(f'pairplot_completo.png')
plt.close()


