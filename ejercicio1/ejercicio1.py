import pandas as pd
import nltk
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import zipfile


categoria = 'True'
numero_palabras = 50

fichero_entrada = categoria + '.csv'
fichero_entrada_zip = fichero_entrada + '.zip'

# Asegurar que la categoría esté en minúsculas para el uso en nombres de archivo
categoria = categoria.lower()

fichero_salida_grafico = categoria + '_nube_palabras.png'
fichero_salida = categoria + '_palabras_top.csv'
fichero_salida_titulos_grafico = categoria + '_titulos_nube_palabras.png'
fichero_salida_titulos = categoria + '_titulos_palabras_top.csv'


def preprocesar_texto(texto):
    """Elimina todo desde el inicio del texto hasta el primer guión."""
    return texto.split('-', 1)[-1] if '-' in texto else texto

def contar_palabras_no_comunes(datos, solo_titulos = 'NO'):
    """Cuenta las palabras no comunes en las columnas de texto y título."""

    if solo_titulos == 'NO':
        # Preprocesar los textos
        datos['text'] = datos['text'].apply(preprocesar_texto)
        # Concatenar los títulos y textos
        todos_los_textos = datos['title'] + " " + datos['text']
    else:
        # Usar solo los títulos
        todos_los_textos = datos['title']

    # Separar en palabras y convertir a minúsculas
    palabras = todos_los_textos.str.cat(sep=' ').lower().split()

    # Filtrar palabras no comunes
    palabras_filtradas = [palabra for palabra in palabras if palabra not in stopwords.words('english') and palabra.isalpha()]

    # Contar las palabras no comunes
    cuenta_palabras = Counter(palabras_filtradas)

    return cuenta_palabras

def guardar_palabras_top(cuenta_palabras, top_n=50, nombre_archivo='palabras_top.csv'):
    """Guarda las top_n palabras más frecuentes en un archivo CSV."""
    palabras_top = cuenta_palabras.most_common(top_n)
    df_palabras = pd.DataFrame(palabras_top, columns=['Palabra', 'Frecuencia'])
    df_palabras.to_csv(nombre_archivo, index=False)
    print(f"Top {top_n} palabras guardadas en {nombre_archivo}")

def crear_nube_palabras(cuenta_palabras, top_n=50, nombre_archivo='nube_palabras.png'):
    """Crea una nube de palabras a partir de las palabras y sus frecuencias, limitando a top_n palabras más comunes.
    Las palabras se muestran en orientaciones horizontales y verticales."""
    # Reducir el contador a top_n elementos
    palabras_top = dict(cuenta_palabras.most_common(top_n))
    nube = WordCloud(
        width=800, height=400,
        background_color='white',
        prefer_horizontal=0.5  # Ajusta este valor para cambiar la proporción de orientación de las palabras
    ).generate_from_frequencies(palabras_top)
    plt.figure(figsize=(10, 5))
    plt.imshow(nube, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(nombre_archivo)  # Guardar la figura
    plt.show()

def ejecutar():
    """Función principal que ejecuta los pasos necesarios para cargar, procesar, guardar los datos y visualizar resultados."""
    # Acceder al archivo ZIP
    with zipfile.ZipFile(fichero_entrada_zip, 'r') as z:
        with z.open(fichero_entrada) as f:
            datos = pd.read_csv(f, usecols=['title', 'text'])

    # Descargar la lista de palabras comunes (stopwords) si es necesario
    nltk.download('stopwords')

    # Contar palabras no comunes
    cuenta_palabras_no_comunes = contar_palabras_no_comunes(datos,solo_titulos='NO')

    # Guardar el resultado en un fichero CSV
    guardar_palabras_top(cuenta_palabras_no_comunes, top_n=numero_palabras, nombre_archivo=fichero_salida)

    # Crear y mostrar la nube de palabras
    crear_nube_palabras(cuenta_palabras_no_comunes, top_n=numero_palabras, nombre_archivo=fichero_salida_grafico)

    # Contar palabras no comunes en titulos
    cuenta_palabras_no_comunes = contar_palabras_no_comunes(datos,solo_titulos='YES')

    # Guardar el resultado en un fichero CSV
    guardar_palabras_top(cuenta_palabras_no_comunes, top_n=numero_palabras, nombre_archivo=fichero_salida_titulos)

    # Crear y mostrar la nube de palabras
    crear_nube_palabras(cuenta_palabras_no_comunes, top_n=numero_palabras, nombre_archivo=fichero_salida_titulos_grafico)

if __name__ == "__main__":
    ejecutar()
