# Tratamiento de datos


import numpy as np
import pandas as pd
import string
import re
import nltk
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from scipy.spatial.distance import cosine
from unidecode import unidecode

# Gráficos
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns

# Preprocesado y modelado

# Configuración warnings
import warnings
warnings.filterwarnings('ignore')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


# Lectura del archivo

# Ruta del archivo CSV en Google Drive
rt_reviews = "train_new.txt"


# Leer el archivo CSV en un DataFrame
df = pd.read_csv(rt_reviews, encoding='ISO-8859-1', sep="|")
df.columns = ["Lenguage", "Text"]


# Verificar que los datos se hayan cargado correctamente


# Limpiando el conjunto
def remove_posE(tokens):
    # Obtener la lista de stopwords (incluye preposiciones) en inglés
    stopwords = set(nltk.corpus.stopwords.words('english'))

    pronouns = set()
    tagged_words = pos_tag(stopwords)
    for word, tag in tagged_words:
        if tag.startswith('PRP'):
            pronouns.add(word)

    tagged_tokens = pos_tag(tokens)
    tokens_without_pos = [token[0] for token in tagged_tokens if token[0]
                          not in stopwords and token[0] not in pronouns]

    return tokens_without_pos


def remove_posD(tokens):
    # Obtener la lista de stopwords (incluye preposiciones) en inglés
    stopwords = set(nltk.corpus.stopwords.words('dutch'))

    pronouns = set()
    tagged_words = pos_tag(stopwords)
    for word, tag in tagged_words:
        if tag.startswith('PRP'):
            pronouns.add(word)

    tagged_tokens = pos_tag(tokens)
    tokens_without_pos = [token[0] for token in tagged_tokens if token[0]
                          not in stopwords and token[0] not in pronouns]

    return tokens_without_pos


def limpiar_tokenizar(lang, texto):
    # Crear instancia del lematizador
    lemmatizer = WordNetLemmatizer()

    nuevo_texto = texto.lower()
    # Eliminación de signos de puntuación
    regex = '[\\!\\"\\#\\$\\%\\&\\\'\\(\\)\\*\\+\\,\\.\\/\\:\\;\\<\\=\\>\\?\\@\\[\\\\\\]\\^_\\`\\{\\|\\}\\~]'
    nuevo_texto = re.sub(regex, ' ', nuevo_texto)
    # Eliminación de números
    nuevo_texto = re.sub("\d+", ' ', nuevo_texto)
    # Eliminación de espacios en blanco múltiples
    nuevo_texto = re.sub("\\s+", ' ', nuevo_texto)
    # Eliminación de diéresis
    nuevo_texto = unidecode(nuevo_texto)
    # Eliminación de fracciones
    nuevo_texto = re.sub("\d+/\d+", ' ', nuevo_texto)
    # Eliminación de fechas
    nuevo_texto = re.sub("\d{1,2}/\d{1,2}/\d{2,4}", ' ', nuevo_texto)
    # Tokenización por palabras individuales
    nuevo_texto = nuevo_texto.split(sep=' ')
    # Eliminar preposiciones, pronombres y stopWords
    # if (lang == "en"):
    #    nuevo_texto = remove_posE(nuevo_texto)
    # else:
    #    nuevo_texto = remove_posD(nuevo_texto)
    # Lematización
    nuevo_texto = [lemmatizer.lemmatize(token) for token in nuevo_texto]
    # Eliminación de tokens con una longitud < 2
    nuevo_texto = [token for token in nuevo_texto if len(token) > 1]

    return nuevo_texto


df['texto_tokenizado'] = df.apply(
    lambda x: limpiar_tokenizar(x[0], x[1]), axis=1)
df.to_csv(
    'resultadoTokeniz.csv', index=False)


texto_tidy = df.explode(column='texto_tokenizado')
texto_tidy = texto_tidy.drop(columns='Text')
texto_tidy = texto_tidy.rename(columns={'texto_tokenizado': 'token'})
texto_tidy.head(10)
