# Importar Keras, TensorFlow y otras librerías útiles

import tensorflow as tf
from tensorflow import keras

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

# Importar el dataset

imdb_dataset = keras.datasets.imdb

"""
Este dataset es utilizado para la clasificación binaria de reseñas (positivas o negativas) de peliculas.
Consiste en 25.000 reseñas tomadas de IMDB, etiquetadas segun sentimiento (positivo o negativo).
Estas reseñas ya han sido pre procesadas, y cada una codificada como una secuencia de índices (enteros).
Estos indices represan palabras que aparecen en la reseña.
Estas palabras fueron indexadas según su frecuencia en el dataset, es decir el índice 5 representa a la
quinta palabra más frecuente en el dataset. Esto permite un rápido filtro durante opeaciones tales como
considerar solo las 5000 palabras más frecuentes del vocabularios y otros.
"""

# Al cargar el dataset se obtienen 4 arreglos NumPy
"""
x_train, x_test: list of sequences, which are lists of indexes (integers).
    If the num_words argument was specific, the maximum possible index value is num_words-1.
    If the maxlen argument was specified, the largest possible sequence length is maxlen.

y_train, y_test: list of integer labels (1 for positive or 0 for negative).
"""

maximum_index = 20000

(train_data, train_labels), (test_data, test_labels) = imdb_dataset.load_data(num_words=maximum_index)


print("TRAIN DATA:", train_data[0], "....\n")
print("TRAIN LABELS:", train_labels)

# Si quisieramos cambiar la proporción de training data vs test data podemos concatenar los datasets y volver
# a dividirlos

all_data = np.concatenate((train_data, test_data), axis=0)
all_labels = np.concatenate((train_labels, test_labels), axis=0)

# Ejemplo de como se ve una de las reviews
indice_ejemplo = 0

index = imdb_dataset.get_word_index()
reverse_index = dict([(value, key) for (key, value) in index.items()])
decoded = " ".join( [reverse_index.get(i - 3, "#") for i in test_data[indice_ejemplo]] )
print("Review: ", decoded, "\n")
print("Label: ", test_labels[indice_ejemplo])

# Normalización del training y test set.

def normalizar(secuencias, dimension = maximum_index):
    #relleno con 0 cuando la dimension es menor
    normalizado = np.zeros((len(secuencias), dimension))
    for i, secuencia in enumerate(secuencias):
        normalizado[i, secuencia] = 1
    return normalizado

normalized_data = normalizar(all_data)
# Convierto los labels a floats
float_labels = np.array(all_labels).astype("float32")

# Separo el dataset en los casos utilizados para entrenamiento y para pruebas.
factor_entrenamiento = 80 #porcentaje para entrenamiento, el resto para testing
cantidad_casos = len(normalized_data)
cantidad_entrenamiento = (int)((cantidad_casos * factor_entrenamiento) / 100)

test_data = normalized_data[cantidad_entrenamiento:]
test_labels = float_labels[cantidad_entrenamiento:]
train_data = normalized_data[:cantidad_entrenamiento]
train_labels = float_labels[:cantidad_entrenamiento]

model = keras.models.Sequential()
# Input - Layer
model.add(keras.layers.Dense(50, activation = "relu", input_shape=(maximum_index, )))
# Hidden - Layers
model.add(keras.layers.Dropout(0.3, noise_shape=None, seed=None))
# Dropout selecciona neuronas al azar para ignorar durante el entrenamiento, previene overfitting
model.add(keras.layers.Dense(50, activation = "relu"))
model.add(keras.layers.Dropout(0.2, noise_shape=None, seed=None))
model.add(keras.layers.Dense(50, activation = "relu"))
# Output- Layer
model.add(keras.layers.Dense(1, activation = "sigmoid"))
model.summary()

# Compila el modelo, utiliza Adam como optimizador
model.compile(
 optimizer = "adam",
 loss = "binary_crossentropy",
 metrics = ["accuracy"]
)

results = model.fit(
 train_data, train_labels,
 epochs = 4,
 batch_size = 500,
 validation_data = (test_data, test_labels)
)

print("Precisión del modelo:" , np.mean(results.history["val_accuracy"]))

# Ejemplo tomo una review
review_index = 0
decoded = " ".join( [reverse_index.get(i - 3, "#") for i in all_data[review_index]] )
print("Reseña: ", decoded, "\n")
print("Esperado : ", all_labels[review_index])

### Ejemplo de predicción con modelo entrenado ###

prediccion = model.predict(test_data)

print(prediccion)
print("Prediccion:" , prediccion[review_index])

# Obtener el caracter con mayores posibilidades de matchear

#print("Índice dentro del array del caracter con mayores probabilidades de matchear:", np.argmax(predictions[0]))

# Examinar el valor asociado a caracter

#print("Dígito predecido:",test_labels[7])
