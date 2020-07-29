# Importar Keras, TensorFlow y otras librerías útiles
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Importar el dataset
mnist = keras.datasets.mnist

# Carga del dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Construcción del modelo
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])

# Compilación del modelo
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Entrenamiento y evaluación del modelo
results = model.fit(train_images, train_labels, epochs=10)
loss, accuracy = model.evaluate(train_images, train_labels)

### Ejemplo 1 de predicción con modelo entrenado ###
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)

# Imprimir array de posibilidades de matcheo para cada caracter.
print("Probabilidades para cada dígito:", predictions[0])

# Obtener el caracter con mayores posibilidades de matchear
print("Índice dentro del array del caracter con mayores probabilidades de matchear:", np.argmax(predictions[0]))

# Examinar el valor asociado a caracter
print("Dígito predecido:", test_labels[7])
