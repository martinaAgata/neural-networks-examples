import tensorflow as tf
import numpy
from tensorflow import keras
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation  
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import utils 
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10 

# Set random seed for purposes of reproducibility
seed = 21
chosenActivation = 'relu'
chosenOptimizer = 'Adam'

# load train and test dataset
def load_dataset():
	# load dataset
	(X_train, y_train), (X_test, y_test) = cifar10.load_data()
	# one hot encode target values
	y_train = to_categorical(y_train)
	y_test = to_categorical(y_test)
	return X_train, y_train, X_test, y_test
 
def scale_pixels(train, test):
	# convert from integers to floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# return normalized images
	return train_norm, test_norm

# Defino modelo CNN
def define_model(X_train, class_num):
	model = Sequential()
	#relues es la activación más común, y padding='same' solo significa que no estamos cambiando el tamaño de la imagen
	model.add(Conv2D(32, (3, 3), input_shape=X_train.shape[1:], activation=chosenActivation, padding='same'))   
	
	#model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), activation=chosenActivation, padding='same'))
	#capa de exclusión para evitar el sobreajuste, que funciona al eliminar al azar algunas de las conexiones entre las capas (0.2 significa que elimina el 20% de las conexiones existentes)
	model.add(Dropout(0.2))
	#normalización de lotes: normaliza las entradas que se dirigen a la siguiente capa, asegurando que la red siempre cree activaciones con la misma distribución que deseamos
	model.add(BatchNormalization())
	#otra capa convolucional, pero el tamaño del filtro aumenta para que la red pueda aprender representaciones más complejas
	model.add(Conv2D(64, (3, 3), activation=chosenActivation, padding='same'))

	# capa de agrupación: ayuda a hacer que el clasificador de imágenes sea más robusto para que pueda aprender patrones relevantes. 
	#También está la deserción y la normalización de lotes
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(BatchNormalization())

	#repetimos estas capas para darle a su red más representaciones para trabajar
	model.add(Conv2D(128, (3, 3), padding='same'))
	model.add(Activation(chosenActivation))
	model.add(Dropout(0.2))
	model.add(BatchNormalization())

	#Flattenlos datos + capa de abandono
	model.add(Flatten())
	model.add(Dropout(0.2))

	#creamos la primera capa densamente conectada
	#Necesitamos especificar el número de neuronas en la capa densa. 
	#El número de neuronas en las capas siguientes disminuye, acercándose finalmente al mismo número de neuronas que hay clases en el conjunto de datos (en este caso 10). 
	#La restricción del kernel puede regularizar los datos a medida que se aprende, otra cosa que ayuda a evitar el sobreajuste
	model.add(Dense(256, kernel_constraint=max_norm(3)))
	model.add(Activation(chosenActivation))
	model.add(Dropout(0.2))
	model.add(BatchNormalization())

	model.add(Dense(128, kernel_constraint=max_norm(3)))
	model.add(Activation(chosenActivation))
	model.add(Dropout(0.2))
	model.add(BatchNormalization())

	#softmaxfunción de activación selecciona la neurona con la mayor probabilidad de salida, votando que la imagen pertenece a esa clase
	model.add(Dense(class_num))
	model.add(Activation('softmax'))

	#optimizador
	optimizer = chosenOptimizer

	#compilador + metrica
	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	
	print(model.summary())
	
	## OTRA OPCION
	#model = Sequential()
	#model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
	#model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	#model.add(MaxPooling2D((2, 2)))
	#model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	#model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	#model.add(MaxPooling2D((2, 2)))
	#model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	#model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	#model.add(MaxPooling2D((2, 2)))
	#model.add(Flatten())
	#model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	#model.add(Dense(10, activation='softmax'))
	# compile model
	#opt = SGD(lr=0.001, momentum=0.9)
	#model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	
	return model
 
# run training model
def run():
	# load dataset
	X_train, y_train, X_test, y_test = load_dataset()
	# prepare data
	X_train, X_test = scale_pixels(X_train, X_test)
	#define model
	class_num = y_test.shape[1]
	model = define_model(X_train, class_num)
	# fit model with epocs
	epochs = 25
	numpy.random.seed(seed)
	model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=64, verbose=0)
	# save model
	model.save('model.h5')
 
# entry point, run the test harness
run()
