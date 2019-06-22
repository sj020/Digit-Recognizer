import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# Immporting the Dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Visulize the dataset
index = 777 # Anysthing upto 60000
print(y_train[index])
plt.imshow(X_train[index], cmap = 'Greys')
plt.show()

# Reshaping the Array
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
input_shape = (28,28,1)

# Making all the values float
X_test = X_test.astype('float32')
X_train = X_train.astype('float32')

# Normalizing the RGB Codes
X_test /= 255
X_train /= 255

# Creating the CNN Architecture
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

model = Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation = tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10, activation = tf.nn.softmax))

# Compiling the Model
model.compile(optimizer = 'adam', loss= 'sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x = X_train, y=y_train, epochs=10)

# Evaluating the Model
model.evaluate(X_test, y_test)

# Trying the Model 
index = 4444
plt.imshow(X_test[index].reshape(28,28), cmap = 'Greys')
plt.show()
pred = model.predict(X_test[index].reshape(1, 28, 28, 1))
print(pred.argmax())