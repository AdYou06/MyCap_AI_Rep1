import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

#Getting the Data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#To check the data size
train_images.shape
test_images.shape

#Normalising the data
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32')/255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32')/255

#1 hot encoding done to not prioritise labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#Making CNN - Model

model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))

#Final layer - Dense
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

#Compiling
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Fitting the Model
model.fit(train_images, train_labels, epochs=5, batch_size=32)

#Testing the Model
test_loss, test_acc = model.evaluate(test_images, test_labels)
