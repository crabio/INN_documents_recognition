import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])

print(x_train[0])

# Creating a Sequential Model and adding the layers
model = Sequential()
model.add(Conv2D(28, kernel_size=(3, 3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())  # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10, activation=tf.nn.softmax))

model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

checkpoint_path = "model/cp.ckpt"

# Create checkpoint callback
# cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
#                                                  save_weights_only=True,
#                                                  verbose=1)
#
# model.fit(x=x_train, y=y_train, epochs=20, validation_data=(x_test, y_test),
#           callbacks=[cp_callback])
#
# print('Trained model test')
# print(model.evaluate(x_test, y_test))

model.load_weights(checkpoint_path)
print('Restored model test')
print(model.evaluate(x_test, y_test))


def predict_image(img_path):
    image = cv2.imread(img_path)
    image_gray = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    image_gray_resized = cv2.bitwise_not(cv2.resize(image_gray, (28, 28), interpolation=cv2.INTER_AREA))
    image_gray_delated = cv2.dilate(image_gray_resized, (3, 3))

    data_float = image_gray_delated.astype('float32')
    data_float /= 255

    pred = model.predict(data_float.reshape(1, 28, 28, 1))

    print(pred.round(decimals=4)[0][pred.argmax()], pred.argmax())

    plt.imshow(image_gray_resized.reshape(28, 28), cmap='Greys')
    plt.show()

images_file_names_list = [predict_image('templates/test/' + file) for file in os.listdir('templates/test') if os.path.isfile('templates/test/' + file)]

for images_file_names in images_file_names_list:
    print(images_file_names)
    predict_image('templates/test/' + images_file_names)
