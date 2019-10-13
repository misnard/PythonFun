from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt


print("TensorFlow version", tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist

"""Asign train images and train label / test images, test labels from fashion mnist dataset"""
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

print("A training image is composed of", train_images[0].shape,
      "pixels and and example of train label id is", train_labels[0])

clothes_class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0


""" Display a set of test images """
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(clothes_class_names[train_labels[i]])
plt.show()

model = keras.Sequential({
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
})

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=['accuracy'])


model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy : ', test_acc)

predictions = model.predict(test_images)

print("Display first predictions :")
for i in range(10):
    print("Prediction : ", clothes_class_names[np.argmax(predictions[i])],
          "/ Explanation : ", clothes_class_names[test_labels[i]])


imageTest = tf.image.decode_jpeg(tf.io.read_file("./test.jpg"), channels=1)
imageTest = tf.image.resize(imageTest, (28, 28))

print(imageTest.shape)

prediction2 = model.predict(imageTest)

print(prediction2)

print("Final test from no where...", clothes_class_names[np.argmax(prediction2[0])])
