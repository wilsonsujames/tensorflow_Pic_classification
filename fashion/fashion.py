import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data =keras.datasets.fashion_mnist

(train_images, train_labels),(test_images, test_labels)=data.load_data()

class_names =["T-shirt/top","Trouser","Pullover","Dress","Coat",
"Sandal","Shirt","Sneaker","Bag","Ankle boot"]

print("第八張圖 numpy arry:")
print(train_images[7])
print("第八張圖答案:")
print(train_labels[7])

plt.imshow(train_images[7])
plt.show()

#標準化，使計算複雜度降低
train_images = train_images/255.0

test_images = test_images/255.0


#神經網路搭建
model = keras.Sequential([
keras.layers.Flatten(input_shape=(28,28)),
keras.layers.Dense(128,activation='relu'),
keras.layers.Dense(64,activation='relu'),
keras.layers.Dense(32,activation='relu'),
keras.layers.Dense(10,activation="softmax")
])


model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])

model.fit(train_images, train_labels,epochs=5)


model.evaluate(test_images, test_labels,verbose=2)


predictions = model.predict(test_images)
print(class_names[np.argmax(predictions[0])])