import keras.activations
from keras.layers import Dense, Flatten, Input
from keras.models import Sequential
from keras.losses import sparse_categorical_crossentropy
from sklearn.preprocessing import Normalizer
import keras.datasets.mnist as kmnist
from Fibonacci import f as fib

(x_train, y_train), (x_test, y_test) = kmnist.load_data()
x_train = x_train / 255
x_test = x_test / 255

model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(units=16, activation="relu"))
model.add(Dense(units=10, activation="softmax"))
model.compile(loss=sparse_categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train)

loss, acc = model.evaluate(x_test, y_test)
print(f'Loss  =   {loss} ')
print(f'accuracy {acc} ')
