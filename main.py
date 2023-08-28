import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

# Устанавливаем seed для повторяемости результатов
np.random.seed(42)

# Загружаем данные
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Преобразование размерности изображений
x_train = x_train.reshape(60000,784)
x_test = x_test.reshape(10000, 784)
# Нормализация данных
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Преобразуем метки в категории
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)
classes = ['футболка', 'брюки', 'свитер', 'платье', 'пальто', 'туфли', 'рубашка', 'кроссовки', 'сумка', 'ботинки']
model = Sequential()
model.add(Dense(800, input_dim=784, activation="relu"))
model.add(Dense(10, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])

print(model.summary())

model.fit(x_train,y_train,batch_size=200,epochs=100,verbose=1)
score = model.evaluate(x_test,y_test,verbose=1)
print("Точность работы на тестовых данных: %.2f%%" % (score[1]*100))
predictions = model.predict(x_train)
print(predictions[0])
print(np.argmax(predictions[0]))
print(np.argmax(y_train[0]))