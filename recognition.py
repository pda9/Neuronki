import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from PIL import Image
from keras.models import load_model
from keras_preprocessing import image
import numpy as np

classes = ['футболка', 'брюки', 'свитер', 'платье', 'пальто', 'туфли', 'рубашка', 'кроссовки', 'сумка', 'ботинки']
# Загрузка сети из файла
model=load_model('fashion_mnist_dense.h5')
model.summary()
# Загрузка изображение
image_path = 'image/bag.jpg'
# Изменение размера изображения
new_image = Image.open(image_path).resize((150, 150))
# Сохранение изменённого изображения
new_image.save('image/bag_resize.jpg')
# Загрузка изменённого изображения
img_path =('image/bag_resize.jpg')
img = image.load_img(img_path, target_size=(28, 28), color_mode = "grayscale")
# Преобразуем картинку в массив
x = image.img_to_array(img)
# Меняем форму массива в плоский вектор
x = x.reshape(1, 784)
# Инвертируем изображение
x = 255 - x
# Нормализуем изображение
x /= 255

prediction = model.predict(x)
prediction
prediction = np.argmax(prediction)
print("Номер класса:", prediction)
print("Название класса:", classes[prediction])