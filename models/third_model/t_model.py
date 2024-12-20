# Датасет - МРТ

import tensorflow as tf
import numpy as np
import pathlib
import matplotlib.pyplot as plt

from keras import utils, layers, losses, models
from keras.src.models import Sequential

data_dir = pathlib.Path("../second_model/mrt")

batch_size = 32
img_height = 180
img_width = 180

train_ds = utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

model_mrt_aug = Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)

plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
  for i in range(9):
    augmented_images = model_mrt_aug(images)
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_images[0].numpy().astype("uint8"))
    plt.axis("off")
plt.savefig('plt_data_mrt_aug.jpg')

model_mrt = Sequential([
  model_mrt_aug,
  layers.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(4, activation='softmax')
])

model_mrt.compile(optimizer='adam',
                  loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

epochs = 15
history = model_mrt.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

test_loss, test_accuracy = model_mrt.evaluate(val_ds)
print("Доля верных ответов на тестовых данных, в процентах:", round(test_accuracy * 100, 4))

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Точность train')
plt.plot(epochs_range, val_acc, label='Точность val')
plt.legend(loc='lower right')
plt.title('Точность train и val')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Потери train')
plt.plot(epochs_range, val_loss, label='Потери val')
plt.legend(loc='upper right')
plt.title('Потери train и val')

plt.savefig('plt_losses_mrt_aug.jpg')

model_mrt.save("mrt_aug.h5")

model = models.load_model('mrt_aug.h5')

img = utils.load_img(
    "1.jpg", target_size=(img_height, img_width)
)
img_array = utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
print(predictions)
score = tf.nn.softmax(predictions[0])
print(score)
print(
    f"Это изображение похоже на {class_names[np.argmax(score)]} с вероятностью {100 * np.max(score)} процентов."
    )

img = utils.load_img(
    "2.jpg", target_size=(img_height, img_width)
)
img_array = utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
print(predictions)
score = tf.nn.softmax(predictions[0])
print(score)
print(
    f"Это изображение похоже на {class_names[np.argmax(score)]} с вероятностью {100 * np.max(score)} процентов."
    )

img = utils.load_img(
    "3.jpg", target_size=(img_height, img_width)
)
img_array = utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
print(predictions)
score = tf.nn.softmax(predictions[0])
print(score)
print(
    f"Это изображение похоже на {class_names[np.argmax(score)]} с вероятностью {100 * np.max(score)} процентов."
    )

img = utils.load_img(
    "4.jpg", target_size=(img_height, img_width)
)
img_array = utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
print(predictions)
score = tf.nn.softmax(predictions[0])
print(score)
print(
    f"Это изображение похоже на {class_names[np.argmax(score)]} с вероятностью {100 * np.max(score)} процентов."
    )