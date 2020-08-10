import tensorflow as tf
from datetime import datetime
from time import process_time
from tensorflow.keras import datasets, layers, models
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import Session

# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# for physical_device in physical_devices:
#     tf.config.experimental.set_memory_growth(physical_device, True)

assert (tf.test.is_built_with_cuda())
tf.keras.backend.clear_session()
tf.config.optimizer.set_jit(False)
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
tf.compat.v1.keras.backend.set_session(Session(config=config))

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.summary()
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
model.summary()
# a = datetime.now()
noXLA_start=process_time()
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10,batch_size=256, shuffle=True,
                    validation_data=(test_images, test_labels))
# b = datetime.now()
noXLA_end=process_time()
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(test_acc)
print("Elapsed time:", noXLA_start, noXLA_end)
print("Elapsed time to train without XLA ",noXLA_end - noXLA_start)
tf.keras.backend.clear_session()
tf.config.optimizer.set_jit(True)

print("Using XLA")
# c = datetime.now()
XLA_start=process_time()
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10,batch_size=256, shuffle=True,
                    validation_data=(test_images, test_labels))
# d = datetime.now()
XLA_end=process_time()
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(test_acc)
print("Elapsed time:", XLA_start, XLA_end)
print("Elapsed time to train with XLA ",XLA_end - XLA_start)
# print((b-a).seconds)
# print((d-c).seconds)
