import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import Session
from tensorflow.keras import Input,Model
from tensorflow.keras.layers import ZeroPadding2D,Conv2D,BatchNormalization,Activation,MaxPooling2D,AveragePooling2D,Flatten,Dense,Add
from tensorflow.keras.regularizers import l2
from time import process_time
from datetime import datetime
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import os
# Check that GPU is available: cf. https://colab.research.google.com/notebooks/gpu.ipynb
# assert (tf.test.is_gpu_available())
assert (tf.test.is_built_with_cuda())
tf.keras.backend.clear_session()
tf.config.optimizer.set_jit(True)

# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# for physical_device in physical_devices:
#     tf.config.experimental.set_memory_growth(physical_device, True)
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
tf.compat.v1.keras.backend.set_session(Session(config=config))
dtype='float16'
tf.keras.backend.set_floatx(dtype)
tf.keras.backend.set_epsilon(1e-3)
def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float16') / 256
    x_test = x_test.astype('float16') / 256

    # Convert class vectors to binary class matrices.
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
    return ((x_train, y_train), (x_test, y_test))


(x_train, y_train), (x_test, y_test) = load_data()


def generate_model():
    input_im = Input(shape=x_train.shape[1:]) # cifar 10 images size
    x = ZeroPadding2D(padding=(3, 3))(input_im)
    # 1st stage  # here we perform maxpooling, see the figure above
    x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    #2nd stage   # frm here on only conv block and identity block, no pooling
    x = res_conv(x, s=1, filters=(64, 256))
    x = res_identity(x, filters=(64, 256))
    x = res_identity(x, filters=(64, 256))
    # 3rd stage
    x = res_conv(x, s=2, filters=(128, 512))
    x = res_identity(x, filters=(128, 512))
    x = res_identity(x, filters=(128, 512))
    x = res_identity(x, filters=(128, 512))
    # 4th stage
    x = res_conv(x, s=2, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))
    # 5th stage
    x = res_conv(x, s=2, filters=(512, 2048))
    x = res_identity(x, filters=(512, 2048))
    x = res_identity(x, filters=(512, 2048))
    # ends with average pooling and dense connection
    x = AveragePooling2D((2, 2), padding='same')(x)
    x = Flatten()(x)
    x = Dense(10, activation='softmax', kernel_initializer='he_normal')(x) #multi-class
    print(x.dtype.name)
    # define the model len(class_types)
    model = Model(inputs=input_im, outputs=x, name='Resnet50')
    return model

def res_identity(x, filters):
    #renet block where dimension doesnot change.
    #The skip connection is just simple identity conncection
    #we will have 3 blocks and then input will be added
    x_skip = x # this will be used for addition with the residual block
    f1, f2 = filters
    #first block
    x = Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #second block # bottleneck (but size kept same with padding)
    x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # third block activation used after adding the input
    x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    # add the input
    x = Add()([x, x_skip])
    x = Activation('relu')(x)
    return x
def res_conv(x, s, filters):
    '''
    here the input size changes'''
    x_skip = x
    f1, f2 = filters
    # first block
    x = Conv2D(f1, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=l2(0.001))(x)
    # when s = 2 then it is like downsizing the feature map
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # second block
    x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #third block
    x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    # shortcut
    x_skip = Conv2D(f2, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=l2(0.001))(x_skip)
    x_skip = BatchNormalization()(x_skip)
    # add
    x = Add()([x, x_skip])
    x = Activation('relu')(x)
    return x

# def generate_model():
#     return tf.keras.models.Sequential([
#         tf.keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]),
#         tf.keras.layers.Activation('relu'),
#         tf.keras.layers.Conv2D(32, (3, 3)),
#         tf.keras.layers.Activation('relu'),
#         tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
#         tf.keras.layers.Dropout(0.25),

#         tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
#         tf.keras.layers.Activation('relu'),
#         tf.keras.layers.Conv2D(64, (3, 3)),
#         tf.keras.layers.Activation('relu'),
#         tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
#         tf.keras.layers.Dropout(0.25),

#         tf.keras.layers.Flatten(),
#         tf.keras.layers.Dense(512),
#         tf.keras.layers.Activation('relu'),
#         tf.keras.layers.Dropout(0.5),
#         tf.keras.layers.Dense(10),
#         tf.keras.layers.Activation('softmax')
#     ])
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)
print('Compute dtype: %s' % policy.compute_dtype)
print('Variable dtype: %s' % policy.variable_dtype)

model = generate_model()


def compile_model(model):
    opt = tf.keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model


model = compile_model(model)


def train_model(model, x_train, y_train, x_test, y_test, epochs=4):
    logs = "/home/mmadala/tmp/scripts/logs/"+ os.environ["XLA_Current_Disabled"] + datetime.now().strftime("%Y%m%d")
    tboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs,
                                                     histogram_freq=1,
                                                     profile_batch='500,510')

    model.fit(x_train, y_train, batch_size=256, epochs=epochs, validation_data=(x_test, y_test), shuffle=True ,callbacks = [tboard_callback])


def warmup(model, x_train, y_train, x_test, y_test):
    # Warm up the JIT, we do not wish to measure the compilation time.
    initial_weights = model.get_weights()
    train_model(model, x_train, y_train, x_test, y_test, epochs=1)
    model.set_weights(initial_weights)


warmup(model, x_train, y_train, x_test, y_test)
noXLA_start=process_time()
train_model(model, x_train, y_train, x_test, y_test)
noXLA_end=process_time()
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
print("Elapsed time:", noXLA_start, noXLA_end)
print("Elapsed time to train with XLA ",noXLA_end - noXLA_start)
# We need to clear the session to enable JIT in the middle of the program.


# tf.keras.backend.clear_session()
# tf.config.optimizer.set_jit(True) # Enable XLA.
# model = compile_model(generate_model())
# (x_train, y_train), (x_test, y_test) = load_data()

# warmup(model, x_train, y_train, x_test, y_test)
# XLA_start=process_time()
# train_model(model, x_train, y_train, x_test, y_test)
# XLA_end=process_time()
# print("Elapsed time:", XLA_start, XLA_end)
# print("Elapsed time to train with XLA ",XLA_end - XLA_start)