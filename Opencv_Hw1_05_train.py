#%tensorflow_version 2.2
#!pip install tensorflow==2.2
import tensorflow as tf
#print(tf.__version__)
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import  Dense, Flatten, MaxPooling2D, Input
from tensorflow.keras.layers import Conv2D, Dropout, BatchNormalization, Activation
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from keras.models import model_from_json
from keras import backend as K
from keras.callbacks import ModelCheckpoint
#from tensorflow.keras.utils import np_utils
#print(tf.__version__)
nb_classes = 10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_train = y_train.reshape(y_train.shape[0])
y_test = y_test.reshape(y_test.shape[0])
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = to_categorical(y_train, nb_classes)
y_test = to_categorical(y_test, nb_classes)

#model = VGG16(weights='imagenet', include_top=True, input_shape=(48, 48, 3))

model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
input = Input( shape=(32,32,3),name = 'image_input' )
#input = Input( shape=(224,224,3),name = 'image_input' )
output_vgg16_conv = model_vgg16_conv(input)
x = Flatten(name='flatten')(output_vgg16_conv)
x = Dense(units=4096,activation="relu")(x)
x = Dense(units=4096,activation="relu")(x)
x = Dense(10, activation='softmax', name='predictions')(x)

#model = Sequential()
#Create your own model 
model = Model(input, x)
#model.add(Flatten())

sgd = SGD(lr=0.001, momentum=0.2, nesterov=True)
model.compile(optimizer = sgd,loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint("best_vgg16_model.hdf5", monitor='loss', verbose=1,
    save_best_only=True, mode='auto', period=1)

train_history = model.fit(x_train,# train x ( feature )
                          y_train,# train y ( label or target )
                          validation_split = 0.2,# use 20% data to validation 
                          epochs = 20,# run 10 times
                          batch_size = 32,# 128 data/times
                          verbose = 1,    # print process  
                          shuffle = False)
model.summary()
model.save_weights('vgg16_train_weights.h5')
model.save('vgg16_train_model.h5')

print(train_history.history.keys())
# summarize history for accuracy
plt.plot(train_history.history['accuracy'])
plt.plot(train_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left') 
plt.savefig('accuracy.jpg')
plt.show()
# summarize history for loss plt.plot(history.history['loss']) plt.plot(history.history['val_loss']) plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.plot(train_history.history['loss'])
#plt.plot(train_history.history['val_loss'])
plt.legend(['train', 'test'], loc='upper left') 
plt.savefig('loss.jpg')
plt.show()
