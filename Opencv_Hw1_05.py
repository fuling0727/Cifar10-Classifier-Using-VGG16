
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import  Dense, Flatten, MaxPooling2D
from tensorflow.keras.layers import Conv2D, Dropout, BatchNormalization, Activation
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model, model_from_json
from keras import utils
import matplotlib.pyplot as plt
import cv2
import sys
import numpy as np
import random
from PyQt5.QtGui import QPixmap, QIcon, QFont, QImage
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel, QPushButton, QWidget, QLineEdit
from PyQt5 import QtCore

model = load_model('vgg16_train_model.h5')
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
'''y_train = y_train.reshape(y_train.shape[0])
y_test = y_test.reshape(y_test.shape[0])
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
nb_classes = 10
y_train = utils.np_utils.to_categorical(y_train, nb_classes)'''

class MainWindow(QMainWindow):
    def __init__(self):
        
        super(MainWindow, self).__init__()
        self.resize(500,700) # smart phone size
        self.title = "2020 Opencvdl HW1_05"
        self.setWindowTitle(self.title)

        button1 = QPushButton(self)
        button1.setText("5.1 Show Image")  # 建立名字
        button1.setStyleSheet("background-color:#FCFCFC; ")
        button1.setFixedHeight(35)
        button1.setFixedWidth(150)
        button1.move(170,120)  # 移動位置
        button1.clicked.connect(self.buttonClicked1) # 設置button啟動function

        button2 = QPushButton(self)
        button2.setText("5.1 Show Parameters")  # 建立名字
        button2.setStyleSheet("background-color:#FCFCFC; ")
        button2.setFixedHeight(35)
        button2.setFixedWidth(150)
        button2.move(170,170)  # 移動位置
        button2.clicked.connect(self.buttonClicked2)

        button3 = QPushButton(self)
        button3.setText("Model Structure")  # 建立名字
        button3.setStyleSheet("background-color:#FCFCFC; ")
        button3.setFixedHeight(35)
        button3.setFixedWidth(150)
        button3.move(170,220)  # 移動位置
        button3.clicked.connect(self.buttonClicked3)

        button4 = QPushButton(self)
        button4.setText("accuracy/lose")  # 建立名字
        button4.setStyleSheet("background-color:#FCFCFC; ")
        button4.setFixedHeight(35)
        button4.setFixedWidth(150)
        button4.move(170,270)  # 移動位置
        button4.clicked.connect(self.buttonClicked4)

        label4 = QLabel(self)
        label4.setText("Test Image Index: ")
        label4.setFixedWidth(150)
        label4.setFixedHeight(35)
        label4.move(180,320)
        self.line1 = QLineEdit(self)
        self.line1.move(165,360)
        self.line1.setFixedWidth(170)
        self.line1.setFixedHeight(35)
        self.line1.setPlaceholderText("0~9999")
        button5 = QPushButton(self)
        button5.setText("Inference")  # 建立名字
        button5.setStyleSheet("background-color:#FCFCFC; ")
        button5.setFixedHeight(35)
        button5.setFixedWidth(150)
        button5.move(170,420)  # 移動位置
        button5.clicked.connect(self.buttonClicked5) # 設置button啟動functio4

    

    def buttonClicked1(self): # 顯示圖片
        first = random.randint(0,50000)
        second = random.randint(0,50000)
        
        img = np.hstack([x_train[first],x_train[second]])
        pic_label(y_train[first])
        pic_label(y_train[second])
        for i in range(8):
            num = random.randint(0,50000)
            img = np.hstack([img, x_train[num]])
            pic_label(y_train[num])
        row,col = img.shape[:2]
        scale_img = cv2.resize(img,(int(4*col),int(4*row)),interpolation=cv2.INTER_CUBIC)
        
        cv2.imshow("test",scale_img)

    def buttonClicked2(self):
        print('Hyperparameters:\nbatch size: 32\nlearning rate: 0.001\noptimizer: SGD')

    def buttonClicked3(self):
        model.summary()

    def buttonClicked4(self):
        img1 = cv2.imread('accuracy.jpg')
        img2 = cv2.imread('loss.jpg')
        cv2.imshow("accuracy",img1)
        cv2.imshow("loss",img2)

    def buttonClicked5(self):
        
        num = int(self.line1.text())
        #print(num)
        img = x_test[num]
        cv2.imshow("img",img)
        #print(y_test[num])
        img = img[np.newaxis,:,:]
        pro = model.predict(img) #機率
        
        prob = np.array([pro[0][0],pro[0][1],pro[0][2],pro[0][3],pro[0][4],pro[0][5],pro[0][6],pro[0][7],pro[0][8],pro[0][9]])
        #print(prob)
        
        answer = np.argmax(pro, axis=1)
        #print(np.argmax(pro, axis=1)) #最像的
        pic_label(answer)
       
        categories=['plane', 'mobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse','ship','truck']  #X軸刻度
        x=np.arange(len(categories))                     #產生X軸座標序列
        plt.ylim(0,1)
        plt.bar(x, prob, tick_label=categories)     #繪製長條圖
        #plt.savefig('probability.jpg')
        plt.show()
        
def pic_label(i):
    if i == 0:
        print("airplane")
    elif i == 1:
        print("automobile")
    elif i == 2:
        print("bird")
    elif i == 3:
        print("cat")
    elif i == 4:
        print("deer")
    elif i == 5:
        print("dog")
    elif i == 6:
        print("frog")
    elif i == 7:
        print("horse")
    elif i == 8:
        print("ship")
    elif i == 9:
        print("truck")
def plot_image(image):                         
    fig=plt.gcf()                                          
    fig.set_size_inches(2, 2)                       
    plt.imshow(image, cmap='binary')       
    plt.show() 

app = QApplication(sys.argv)
w = MainWindow()
w.show()
sys.exit(app.exec_())
'''(x_train, y_train), (x_test, y_test) = cifar10.load_data()
cv2.imshow("test",x_train[0])
print(y_train[0])

cv2.waitKey(0)
cv2.destroyAllWindows()'''
'''
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)'''

'''epoch = 5
batch_size = 16
weight_decay = 0.0005
#1
model = Sequential()
model.add(Conv2D(64, (3, 3), padding='same', input_shape=(32,32,3),kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
#2
model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
#3
model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
#4
model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
#5
model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
#6
model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
#layer7 8*8*256
model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
#layer8 4*4*256
model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
#layer9 4*4*512
model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
#layer10 4*4*512
model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
#layer11 2*2*512
model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
#layer12 2*2*512
model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
#layer13 2*2*512
model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
#layer14 1*1*512
model.add(Flatten())
model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
#layer15 512
model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
#layer16 512
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
model.fit(x_train, y_train, epochs=epoch, batch_size=batch_size, validation_split=0.1, verbose=1)

loss, acc = model.evaluate(x_val, y_val, batch_size=batch_size, verbose=1, sample_weight=None)
print("accuracy:",acc)
print("loss",loss)

y_pred = model.predict_classes(x_test, batch_size=32, verbose=0)
y_pred = keras.utils.to_categorical(y_pred, 10)
#测试集的准确率
print("accuracy score:", accuracy_score(y_test, y_pred))
#分类报告
print(classification_report(y_test, y_pred))
#保存模型
#官方文档不推荐使用pickle或cPickle来保存Keras模型
model.save('D:/florrie/four1/vision/Opencv_Hw1_05/keras_vgg16_cifar10.h5')
#.h5文件是 HDF5文件，改文件包含：模型结构、权重、训练配置（损失函数、优化器等）、优化器状态
#使用keras.models.load_model(filepath)来重新实例化你的模型

#只保存模型结构
# save as JSON
json_string = model.to_json()
# save as YAML
yaml_string = model.to_yaml()'''