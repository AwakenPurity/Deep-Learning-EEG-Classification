from tensorflow.python import keras
from tensorflow.python.keras import Input

import loadData
import seaborn
import tensorflow as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD,Adam
from tensorflow.python.keras.models import Sequential 
from tensorflow.python.keras.layers import SimpleRNN
from tensorflow.python.keras.layers.core import *
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow import sigmoid
# import tensorflow.python.keras.metrics.loss
from tensorflow.keras import backend as K
from sklearn.metrics import confusion_matrix
import os
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pdb

print('loading data.......')
X_train, Y_train, X_test, Y_test = loadData.loadData()

print('finished loading data.......')

def buildModel():
    newmodel = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(3000, 1)),
        # 第一个卷积层, 4 个 21x1 卷积核
        tf.keras.layers.Conv1D(filters=4, kernel_size=21, strides=1, padding='SAME', activation='relu'),
        # 第一个池化层, 最大池化,4 个 3x1 卷积核, 步长为 2
        tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='SAME'),
        # 第二个卷积层, 16 个 23x1 卷积核
        tf.keras.layers.Conv1D(filters=16, kernel_size=23, strides=1, padding='SAME', activation='relu'),
        # 第二个池化层, 最大池化,4 个 3x1 卷积核, 步长为 2
        tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='SAME'),
        # 第三个卷积层, 32 个 25x1 卷积核
        tf.keras.layers.Conv1D(filters=32, kernel_size=25, strides=1, padding='SAME', activation='relu'),
        # 第三个池化层, 平均池化,4 个 3x1 卷积核, 步长为 2
        tf.keras.layers.AvgPool1D(pool_size=3, strides=2, padding='SAME'),
        # 第四个卷积层, 64 个 27x1 卷积核
        tf.keras.layers.Conv1D(filters=64, kernel_size=27, strides=1, padding='SAME', activation='relu'),
        # 打平层,方便全连接层处理'
        tf.keras.layers.GlobalAvgPool1D(),
        # 全连接层,128 个节点 转换成128个节点
        # tf.keras.layers.Dense(128, activation='relu'),
        # Dropout层,dropout = 0.2
        tf.keras.layers.Dropout(rate=0.2),
        # # 全连接层,5 个节点
        # tf.keras.layers.Dense(5, activation='softmax')
    ])
    return newmodel

adam = Adam(learning_rate=1e-4)

# checkpoint = ModelCheckpoint(
#         f'./savemodel/NSV/model_best_for_NSV.h5',
#         monitor="val_accuracy",
#         mode="max", save_best_only=True, verbose=1)
# callbacks = [checkpoint]

# 构建CNN模型
model = buildModel()
model.compile(optimizer=adam,
                  loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
model.summary()

# 训练与验证
model.fit(X_train, Y_train, epochs=30,
                  batch_size=250,
                  # 训练集所占比例
                  validation_split=0.2,
                   # callbacks=callbacks
                  )
# 预测
# Y_pred = model.predict_classes(X_test)
Y_pred = np.argmax(model.predict(X_test), axis=-1)
