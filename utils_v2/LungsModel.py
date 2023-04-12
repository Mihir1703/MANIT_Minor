import itertools, cv2, os, time

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.preprocessing import MinMaxScaler

# import tensorflow_hub as hub

from tensorflow import keras
from tensorflow.keras.applications.efficientnet import EfficientNetB7, preprocess_input
from tensorflow.keras.applications.efficientnet import EfficientNetB6, preprocess_input
from tensorflow.keras.applications.efficientnet import EfficientNetB5, preprocess_input
from tensorflow.keras.layers import InputLayer, Input, Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from keras.callbacks import CSVLogger

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import plot_confusion_matrix

from glob import glob


# ignore warnings
import warnings
warnings.filterwarnings('ignore')


train_dir = os.path.join('lung_colon_image_set/lung_image_sets')

# lung adenocarcinoma path
lung_aca = os.path.join('lung_colon_image_set/lung_image_sets/lung_aca')

# lung benign path
lung_n = os.path.join('lung_colon_image_set/lung_image_sets/lung_n')

# lung squamos cell carcinoma path
lung_scc = os.path.join('lung_colon_image_set/lung_image_sets/lung_scc')


train_aca_names = os.listdir(lung_aca)
print(f'TRAIN SET ADENOCARCINOMA: {train_aca_names[:10]}')
print('\n')

train_n_names = os.listdir(lung_n)
print(f'TRAIN SET BENIGN: {train_n_names[:10]}')
print('\n')

train_scc_names = os.listdir(lung_scc)
print(f'TRAIN SET SQUAMOS CELL CARCINOMA: {train_scc_names[:10]}')


print(f'total training Adenocarcinoma images: {len(os.listdir(lung_aca))}')
print(f'total training Benign images: {len(os.listdir(lung_n))}')
print(f'total training Squamous Cell Carcinoma images: {len(os.listdir(lung_scc))}')

# calculate number of training images
train_aca = len(os.listdir(lung_aca))
train_n = len(os.listdir(lung_n))
train_scc = len(os.listdir(lung_scc))
total = train_aca + train_n + train_scc

# print total number of images
print('Total Images in dataset: %s' % str((total))) 


data_dir = 'lung_colon_image_set/lung_image_sets/'

# making a data split of 80-20
data = ImageDataGenerator(validation_split = 0.2)

# setting up the batch size
BATCH_SIZE = 128

# setting up the image size
X = Y = 224

# making training dataset
train_data = data.flow_from_directory(data_dir,
                                    class_mode = "categorical",
                                    target_size = (X, Y),
                                    color_mode="rgb",
                                    batch_size = BATCH_SIZE, 
                                    shuffle = False,
                                    subset='training',
                                    seed = 42)

# making validation dataset
val_data = data.flow_from_directory(data_dir,
                                      class_mode = "categorical",
                                      target_size = (X, Y),
                                      color_mode="rgb",
                                      batch_size = BATCH_SIZE, 
                                      shuffle = False,
                                      subset='validation',
                                      seed = 42)



label =  {0: "lung adenocarcinoma", 1: "benign", 2: "squamous cell carcinoma"}
for t in label.keys():
    print(t, label[t])



# initializing efficientnet b7 cnn model
eff_5 = EfficientNetB5 (
        input_shape=(X, Y, 3),
        weights='imagenet',
        include_top=False
        )

# setting trainable to false    
eff_5.trainable = False

# global avg pooling layer
x = GlobalAveragePooling2D()(eff_5.output)
# flatten layer
x = Flatten()(x)
# fully connected layer 1
x = Dense(128, activation='relu')(x)
# fully connected layer 2
x = Dense(64, activation='relu')(x)

# outout layer
y = Dense(3, activation='softmax')(x) # since we have three outputs, we will use 3 neurons in last layer

# setting up y = f(x)
# tie it together
model_efb5 = Model(inputs=eff_5.input, 
              outputs=y)



model_efb5.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)


# for accuracy
class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if logs.get('val_accuracy') >= 0.998:
                print('\nReached 99.8% accuracy so cancelling training!')
                self.model.stop_training=True
                
callbacks = myCallback()



early_stopping = EarlyStopping(monitor='val_loss', 
                               patience=3)

logger = CSVLogger('logs_efb5.csv', append=True)

EPOCHS = 50

# fitting the model to training data
history = model_efb5.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[callbacks, logger]
)



hist = history.history
hist.keys()


# save the model
model_efb5.save('model_efb5_99.83acc.h5')


scores = model_efb5.evaluate(val_data)
print("%s: %.2f%%" % (model_efb5.metrics_names[1], scores[1] * 100))


model_efb5.save('model_optimized_99.87acc.h5')


y_hat = model_efb5.predict(val_data)
y_pred = np.argmax(y_hat, axis=1)

# print classification report
print(classification_report(val_data.classes, y_pred))


