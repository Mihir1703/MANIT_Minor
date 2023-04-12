import cv2
import numpy as np

img1 = cv2.imread('/home/mihir/Minor/Minor_project/raw_files/img_dataset/Photo_Dataset/Photos Dataset/3.jpg')
img2 = cv2.imread('/home/mihir/Minor/Minor_project/processed_files/compressed/3.jpg.webp')

width = 800
height = 600
img1 = cv2.resize(img1, (width, height))
img2 = cv2.resize(img2, (width, height))


img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float32)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float32)

mse = np.mean((img1_gray - img2_gray) ** 2)
max_val = np.amax(img1_gray)
psnr = 20 * np.log10(max_val / np.sqrt(mse))

print('PSNR:', psnr)



# import pandas as pd
# import numpy as np
# import tensorflow as tf
# import os
# import cv2
# import matplotlib.pyplot as plt
# from tensorflow.keras import layers
# from tensorflow.keras import Model
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.applications import ResNet101V2

# train_path = '/home/mihir/Minor/Minor_project/raw_files/Lungs Cancer Dataset/Data/train'
# val_path = '/home/mihir/Minor/Minor_project/raw_files/Lungs Cancer Dataset/Data/valid'
# test_path = '/home/mihir/Minor/Minor_project/raw_files/Lungs Cancer Dataset/Data/test'

# def plot_images(img_dir, top=10):
#     all_img_dirs = os.listdir(img_dir)
#     img_files = [os.path.join(img_dir, file) for file in all_img_dirs][:5]
#     plt.figure(figsize=(10, 10))  
#     for idx, img_path in enumerate(img_files):
#         plt.subplot(1, 5, idx+1)
#         img = plt.imread(img_path)
#         plt.tight_layout()        
#         plt.axis('off')
#         plt.imshow(img, cmap='gray') 


# path = "/home/mihir/Minor/Minor_project/raw_files/Lungs Cancer Dataset/Data/train"
# filedirectory = []
# for files in os.listdir(path):
#     filedirectorys = filedirectory.append(os.path.join(path,files))


# train_datagen = ImageDataGenerator(rescale = 1.0/255.0,
#                                   horizontal_flip = True,
#                                   fill_mode = 'nearest',
#                                   zoom_range=0.2,
#                                   shear_range = 0.2,
#                                   width_shift_range=0.2,
#                                   height_shift_range=0.2,
#                                   rotation_range=0.4)
# train_generator = train_datagen.flow_from_directory(train_path,
#                                                    batch_size = 5,
#                                                    target_size = (350,350),
#                                                    class_mode = 'categorical')

# test_datagen = ImageDataGenerator(rescale = 1.0/255.0)
# test_generator = test_datagen.flow_from_directory(test_path,
#                                                    batch_size = 5,
#                                                    target_size = (350,350),
#                                                    class_mode = 'categorical')


# baseModel = ResNet101V2(weights="imagenet", include_top=False,input_shape = (350,350,3))
# for layer in baseModel.layers:
#     layer.trainable = False

# x = baseModel.output
# x = tf.keras.layers.Flatten()(x)
# x = tf.keras.layers.Dense(1024, activation = 'relu')(x)
# x = tf.keras.layers.Dropout(0.2)(x)
# x = tf.keras.layers.Dense(4, activation = "softmax")(x)

# model = Model(inputs= baseModel.input , outputs = x)

# model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics = ['acc'])
# earlystop = EarlyStopping(patience=5)

# history = model.fit(train_generator,
#                     steps_per_epoch = 100,
#                     epochs = 15,
#                     verbose = 1,
#                     validation_data = test_generator,
#                     validation_steps = 50,
#                     callbacks = [earlystop])


