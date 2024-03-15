#------------------------------------------------------------------------------------------------------------------------
"IMPORT THE LIBRARIES"

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import glob
import seaborn as sns
import pandas as pd

from keras.models import Model
from keras.layers import Flatten, Dense, Dropout,Conv2D,MaxPool2D ,BatchNormalization
from keras.models import load_model
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

#------------------------------------------------------------------------------------------------------------------------
"IMAGE PREPROCESSING"

SIZE = 75  #Resize images

#Capture training data and labels into respective lists
data_images = []
data_labels = []

for directory_path in glob.glob("/content/data/SEN12FLOOD/S2/*.tif"):
    label = directory_path.split("\\")[-1]
    print(label)
    for img_path in glob.glob(os.path.join(directory_path, "*.tif")):
        print(img_path)
        img = cv2.imread(img_path, -1)
        img = cv2.resize(img, (SIZE, SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        data_images.append(img)
        data_labels.append(label)
        print(label)



#Convert lists to arrays
data_images = np.array(data_images)
data_labels = np.array(data_labels)


#Encode labels from text to integers.
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(data_labels)
data_labels_encoded = le.transform(data_labels)

# Data Normalization
data_images = data_images / 255.0


img_dir='/content/data/SEN12FLOOD/S2'
datagen=ImageDataGenerator(rescale=1/255)


data_gen=datagen.flow_from_directory(img_dir,
                                      target_size=(75,75),
                                      batch_size=4,
                                      class_mode='binary')
print(data_gen)


#------------------------------------------------------------------------------------------------------------------------
"FEATURE EXTRACTION USING CNN AND VGG16"


"CNN Model Feature Extraction"


model = Sequential()
model.add(Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (75,75,3)))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Conv2D(128 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Conv2D(256 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Conv2D(512 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Flatten())
model.add(Dense(units = 128 , activation = 'relu'))
model.add(Dropout(0.2))
#model.add(Dense(units = 10 , activation = 'sigmoid'))
model.compile(optimizer = "rmsprop" , loss = 'binary_crossentropy')
model.summary()


history = model.fit(data_images,data_labels_encoded, batch_size = 32 ,epochs = 1 )
