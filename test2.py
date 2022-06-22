from msilib.schema import _Validation_records
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt
from keras.layers.advanced_activations import LeakyReLU
import numpy as np
import os
import matplotlib.pyplot as plt
#import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
##from tensorflow.keras.optimizers import Adam
from keras.models import Sequential, Input, Model
from keras.layers import Conv2D, MaxPooling2D, Dropout
##from sklearn.model_selection import train_test_split
#from tensorflow.keras.applications import ResNet50
import tensorflow as tf
#-----
#import keras
#from tensorflow.python.keras.models import Sequential
#from tensorflow.python.keras.layers import Dense
#from tensorflow.keras.applications import ResNet50
from tensorflow import keras
#-----

from tensorflow.keras.preprocessing.image import ImageDataGenerator



import seaborn as sns 
from sklearn import metrics 
from sklearn.metrics import confusion_matrix

from numpy import expand_dims

import os
DIRECTORY = r"C:\Users\benja\Desktop\face mask recognition\slike"
CATEGORIES = ["with_mask", "without_mask"]

data = []
labels = []

model = keras.models.load_model('mask_detector_tl.model')


for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(256, 256))
        image = img_to_array(image)
        image = preprocess_input(image)
        image = np.array(image, dtype="float32")
        image = np.array(image, dtype=np.float16) / 225.0
        image = image.reshape(-1,256,256,3)
        predictmodel = model.predict(image)
        #print(predictmodel)
        classes = np.argmax(predictmodel, axis = 1)
        print(classes)



