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
#import keras
#from tensorflow.python.keras.models import Sequential
#from tensorflow.python.keras.layers import Dense
#from tensorflow.keras.applications import ResNet50
#from tensorflow import keras


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers.advanced_activations import LeakyReLU
import keras


import seaborn as sns 
from sklearn import metrics 
from sklearn.metrics import confusion_matrix




model = keras.models.load_model('mask_detector.model')
#testing the model
img = load_img("with_mask.jpg", target_size=(256, 256))
img = img_to_array(img)
img = preprocess_input(img)
img = np.array(img, dtype=np.float16) / 225.0
img = img.reshape(-1,256,256,3)
predictmodel = model.predict(img)
print(predictmodel)

# Generate arg maxes for predictions
classes = np.argmax(predictmodel, axis = 1)
print(classes)

img2 = load_img("with_mask2.jpg", target_size=(256, 256))
img2 = img_to_array(img2)
img2 = preprocess_input(img2)
img2 = np.array(img2, dtype=np.float16) / 225.0
img2 = img2.reshape(-1,256,256,3)
predictmodel = model.predict(img2)
print(predictmodel)

# Generate arg maxes for predictions
classes = np.argmax(predictmodel, axis = 1)
print(classes)


img = load_img("with_mask3.jpg", target_size=(256, 256))
img = img_to_array(img)
img = preprocess_input(img)
img = np.array(img, dtype=np.float16) / 225.0
img = img.reshape(-1,256,256,3)
predictmodel = model.predict(img)
print(predictmodel)

# Generate arg maxes for predictions
classes = np.argmax(predictmodel, axis = 1)
print(classes)

img2 = load_img("without_mask.jpg", target_size=(256, 256))
img2 = img_to_array(img2)
img2 = preprocess_input(img2)
img2 = np.array(img2, dtype=np.float16) / 225.0
img2 = img2.reshape(-1,256,256,3)
predictmodel = model.predict(img2)
print(predictmodel)

# Generate arg maxes for predictions
classes = np.argmax(predictmodel, axis = 1)
print(classes)


img = load_img("without_mask2.jpg", target_size=(256, 256))
img = img_to_array(img)
img = preprocess_input(img)
img = np.array(img, dtype=np.float16) / 225.0
img = img.reshape(-1,256,256,3)
predictmodel = model.predict(img)
print(predictmodel)

# Generate arg maxes for predictions
classes = np.argmax(predictmodel, axis = 1)
print(classes)

img2 = load_img("without_mask3.jpg", target_size=(256, 256))
img2 = img_to_array(img2)
img2 = preprocess_input(img2)
img2 = np.array(img2, dtype=np.float16) / 225.0
img2 = img2.reshape(-1,256,256,3)
predictmodel = model.predict(img2)
print(predictmodel)

# Generate arg maxes for predictions
classes = np.argmax(predictmodel, axis = 1)
print(classes)