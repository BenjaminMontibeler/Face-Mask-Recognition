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


DIRECTORY = r"C:\Users\benja\Desktop\face mask recognition\dataset"
CATEGORIES = ["with_mask", "without_mask"]

data = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(256, 256))
        image = img_to_array(image)
        image = preprocess_input(image)
        data.append(image)
        labels.append(category)


print(labels)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype="float32")
labels = np.array(labels)


print(labels)

label_counts = pd.DataFrame(labels).value_counts()

print(data.shape, labels.shape)
#(trainX, testX, trainY, testY) = train_test_split(data, labels,
	#test_size=0.20, stratify=labels, random_state=8)

# Normalize and reshape data
#trainX = np.array(trainX, dtype=np.float16) / 225.0
#trainX = trainX.reshape(-1,500,500,3)
#testX = np.array(testX, dtype=np.float16) / 225.0
#testX = testX.reshape(-1,500,500,3)



# Splitting the training data set into training and validation data sets
(trainX, valX, trainY, valY) = train_test_split(data, labels, 
    test_size=0.25, stratify=labels, random_state= 8)


#print(trainX.shape,valX.shape,trainY.shape, valY.shape)

#augmentation




# Normalize and reshape data
trainX = np.array(trainX, dtype=np.float16) / 225.0
trainX = trainX.reshape(-1,256,256,3)
valX = np.array(valX, dtype=np.float16) / 225.0
testX = valX.reshape(-1,256,256,3)
print(trainX.shape, valX.shape, trainY.shape, valY.shape)


aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")


# Label binarizing
#lb = LabelBinarizer()
#trainY = lb.fit_transform(trainY)
#valY = lb.fit_transform(valY)
"""
# Building model architecture
model = Sequential()
model.add(Conv2D(64, (3, 3), padding="same",input_shape=(500,500,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(246, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(1, activation="sigmoid"))


# Compiling model
model.compile(loss = 'binary_crossentropy', optimizer = Adam(0.0005),metrics=['accuracy'])
model.summary()
# Training the model
epochs = 20
batch_size = 128
history = model.fit(trainX, trainY, batch_size = batch_size, epochs = epochs, validation_data = (valX, valY))
"""
#konvolucijska
"""

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(256,256,3),padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2),padding='same'))
model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(7, 7),padding='same'))
model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))                  
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Flatten())
model.add(Dense(128, activation='linear'))
model.add(LeakyReLU(alpha=0.1))                  
model.add(Dense(2, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(),metrics=['accuracy'])
model.summary()
history = model.fit(aug.flow(trainX, trainY, batch_size=128),epochs=20,verbose=1,validation_data=(valX, valY))

# plot the training loss and accuracy
N = 20
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), history.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), history.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.show()
plt.savefig("plot1.png")


#------------------

y_pred=model.predict(valX) 
y_pred=np.argmax(y_pred, axis=1)
y_test=np.argmax(valY, axis=1)
cm = confusion_matrix(y_test, y_pred)
print(cm)

cf_matrix = confusion_matrix(y_test, y_pred)

ax = sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
            fmt='.2%', cmap='Blues')

ax.set_title('Seaborn Confusion Matrix with labels\n\n')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['With mask','Without mask'])
ax.yaxis.set_ticklabels(['With mask','Without mask'])

## Display the visualization of the Confusion Matrix.
plt.show()
plt.savefig("cf1.png")


# Saving model
model.save("mask_detector.model", save_format="h5")
"""
#-----
"""
#predict labels
predicted_classes = model.predict(valX)
predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
print(predicted_classes.shape, valY.shape)
correct = np.where(predicted_classes==valY)[0]
print ("Found %d correct labels" % len(correct))
for i, correct in enumerate(correct[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(valX[correct].reshape(256,256), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], valY[correct]))
    plt.tight_layout()

incorrect = np.where(predicted_classes!=valY)[0]
print ("Found %d incorrect labels" % len(incorrect))
for i, incorrect in enumerate(incorrect[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(valX[incorrect].reshape(256,256), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], valY[incorrect]))
    plt.tight_layout()
"""
#clasiffication report
#predicted_classes = model.predict(valX)
#from sklearn.metrics import classification_report
#target_names = ["Class {}".format(i) for i in range(2)]
#print(classification_report(valY, predicted_classes, target_names=target_names))





# Building a model with transfer learning
"""
model = Sequential()
model.add(ResNet50(include_top = False, pooling = 'avg', weights='imagenet'))
model.add(Dense(2, activation = 'softmax'))

model.layers[0].trainable = False
#compile the model with your favourite optimizer, loss function and metrics 
model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = "accuracy")
"""





"""
data_augmentation = keras.Sequential(
    [       keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
   keras.layers.experimental.preprocessing.RandomRotation(0.1),
    ]
)

base_model = keras.applications.Xception(
    weights='imagenet',  
    input_shape=(256, 256, 3),
    include_top=False)

base_model.trainable = False

inputs = keras.Input(shape=(256, 256, 3))

x = data_augmentation(inputs) 

x = tf.keras.applications.xception.preprocess_input(x)

x = base_model(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)  
outputs = keras.layers.Dense(2)(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),metrics=keras.metrics.CategoricalAccuracy())
"""



tf.keras.applications.Xception(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
)

from tensorflow import keras
from tensorflow.keras import layers
data_augmentation = keras.Sequential(
   [layers.RandomFlip("horizontal"), layers.RandomRotation(0.1),]
)

model = Sequential()
model.add(Flatten())
model.add(Dense(128, activation='linear'))
model.add(LeakyReLU(alpha=0.1))                  
model.add(Dense(2, activation='softmax'))

#base model

base_model = keras.applications.Xception(
   weights="imagenet",  # Weights pre-trained on ImageNet.
   input_shape=(256, 256, 3),
   include_top=False,
)

base_model.trainable = False


#train the top layer

inputs = keras.Input(shape=(256, 256, 3))
x = data_augmentation(inputs)  # Apply random data augmentation

#fine tuning
base_model.trainable = True

model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(1e-5),metrics=['accuracy'])
epochs = 20
history = model.fit(trainX, trainY, batch_size=128,epochs=20,verbose=1,validation_data=(valX, valY))


# plot the training loss and accuracy
N = 20
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), history.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), history.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.show()
plt.savefig("plot1.png")


#confusion matrix
y_pred=model.predict(valX) 
y_pred=np.argmax(y_pred, axis=1)
y_test=np.argmax(valY, axis=1)
cm = confusion_matrix(y_test, y_pred)
print(cm)

cf_matrix = confusion_matrix(y_test, y_pred)

ax = sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
            fmt='.2%', cmap='Blues')

ax.set_title('Seaborn Confusion Matrix with labels\n\n')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['With mask','Without mask'])
ax.yaxis.set_ticklabels(['With mask','Without mask'])

## Display the visualization of the Confusion Matrix.
plt.show()
plt.savefig("cf1.png")

model.save("mask_detector_tl.model", save_format="h5")

# Saving model
