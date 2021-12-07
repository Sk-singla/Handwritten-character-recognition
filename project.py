import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

# importing dataset
dataset = pd.read_csv("english.csv")
print(dataset.head())

# data pre-processing
X = []
for imgUrl in dataset.iloc[:,0].values:
    X.append(cv2.imread(imgUrl))

x = np.array(X)
print(x.shape)
y = dataset.iloc[:,1].values

y_unique = len(np.unique(dataset.iloc[:,1].values))
imgSize = (50,50)
def preProcessing(img):
    img = cv2.resize(img,imgSize)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    img = img.reshape(imgSize[0],imgSize[0],1)
    return img

def showImg(img):
    cv2.imshow("randojm",img)
    cv2.waitKey(0)


X = [preProcessing(i) for i in x]
X = np.array(X)

le = LabelEncoder()
y = le.fit_transform(y)
y = to_categorical(y, len(np.unique(y)))

# spliting data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Defining Neural network Model
def myModel():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',padding="same",
                     input_shape=(imgSize[0],imgSize[1], 1)))
    model.add(Conv2D(64, (3, 3), activation='relu',padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(y_unique, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy',tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])
    return model

model = myModel()
print(model.summary())

history = model.fit(X_train, y_train, batch_size=128,
                    epochs=50, verbose=1, validation_split=0.3)

# Testing
loss, accuracy, precision, recall = model.evaluate(X_test, y_test)
print("Accuracy:",accuracy)
print("Loss:",loss)
print("Precision:",precision)
print("Recall:",recall)

# Saving Model in external file
model.save('digits.model')

# Ploting Accuracy, loss, precision and recall values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


plt.plot(history.history['precision_1'])
plt.plot(history.history['val_precision_1'])
plt.title('model precision')
plt.ylabel('precision')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.plot(history.history['recall_1'])
plt.plot(history.history['val_recall_1'])
plt.title('model recall')
plt.ylabel('recall')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# Testing on own Image
test_img = cv2.imread("a.png")
showImg(test_img)
test_img = preProcessing(test_img)
predictions = model.predict(test_img.reshape(1,X_test[0].shape[0],X_test[0].shape[1],1))
prob = np.argmax(predictions)
print(predictions)
print(le.inverse_transform([prob]))

