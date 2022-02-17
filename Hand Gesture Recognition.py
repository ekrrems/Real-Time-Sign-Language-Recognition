
import pandas as pd
import numpy as np
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import os


from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Activation,Dropout,Dense,Flatten,BatchNormalization
from keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img,array_to_img
from keras.utils import to_categorical
import pickle



train_data = pd.read_csv('C:\\Users\\ekrem\\OneDrive\\Masaüstü\\Machine learning DATAS\\sign_mnist_train.csv')
test_data = pd.read_csv('C:\\Users\\ekrem\\OneDrive\\Masaüstü\\Machine learning DATAS\\sign_mnist_test.csv')

train_data.head()

y_train = train_data['label']
X_train = train_data.drop('label',axis=1)

print(X_train.shape)
print(y_train.shape)

test_data.head()
y_test = test_data['label']
X_test = test_data.drop('label',axis=1)

print(X_test.shape)
print(y_test.shape)

X_train = np.array(X_train).reshape((X_train.shape[0],28,28,1))
X_test = np.array(X_test).reshape((X_test.shape[0],28,28,1))

X_train = X_train/255
X_test = X_test/255


plt.imshow(X_train[0].reshape((28,28)))
print(y_train[0])



numofclass = len((y_train.unique()))
print(numofclass)



# CNN Model

model = Sequential()

model.add(Conv2D(32,(3,3),input_shape=X_train[0].shape))
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Conv2D(32,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D())


model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(1024)) 
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(25))
model.add(Activation('softmax'))

model.compile(loss='sparse_categorical_crossentropy',
             optimizer='rmsprop',
             metrics=['accuracy'])
print('ended')

batch_size = 32

train_datagen = ImageDataGenerator(shear_range=0.1,
                                   zoom_range=0.3,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   rotation_range=5)
test_datagen = ImageDataGenerator()
                                 

train_generator = train_datagen.flow(X_train,y_train,
                                     batch_size=batch_size)
test_generator = test_datagen.flow(X_test,y_test)


X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],X_train.shape[2],1))
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],X_test.shape[2],1))

hist = model.fit_generator(train_generator,
                           epochs=50,
                           steps_per_epoch =X_train.shape[0]//batch_size,
                           validation_data = test_generator,validation_steps=15)





pickle_out = open("sign_model_normalized.p","wb")
pickle.dump(model,pickle_out)
pickle_out.close()


pickle_in = open("sign_model_normalized.p","rb")
model1 = pickle.load(pickle_in)


predict = model1.predict(X_train[2].reshape((1,28,28,1)))
print("Model's Prediction is : "+ str(np.argmax(predict.astype('int'))))



































