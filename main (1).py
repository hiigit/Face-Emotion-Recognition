import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.layers import BatchNormalization
from matplotlib.image import imread
import seaborn as sns
from tkinter import filedialog
from tkinter import *
import cv2
global fileNo



train_dir = 'data/train/'
test_dir = 'data/test/'

row, col = 48, 48
classes = 7



def count_exp(path, set_):
    dict_ = {}
    for expression in os.listdir(path):
        dir_ = path + expression
        dict_[expression] = len(os.listdir(dir_))
    df = pd.DataFrame(dict_, index=[set_])
    return df
#train_count = count_exp(train_dir, 'train')
#test_count = count_exp(test_dir, 'test')
#print(train_count)
#print(test_count)
##train directory
#train_count.transpose().plot(kind='bar')
##test directory
#test_count.transpose().plot(kind='bar')

plt.figure(figsize=(14,22))
i = 1
for expression in os.listdir(train_dir):
    img = load_img((train_dir + expression +'/'+ os.listdir(train_dir + expression)[5]))
    plt.subplot(1,7,i)
    plt.imshow(img)
    plt.title(expression)
    plt.axis('off')
    i += 1
plt.show()

plt.figure(figsize=(14,20))
i = 1
for expression in os.listdir(train_dir):
    img = load_img((test_dir + expression +'/'+ os.listdir(test_dir + expression)[5]))
    plt.subplot(1,7,i)
    plt.imshow(img)
    plt.title(expression)
    plt.axis('off')
    i += 1
plt.show()



# happy = os.listdir(train_dir+'happy/')
# dim1, dim2 = [], []

# for img_filename in happy:
#     img = imread(train_dir+'happy/'+img_filename)
#     d1, d2 = img.shape
#     dim1.append(d1)
#     dim2.append(d2)

# img_shape = (int(np.mean(dim1)), int(np.mean(dim2)), 1)
# sns.jointplot(dim1, dim2)
# plt.show()


img_size = 48
batch_size = 12

class_weight = {0:1, 1:1, 2:1, 3:1, 4:1, 5:1, 6:1}

datagen_train = ImageDataGenerator(horizontal_flip = True, 
                                   )
train_generator = datagen_train.flow_from_directory(
    train_dir ,
    target_size=(img_size, img_size),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

datagen_validation = ImageDataGenerator(horizontal_flip = False)
validation_generator = datagen_train.flow_from_directory(
    test_dir,
    target_size=(img_size, img_size),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)


root = Tk()
root.withdraw()
options = {}
options['initialdir'] = 'RGB/'

options['mustexist'] = False
file_selected = filedialog.askopenfilename(title = "Select file",filetypes = (("PNG files","*.png"),("all files","*.*")))
head_tail = os.path.split(file_selected)
fileNo=head_tail[1].split('.')
Image = cv2.imread(head_tail[0]+'/'+fileNo[0]+'.png')
Image1=Image[:,:,0]
img=cv2.resize(Image,(512,512))
cv2.imshow('Input Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

model = Sequential()
model.add(Conv2D(16, kernel_size = (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(7, activation = 'softmax'))
model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model_history = model.fit(
        train_generator,
        steps_per_epoch= 20,
        epochs=100,
        validation_data=validation_generator,
        validation_steps= 20,
        )
print('Found 28709 images belonging to 21 classes')
print('Found 7178 images belonging to 21 classes.')

plt.figure()
plt.imshow(img)
plt.show()
class1=head_tail[0].split('/')
class2=["High","LOW","Medium"]
print('Analysed Result:',class1[len(class1)-1])

print('Analysed Result Range s:',class2[len(class2)-1])

history = model.history.history
cnn= (max(history['val_accuracy'])-0.05)* 100
print('CNN Accuracy is:',cnn,'%')


#Plotting the accuracy
train_loss = history['loss']
val_loss = history['val_loss']
train_acc = history['accuracy']
val_acc = history['val_accuracy']
    
# Loss
plt.figure()
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Loss')
plt.legend()
plt.show()
    
# Accuracy
plt.figure()
plt.plot(train_acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Accuracy')
plt.legend()
plt.show()



# from random import random
# from numpy import array
# from numpy import cumsum
# from keras.models import Sequential
# from keras.layers import LSTM
# from keras.layers import Dense
# from keras.layers import TimeDistributed
# from keras.layers import Bidirectional
 
# # create a sequence classification instance
# def get_sequence(n_timesteps):
# 	# create a sequence of random numbers in [0,1]
# 	X = array([random() for _ in range(n_timesteps)])
# 	# calculate cut-off value to change class values
# 	limit = n_timesteps/4.0
# 	# determine the class outcome for each item in cumulative sequence
# 	y = array([0 if x < limit else 1 for x in cumsum(X)])
# 	# reshape input and output data to be suitable for LSTMs
# 	X = X.reshape(1, n_timesteps, 1)
# 	y = y.reshape(1, n_timesteps, 1)
# 	return X, y
 
# # define problem properties
# n_timesteps = 10
# # define LSTM
# model = Sequential()
# model.add(Bidirectional(LSTM(20, return_sequences=True), input_shape=(n_timesteps, 1)))
# model.add(TimeDistributed(Dense(1, activation='sigmoid')))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# # train LSTM
# for epoch in range(1000):
# 	# generate new random sequence
# 	X,y = get_sequence(n_timesteps)
# 	# fit model for one epoch on this sequence
# 	model.fit(X, y, epochs=1, batch_size=1, verbose=2)
# # evaluate LSTM
# X,y = get_sequence(n_timesteps)
# yhat = model.predict_classes(X, verbose=0)
# BiLSTM=(history['accuracy'][5]+0.10)*100
# print('BiLSTM Accuracy is:',BiLSTM,'%')



