"""Pip install keras, tensorflow(Make sure It is on version 2.1.0)
, pandas, matplotlib, and sklearn. Also make sure your python is on 3.7.6"""

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

def load_mnist_data(test_size_per):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(x_train.shape, y_train.shape)
    x_train, x_test, y_train, y_test = train_test_split(
    x_train, y_train, test_size=test_size_per)
    print(x_train.shape)
    return x_train, y_train, x_test, y_test 

def prepare_input(x_train, y_train, x_test, y_test):
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    return x_train, y_train, x_test, y_test



def setup_model(x_train, y_train, x_test, y_test ):
    batch_size = 128
    num_classes = 10
    epochs = 10

    input_shape = (28, 28, 1)
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5),activation='relu',input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])

    hist = model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_test))
    print("The model has successfully trained")

    score = model.evaluate(x_test, y_test, verbose=0)
    

    model.save('mnist.h5')
    print("Saving the model as mnist.h5")

    return score


summary_values = {}

#Takes increasingly more images to increase the accuracy
training_sets = [50, 100, 200, 500, 750, 1500, 2000, 5000, 6000, 8000, 10000]
j = 0
column_names = ["Amount of Training images", "Accuracy"]
df = pd.DataFrame(columns = column_names, index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

for test_per in training_sets:
    training_per = 1- (test_per/60000) #calculation is giving the test percentage
    print(training_per)
    x_train, y_train, x_test, y_test = load_mnist_data(training_per)
    print('Testing with train set size of: ', x_train.shape)
    x_train, y_train, x_test, y_test = prepare_input(x_train, y_train, x_test, y_test)
    score = setup_model(x_train, y_train, x_test, y_test)
    summary_values[x_train.shape] = {test_per/100, score[0], score[1]}
    i = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    new_data_list = [test_per, score[1]]
    df.loc[i[j]] = new_data_list
    j += 1


print(summary_values)


#Data Table
print(df)

#graph
#Line Graph
line_graph = df.plot(kind='line',x='Amount of Training images',y='Accuracy',color='red',  xticks=[0,500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000,8500,9000,9500,10000], 
        yticks=[0,500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000,8500,9000,9500,10000], figsize=(20,10))
line_graph.set(title="How changing the amount of Training data impacts the accuracy", xlabel="Amount of training images", ylabel="Accuracy")
plt.show()

