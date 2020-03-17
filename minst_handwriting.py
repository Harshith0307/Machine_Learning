import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

def load_minst_data(test_size_per):
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
training_sets = [100, 200, 500, 750, 1500, 2000, 5000, 6000, 10000, 20000, 40000, 50000]

for test_per in training_sets:
    training_per = 1- (test_per/60000) #calculation is giving the test percentage
    print(training_per)
    x_train, y_train, x_test, y_test = load_minst_data(training_per)
    print('Testing with train set size of: ', x_train.shape)
    x_train, y_train, x_test, y_test = prepare_input(x_train, y_train, x_test, y_test)
    score = setup_model(x_train, y_train, x_test, y_test)
    summary_values[x_train.shape] = {test_per/100, score[0], score[1]}

print(summary_values)