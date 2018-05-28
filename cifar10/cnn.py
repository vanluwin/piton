from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

from matplotlib import pyplot as plt

import numpy as np

import time

class CNN:
    def __init__(self):
        # Debug Mode
        self.debug = 0

        # Input shape
        self.img_rows = 32
        self.img_cols = 32
        self.channels = 3
        self.input_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = 10

        self.get_data()

        # Defnes the optimizer and loss function
        optimizer = 'rmsprop'
        loss = ['categorical_crossentropy']

        # Defining the activation functions
        self.convActivation = 'relu'
        self.denseActivation = 'relu'

        # Build and compile the NN
        self.cnn = self.build_cnn()
        self.cnn.compile(
            loss = loss,
            optimizer = optimizer,
            metrics = ['acc']
        )
    
    def get_data(self):
        (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

        train_data = train_images.reshape(train_images.shape[0], self.img_rows, self.img_cols, self.channels)
        test_data = test_images.reshape(test_images.shape[0], self.img_rows, self.img_cols, self.channels)

        # Change to float datatype
        train_data = train_data.astype('float32')
        test_data = test_data.astype('float32')

        # Scale the data to lie between 0 to 1
        train_data /= 255
        test_data /= 255

        # Change the labels from integer to categorical data
        train_labels_one_hot = to_categorical(train_labels)
        test_labels_one_hot = to_categorical(test_labels)

        self.train_data = train_data
        self.train_labels = train_labels_one_hot

        self.test_data = test_data
        self.test_labels = test_labels_one_hot

    def build_cnn(self):
        # CNN model
        model = Sequential()
        
        model.add(Conv2D(32, (3, 3), padding='same', activation=self.convActivation, input_shape=self.input_shape))
        model.add(Conv2D(32, (3, 3), activation=self.convActivation))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same', activation=self.convActivation))
        model.add(Conv2D(64, (3, 3), activation=self.convActivation))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same', activation=self.convActivation))
        model.add(Conv2D(64, (3, 3), activation=self.convActivation))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512, activation=self.denseActivation))
        model.add(Dropout(0.5))
        model.add(Dense(512, activation=self.denseActivation))
        model.add(Dropout(0.5))
        model.add(Dense(512, activation=self.denseActivation))
        model.add(Dropout(0.5))

        # Normalizes the network input weights between 0 and 1
        model.add(BatchNormalization())

        model.add(Dense(self.num_classes, activation='tanh'))

        if(self.debug):
            model.summary()

        return model

    def train(self, epochs=1, batch_size=1):
        self.epochs = epochs

        datagen = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=False
        )

        training_set = datagen.flow(
            self.train_data,
            self.train_labels,
            batch_size=batch_size
        )

        start = time.time()

        history = self.cnn.fit_generator(
            training_set,
            steps_per_epoch = int(np.ceil(self.train_data.shape[0] / float(batch_size))),
            epochs = epochs,
            validation_data = (self.test_data, self.test_labels),
            validation_steps = int(np.ceil(self.test_data.shape[0] / float(batch_size)))
        )

        self.history = history

        end = time.time()

        print("Model took %0.2f seconds to train"%(end - start))

    def save_plots(self):
        # Plot the curves
        plt.figure(figsize=[8,6])
        plt.plot(self.history.history['loss'],'r',linewidth=3.0)
        plt.plot(self.history.history['val_loss'],'b',linewidth=3.0)
        plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
        plt.xlabel('Epochs ',fontsize=16)
        plt.ylabel('Loss',fontsize=16)
        plt.title('Loss Curves',fontsize=16)

        plt.savefig('results/img/cnn_loss_{}e.png'.format(self.epochs))

        plt.figure(figsize=[8,6])
        plt.plot(self.history.history['acc'],'r',linewidth=3.0)
        plt.plot(self.history.history['val_acc'],'b',linewidth=3.0)
        plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
        plt.xlabel('Epochs ',fontsize=16)
        plt.ylabel('Accuracy',fontsize=16)
        plt.title('Accuracy Curves',fontsize=16)

        plt.savefig('results/img/cnn_acc_{}e.png'.format(self.epochs))

        plt.show()

    def save_logs(self):
        loss_history = np.array(self.history.history['loss'])
        np.savetxt(
            'results/logs/log_loss_{}e.txt'.format(self.epochs), 
            loss_history, 
            fmt='%d'
        )

        acc_history = np.array(self.history.history['acc'])
        np.savetxt(
            'results/logs/log_acc_{}e.txt'.format(self.epochs),
            acc_history,
            fmt='%d'
        )

    def save_model(self):
        self.cnn.save('results/models/cnn_{}e.h5'.format(self.epochs))

if __name__ == '__main__':
    cnn = CNN()
    cnn.train(epochs=10, batch_size=256)
    cnn.save_model()
    cnn.save_plots()
    cnn.save_logs()