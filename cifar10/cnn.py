from keras.datasets import cifar10

from keras.regularizers import l2

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D
from keras.initializers import he_normal

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.optimizers import SGD

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

        self.dropout = 0.5
        self.weight_decay = 0.0001

        # Retrieves the data
        self.get_data()

        # Defnes the optimizer and loss function
        optimizer = SGD(lr=.1, momentum=0.9, nesterov=True)
        loss = ['categorical_crossentropy']

        # Build and compile the NN
        self.cnn = self.build_cnn()
        self.cnn.compile(
            loss = loss,
            optimizer = optimizer,
            metrics = ['accuracy']
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
        # VGG 19 Model
        model = Sequential()
        
        # Block 1
        model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(self.weight_decay), kernel_initializer=he_normal(), name='block1_conv1', input_shape=(self.img_rows, self.img_cols, self.channels)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(self.weight_decay), kernel_initializer=he_normal(), name='block1_conv2'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

        # Block 2
        model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(self.weight_decay), kernel_initializer=he_normal(), name='block2_conv1'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(self.weight_decay), kernel_initializer=he_normal(), name='block2_conv2'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

        # Block 3
        model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(self.weight_decay), kernel_initializer=he_normal(), name='block3_conv1'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(self.weight_decay), kernel_initializer=he_normal(), name='block3_conv2'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(self.weight_decay), kernel_initializer=he_normal(), name='block3_conv3'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(self.weight_decay), kernel_initializer=he_normal(), name='block3_conv4'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

        # Block 4
        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=l2(self.weight_decay), kernel_initializer=he_normal(), name='block4_conv1'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=l2(self.weight_decay), kernel_initializer=he_normal(), name='block4_conv2'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=l2(self.weight_decay), kernel_initializer=he_normal(), name='block4_conv3'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=l2(self.weight_decay), kernel_initializer=he_normal(), name='block4_conv4'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

        # Block 5
        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=l2(self.weight_decay), kernel_initializer=he_normal(), name='block5_conv1'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=l2(self.weight_decay), kernel_initializer=he_normal(), name='block5_conv2'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=l2(self.weight_decay), kernel_initializer=he_normal(), name='block5_conv3'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=l2(self.weight_decay), kernel_initializer=he_normal(), name='block5_conv4'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

        # model modification for cifar-10
        model.add(Flatten(name='flatten'))
        model.add(Dense(4096, use_bias = True, kernel_regularizer=l2(self.weight_decay), kernel_initializer=he_normal(), name='fc_cifa10'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(self.dropout))
        model.add(Dense(4096, kernel_regularizer=l2(self.weight_decay), kernel_initializer=he_normal(), name='fc2'))  
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(self.dropout))      
        model.add(Dense(self.num_classes, kernel_regularizer=l2(self.weight_decay), kernel_initializer=he_normal(), name='predictions_cifa10'))        
        model.add(BatchNormalization())
        model.add(Activation('softmax'))

        # load pretrained weight from VGG19 by name      
        model.load_weights('results/models/vgg19_weights.h5', by_name=True)

        if(self.debug):
            model.summary()

        return model

    def train(self, epochs=1, batch_size=1):
        self.epochs = epochs

        datagen = ImageDataGenerator(
            horizontal_flip=True,
            width_shift_range=0.125,
            height_shift_range=0.125,
            fill_mode='constant',
            cval=0.
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
        
        end = time.time()

        self.cnn.evaluate(self.test_data, self.test_labels)

        self.history = history

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
    cnn.train(epochs=100, batch_size=128)
    cnn.save_model()
    cnn.save_plots()
    cnn.save_logs()