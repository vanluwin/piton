from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import SGD
from keras.metrics import categorical_accuracy

# Definição do modelo
classifier = Sequential()

# Passo 1 - Convolução
classifier.add(Conv2D(32, (3, 3), input_shape = (128, 128, 3), padding='same', activation = 'relu'))
classifier.add(Conv2D(32, (3, 3), activation='relu'))

# Passo 2 - Pooling 
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Passo 3 - Dropout
classifier.add(Dropout(0.25))

# Segunda camada de convolução 
classifier.add(Conv2D(64, (3, 3),  padding='same', activation = 'relu'))
classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.25))

# Terceira camada de convolução
classifier.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.25))

# Flattening
classifier.add(Flatten())

# Camda totalmente conectada 
classifier.add(Dense(512, activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units = 4, activation = 'sigmoid'))

gld = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# Compilando a CNN
classifier.compile(
    optimizer = gld, 
    loss = 'binary_crossentropy', 
    #metrics = ['accuracy'])
    metrics=['accuracy', categorical_accuracy]
)

# 'Ajustando a CNN as imagens 
from keras.preprocessing.image import ImageDataGenerator

# Augmentation configuration
train_data = ImageDataGenerator(
    rescale = 1./255,
    rotation_range=180.,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True
)

test_data = ImageDataGenerator(
    rescale = 1./255
)

training_set = train_data.flow_from_directory(
    'dataset/traning_set',
    target_size = (128, 128),
    batch_size = 32,
    class_mode = 'categorical'
)

test_set = test_data.flow_from_directory(
    'dataset/test_set',
    target_size = (128, 128),
    batch_size = 32,
    class_mode = 'categorical'
)

# Fit the model
history = classifier.fit_generator(
    training_set,
    steps_per_epoch = 4000,
    epochs = 5,
    validation_data = test_set,
    validation_steps = 2000
)

# Salvar o modelo
classifier.save('drinks.h5')

import matplotlib.pyplot as plt

# Loss Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['loss'],'r',linewidth=3.0)
plt.plot(history.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)
 
# Accuracy Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['acc'],'r',linewidth=3.0)
plt.plot(history.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)