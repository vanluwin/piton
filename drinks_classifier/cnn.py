from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import SGD

# Definição do modelo
classifier = Sequential()

# Passo 1 - Convolução
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Passo 2 - Pooling 
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Segunda camada de convolução 
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Passo 3 - Flattening
classifier.add(Flatten())

# Passo 4 - Camda totalmente conectada 
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 4, activation = 'sigmoid'))

gld = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# Compilando a CNN
classifier.compile(optimizer = gld, loss = 'binary_crossentropy', metrics = ['accuracy'])

# 'Ajustando a CNN as imagens 
from keras.preprocessing.image import ImageDataGenerator

train_data = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True
)

test_data = ImageDataGenerator(
    rescale = 1./255
)

training_set = train_data.flow_from_directory(
    'dataset/traning_set',
    target_size = (64, 64),
    batch_size = 32,
    class_mode = 'categorical'
)

test_set = test_data.flow_from_directory(
    'dataset/test_set',
    target_size = (64, 64),
    batch_size = 32,
    class_mode = 'categorical'
)

classifier.fit_generator(
    training_set,
    steps_per_epoch = 4000,
    epochs = 5,
    validation_data = test_set,
    validation_steps = 1000
)

# Salvar o modelo
classifier.save('drinks.h5')