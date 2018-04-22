import numpy as np
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator, load_img

# importing the convolitional layers
vgg_conv = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

train_dir = './dataset/training_set'
validation_dir = './dataset/test_set'
 
nTrain = 8000
nVal = 2000

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 32

train_features = np.zeros(shape=(nTrain, 7, 7, 512))
train_labels = np.zeros(shape=(nTrain,3))

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

i = 0
for inputs_batch, labels_batch in train_generator:
    features_batch = vgg_conv.predict(inputs_batch)
    train_features[i * batch_size : (i + 1) * batch_size] = features_batch
    train_labels[i * batch_size : (i + 1) * batch_size] = labels_batch
    i += 1
    if i * batch_size >= nTrain:
        break
        
train_features = np.reshape(train_features, (nTrain, 7 * 7 * 512))

validation_features = np.zeros(shape=(nVal, 7, 7, 512))
validation_labels = np.zeros(shape=(nVal,3))

validation_generator = datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

i = 0
for inputs_batch, labels_batch in validation_generator:
    features_batch = vgg_conv.predict(inputs_batch)
    validation_features[i * batch_size : (i + 1) * batch_size] = features_batch
    validation_labels[i * batch_size : (i + 1) * batch_size] = labels_batch
    i += 1
    if i * batch_size >= nVal:
        break

validation_features = np.reshape(validation_features, (nVal, 7 * 7 * 512))

# My own model
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

model = Sequential()

model.add(Dense(512, activation='relu', input_dim=7 * 7 * 512))
model.add(Dropout(0.5))
""" model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5)) """
model.add(Dense(2, activation='softmax'))

model.compile(
    optimizer=RMSprop(lr=2e-4),
    loss='categorical_crossentropy',
    metrics=['acc']
)

history = model.fit(
    train_features,
    train_labels,
    epochs=20,
    batch_size=batch_size,
    validation_data=(validation_features,validation_labels)
)



