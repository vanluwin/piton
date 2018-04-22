import numpy as np 
from sys import argv
from os import system
from keras.preprocessing import image
from keras.models import load_model

img = argv[1]

test_image = image.load_img(
    img,
    target_size = (128, 128)
)

# Transforma a imagem em um array
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

# Carrega o classificador 
classifier = load_model('results/models/catDog_20e.h5')

# Classifica 
res = classifier.predict(test_image)

max_index = np.argmax(res)

if(max_index):
    prediction = 'Dog'
else: 
    prediction = 'Cat'

print('Prediction for {}: {}'.format(img, prediction))
