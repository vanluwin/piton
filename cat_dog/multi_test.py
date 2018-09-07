from os import system
import numpy as np 
from keras.preprocessing import image
from keras.models import load_model 

# Carrega o classificador 
classifier = load_model('results/models/catDog_20e.h5')

system('clear')

for i in range(1, 8): 
    img = 'cat_or_dog_{}.jpg'.format(i)
    
    # Carrega a imagem 
    test_image = image.load_img(
        'dataset/single_prediction/{}'.format(img),
        target_size = (128, 128)
    )

    # Transforma a imagem em um array
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)

    # Classifica 
    res = classifier.predict(test_image)

    max_index = np.argmax(res)

    if(max_index):
        prediction = 'Dog'
    else: 
        prediction = 'Cat'

    print('Prediction for {}: {}'.format(img, prediction))
