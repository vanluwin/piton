from os import system
import numpy as np 
from keras.preprocessing import image
from keras.models import load_model 

# Carrega o classificador 
classifier = load_model('drinks_20e.h5')

system('clear')

for i in range(0, 4): 
    img = 'drink_{}.jpg'.format(i)
    
    # Carrega a imagem 
    test_image = image.load_img(
        'dataset/single_prediction/{}'.format(img),
        target_size = (64, 64)
    )

    # Transforma a imagem em um array
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)

    # Classifica 
    res = classifier.predict(test_image)

    # Mostra o resultado
    if res[0][0] > 0.95:
        prediction = 'Beer'
    elif res[0][1] > 0.95:
        prediction = 'Coffee'
    elif res[0][2] > 0.95:
        prediction = 'Tea'
    else:
        prediction = 'Wine'

    print('\tResultado da classificação para {}: {}'.format(img, prediction))