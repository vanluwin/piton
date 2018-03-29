import numpy as np 
import sys
from keras.preprocessing import image
from keras.models import load_model 

if __name__ == 'main':
    img = args[1]
    
    # Carrega a imagem 
    test_image = image.load_img(
        img,
        target_size = (64, 64)
    )

    # Transforma a imagem em um array
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)

    # Carrega o classificador 
    classifier = load_model('drinks_20e.h5')

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


    print('Resultado da classificação para {}: {}'.format(img, prediction))