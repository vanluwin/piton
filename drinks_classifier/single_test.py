import numpy as np 
from sys import argv
from os import system
from keras.preprocessing import image
from keras.models import load_model

img = argv[1]

# Carrega a imagem 
test_image = image.load_img(
    img,
    target_size = (64, 64)
)

# Transforma a imagem em um array
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

# Carrega o classificador 
classifier = load_model('drinks_100e.h5')

# Classifica 
res = classifier.predict(test_image)

"""
# Mostra o resultado
if res[0][0] > 0.95:
    prediction = 'Beer'
elif res[0][1] > 0.95:
    prediction = 'Coffee'
elif res[0][2] > 0.95:
    prediction = 'Tea'
else:
    prediction = 'Wine'
"""
system('clear')
print(res)
#print("{}\nResultado da classificação para '{}': {} \n".format(res, img, prediction))