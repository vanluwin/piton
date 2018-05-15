from sys import argv
from os import system

from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

import numpy as np

classifier = load_model('results/models/catDog_20e.h5')

test_set = ImageDataGenerator().flow_from_directory(
    'dataset/test_set/',
    target_size = (128, 128),
    shuffle = False
)

for (img, label) in test_set:  
    res = classifier.predict(img)

    if(np.argmax(res)):
        prediction = 'Dog'
    else: 
        prediction = 'Cat'

    if(np.argmax(label)):
        should_be = 'Dog'
    else: 
        should_be = 'Cat'
    
    if(prediction != should_be):
        msg = 'Wrong moving to debug folder...'

        idx = (test_set.batch_index - 1) * test_set.batch_size
        file = test_set.filenames[idx] 

        if(should_be == 'Dog'):
            system('cp dataset/test_set/{} debug/dogs'.format(file))
        else: 
            system('cp dataset/test_set/{} debug/cats'.format(file))

    else:
        msg = '✓'

    
    print('Predition: {}, should be: {} \t {}'.format(prediction, should_be, msg))
