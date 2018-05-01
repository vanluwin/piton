import numpy as np
from sys import argv
from keras.preprocessing import image
from keras.models import load_model

img = argv[1]

# Loads the image
test_image = image.load_img(
    img,
    target_size = (28, 28, 1)
)

# Transform the img into a array
test_image = image.img_to_array(test_image)
test_image = np.resize(test_image, (28, 28, 1))
test_image = np.expand_dims(test_image, axis = 0)

test_image = test_image.astype('float32')

test_image /= 255

# Loads the classifier model
classifier = load_model('results/models/cnn_100e.h5')

# Predicts the label
res = classifier.predict_classes(test_image)

print(res)
