from os import system
from sys import argv

import tensorflow as tf 
with tf.device('/device:GPU:0'):
    system('python {}'.format(argv[1]))


