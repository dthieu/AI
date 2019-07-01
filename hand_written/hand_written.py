import cv2
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers import regression
from tflearn.data_utils import to_categorical

BATCH_SIZE = 32
IMG_SIZE = 28
N_CLASSES = 4
LR = 0.001
N_EPOCHS = 50

tf.reset_default_graph()

network = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1]) #1

network = conv_2d(network, 32, 3, activation='relu') #2
network = max_pool_2d(network, 2) #3

network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)

network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)

network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)

network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)

network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)

network = fully_connected(network, 1024, activation='relu') #4
network = dropout(network, 0.8) #5

network = fully_connected(network, N_CLASSES, activation='softmax')#6
network = regression(network)

# Load model
model = tflearn.DNN(network) #7
model.load("hand_written_reg_model.tflearn")

img = cv2.imread("chuB.png", 0)
result = model.predict(img)
print(result)