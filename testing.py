import h5py
import time
import os
import numpy as np
import scipy
import matplotlib.pyplot as plt
import tensorflow as tf
from network.loss import *
from network.quickNAT import quick_nat
from preprocessing.data_utils import  *
from scipy import ndimage
from sklearn.model_selection import train_test_split
import scipy.misc

test_filepath = '/home/caelyn/imdbTesting.mat'


# Load test images
test_data, test_labels = read_dataset(filepath=test_filepath, file_for="train")
test_data = test_data.reshape(test_data.shape[0], test_data.shape[2], test_data.shape[3], test_data.shape[1])# (NumOfData, height, weight, channel=1 for grayscale data

#scipy.misc.imsave('outfile.jpg', test_data[150].reshape(256,256))


# Load saved model
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.import_meta_graph("/home/caelyn/saved_models_wholebrain/model.ckpt-175.meta")
saver.restore(sess, tf.train.latest_checkpoint("/home/caelyn/saved_models_wholebrain"))
X = tf.get_collection("inputs")[0]
mode = tf.get_collection("inputs")[1]
pred1 = tf.get_collection("outputs")[0]
print(test_data.shape)

test_image = centeredCrop((test_data[150]).reshape(1,256,256,1),256,256)
pred = sess.run(pred1, feed_dict={X: test_image , mode: False})
scipy.misc.imsave('outfile.jpg', np.argmax(pred[0],2))

