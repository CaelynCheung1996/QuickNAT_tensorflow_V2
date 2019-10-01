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


# Define the path to the dataset
train_filepath = '/home/caelyn/imdbTraining.mat' 

# load dataset
train_data, train_labels = read_dataset(filepath=train_filepath, file_for="train")
# (NumOfSlice, height, weight, channel=1 for grayscale data)
train_data = train_data.reshape(train_data.shape[0], train_data.shape[2], train_data.shape[3], train_data.shape[1])
print(train_data.shape)
#  preprocessing data
train_labels = train_labels[:, 0, :, :] #squeeze train_labels to 3D (channel eliminated)
train_data, train_labels = remove_back_pixels(train_data, train_labels) # remove the slices with all zero values 

# split data for train and test
X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.3, random_state=42)

# Training Configuration 
num_classes = 5
epochs = 15
batch_size = 8
n_train = X_train.shape[0]
n_valid = X_test.shape[0]
train_total_batch = int(n_train / batch_size)  
val_total_batch = int(n_valid / batch_size)  
train_logs_path = "logs/train"
val_logs_path = "logs/val"
learning = 0.0001
momentum = 0.9
nestrov = True
ckdir = "saved_models_wholebrain/model.ckpt"
config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)  # GPU Configuration
# 
global_index = 0
list_index = 0
len_entries = 0

def train(restore=False, testing=False):
    # log directory of graphs
    current_time = time.strftime("%m/%d/%H/%M/%S")
    train_logdir = os.path.join(train_logs_path, "train_deep_sdnet", current_time)
    test_logdir = os.path.join(val_logs_path, "test_deep_sdnet", current_time)

    # train_dataset
    X = tf.placeholder(tf.float32, shape=[None, 256, 256, 1], name="X") # Shape of size of centercrop and single slice indicated by dim 1
    y = tf.placeholder(tf.float32, shape=[None, 256, 256, 5], name="y") # Shape of size of centercrop and number of classes
    mode = tf.placeholder(tf.bool, name="mode")
    pred1 = quick_nat(X, mode, 5) # Change the constant variable in definition of quicknat function
    tf.add_to_collection("inputs", X)
    tf.add_to_collection("inputs", mode)
    tf.add_to_collection("outputs", pred1)
    pred_prob = tf.nn.softmax(pred1, 3) # (logits, axis)

   # Assign weights to the different classes
    with tf.name_scope('loss'):
        loss_op_1 = weighted_cross_entropy_plus_dice(pred1, y, )
        loss_op_1 = tf.Print(loss_op_1, [loss_op_1], message="Loss step1: ")
        tf.summary.scalar("Loss", loss_op_1)

    # For each class, computer dice loss
    with tf.name_scope('loss_background'):
        loss_op_back = (-dice_coef_0(pred1, y))
        tf.summary.scalar("Loss background", loss_op_back)

    with tf.name_scope('dice_loss_whitematter'):
        loss_wm = (-dice_coef_1(pred1, y))
        tf.summary.scalar("white matter", loss_wm)

    with tf.name_scope('dice_loss_greymatter'):
        loss_gm = (-dice_coef_2(pred1, y))
        tf.summary.scalar("Loss greymatter", loss_gm)

    with tf.name_scope('dice_loss_csf'):
        loss_csf = (-dice_coef_3(pred1, y))
        tf.summary.scalar("Loss csf", loss_csf)

    with tf.name_scope('dice_loss_vdc'):
        loss_vdc = (-dice_coef_4(pred1, y))
        tf.summary.scalar("Loss vdc", loss_vdc)

    # SGD and Accuracy
    with tf.name_scope('SGD'):
        train_op_1 = make_train_op(pred1, y, learning, momentum, nestrov, 5)

    with tf.name_scope('Accuracy'):
        acc = tf.equal(tf.argmax(y, 3), tf.argmax(pred1, 3))
        acc = tf.reduce_mean(tf.cast(acc, tf.float32))
        accuracy = tf.Print(acc, [acc], message="accuracy: ")
        tf.summary.scalar('Accuracy', acc)

    # For each class, compute dice coefficient
    with tf.name_scope('DiceCoef_background'):
        dice_coef_bg = dice_coef_0(pred_prob, y)
        dice_coef_bg = tf.Print(dice_coef_bg, [dice_coef_bg], message="DiceCoef_background: ")
        tf.summary.scalar('DiceCoef_background', dice_coef_bg)

    with tf.name_scope('DiceCoef_whitematter'):
        dice_coef_wm = dice_coef_1(pred_prob, y)
        dice_coef_wm = tf.Print(dice_coef_wm, [dice_coef_wm], message="DiceCoef_whitematter")
        tf.summary.scalar('DiceCoef_whitematter', dice_coef_wm)

    with tf.name_scope('DiceCoef_greymatter'):
        dice_coef_gm = dice_coef_2(pred_prob, y)
        dice_coef_gm = tf.Print(dice_coef_gm, [dice_coef_gm], message="DiceCoef_greymatter")
        tf.summary.scalar('DiceCoef_greymatter', dice_coef_gm)

    with tf.name_scope('DiceCoef_csf'):
        dice_coef_csf = dice_coef_3(pred_prob, y)
        dice_coef_csf = tf.Print(dice_coef_csf, [dice_coef_csf], message="DiceCoef_csf")
        tf.summary.scalar('DiceCoef_csf', dice_coef_csf)

    with tf.name_scope('DiceCoef_vdc'):
        dice_coef_vdc = dice_coef_4(pred_prob, y)
        dice_coef_vdc = tf.Print(dice_coef_vdc, [dice_coef_vdc], message="DiceCoef_vdc")
        tf.summary.scalar('DiceCoef_vdc', dice_coef_vdc)

    brain_prediction = tf.reshape(tf.cast(tf.argmax(pred1, axis=3), tf.float32), shape=[batch_size, 256, 256, 1])
    ground_truth = tf.reshape(tf.cast(tf.argmax(y, axis=3), tf.float32), shape=[batch_size, 256, 256, 1])

    TP = tf.count_nonzero(brain_prediction * ground_truth, dtype=tf.float32)
    TN = tf.count_nonzero((brain_prediction - 1) * (ground_truth - 1), dtype=tf.float32)
    FP = tf.count_nonzero(brain_prediction * (ground_truth - 1), dtype=tf.float32)
    FN = tf.count_nonzero((brain_prediction - 1) * ground_truth, dtype=tf.float32)

    with tf.name_scope('precision'):
        precision = TP / (TP + FP)
        tf.Print(precision, [precision], message="Precision: ")

    with tf.name_scope('recall'):
        recall = TP / (TP + FN)
        tf.Print(recall, [recall], message="Recall: ")

    with tf.name_scope('FPR'):
        fallout = FP / (FP + TN)
        tf.summary.scalar('False Positive Rate', fallout)

    with tf.name_scope('F1_score'):
        f1_score = (2 * (precision * recall)) / (precision + recall)
        tf.summary.scalar('F1 score', f1_score)

    tf.summary.image("Ground Truth", ground_truth, max_outputs=3) # max_outputs=3 is fixed
    tf.summary.image("Predicted Image", brain_prediction, max_outputs=3)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state("./saved_models_wholebrain")
    summary_op = tf.summary.merge_all()
    init = tf.global_variables_initializer()

    with tf.Session(config=config) as sess:
        # create log writer object
        train_summary_writer = tf.summary.FileWriter(train_logdir, graph=sess.graph)
        test_summary_writer = tf.summary.FileWriter(test_logdir)
        global_step = tf.train.get_global_step(sess.graph)
        sess.run(init)

        for epoch in range(epochs):
            print(epoch, "/", epochs)
            step_count_train = int(n_train / batch_size)
            for i in range(step_count_train):
                X_batch_op, y_batch_op = data_generator(batch_size, X_train, y_train,num_classes,test_data= True).__next__()
                print("-----------training---------------")
                print("epoch ", epoch, " step ", i, "/", step_count_train)
                _, step_loss_1, step_summary, global_step_value = sess.run(
                    [train_op_1, loss_op_1, summary_op, global_step],
                    feed_dict={X: X_batch_op,
                               y: y_batch_op,
                               mode: True})
                # write log
                train_summary_writer.add_summary(step_summary, (epoch))
                if (i + 1) % 5 == 0:
                    saver.save(sess, ckdir, global_step=(i + 1))
                    print("Model saved in file: %s" % ckdir)
            step_count_valid = int(n_valid / batch_size)

            for i in range(step_count_valid):
                X_valid_op, y_valid_op = data_generator(batch_size, X_test, y_test,num_classes,test_data=True).__next__()
                print("-----------validation-------------")
                print("epoch ", epoch)
                print(i, "/", step_count_valid)
                _, step_loss_1, step_summary = sess.run(
                    [train_op_1, loss_op_1, summary_op],
                    feed_dict={X: X_valid_op,
                               y: y_valid_op,
                               mode: False})
                test_summary_writer.add_summary(step_summary, (epoch))

        train_summary_writer.close()
        test_summary_writer.close()
train()
