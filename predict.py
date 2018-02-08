# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 21:18:21 2018

@author: guido
"""

#encoding: utf-8

from datetime import datetime
from tensorflow.python.platform import gfile
import numpy as np
import tensorflow as tf
from dataset import DataSet
from dataset import output_predict
import model
import train_operation as op

# Option for prediction is to copy this script, change batch size to 1 and put an image we want in there.

MAX_STEPS = 10000000
LOG_DEVICE_PLACEMENT = False
BATCH_SIZE = 8
TRAIN_FILE = "test.csv"
COARSE_DIR = "coarse"
REFINE_DIR = "refine"


def predict():
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        dataset = DataSet(BATCH_SIZE)
        images, depths, invalid_depths = dataset.csv_inputs(TRAIN_FILE)
        keep_conv = tf.placeholder(tf.float32)
        keep_hidden = tf.placeholder(tf.float32)

        print("prediction");
        coarse = model.inference(images, keep_conv, trainable=False)
        logits = model.inference_refine(images, coarse, keep_conv, keep_hidden, trainable=False)

        loss = model.loss(logits, depths, invalid_depths)
        train_op = op.train(loss, global_step, BATCH_SIZE)
        init_op = tf.initialize_all_variables()

        # Session
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=LOG_DEVICE_PLACEMENT))
        sess.run(init_op)    

        # parameters
        coarse_params = {}
        refine_params = {}
        for variable in tf.all_variables():
            variable_name = variable.name
            print("parameter: %s" % (variable_name))
            if variable_name.find("/") < 0 or variable_name.count("/") != 1:
                continue
            if variable_name.find('coarse') >= 0:
                coarse_params[variable_name] = variable
            print("parameter: %s" %(variable_name))
            if variable_name.find('fine') >= 0:
                refine_params[variable_name] = variable

        # define saver
        print coarse_params
        saver_coarse = tf.train.Saver(coarse_params)
        saver_refine = tf.train.Saver(refine_params)
        # fine tune

        coarse_ckpt = tf.train.get_checkpoint_state(COARSE_DIR)
        if coarse_ckpt and coarse_ckpt.model_checkpoint_path:
            print("Pretrained coarse Model Loading.")
            saver_coarse.restore(sess, coarse_ckpt.model_checkpoint_path)
            print("Pretrained coarse Model Restored.")
        else:
            print("No Pretrained coarse Model.")

        refine_ckpt = tf.train.get_checkpoint_state(REFINE_DIR)
        if refine_ckpt and refine_ckpt.model_checkpoint_path:
            print("Pretrained refine Model Loading.")
            saver_refine.restore(sess, refine_ckpt.model_checkpoint_path)
            print("Pretrained refine Model Restored.")
        else:
            print("No Pretrained refine Model.")

        # train
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # run the network on the batch:
        _, loss_value, logits_val, images_val = sess.run([train_op, loss, logits, images], feed_dict={keep_conv: 0.8, keep_hidden: 0.5})


        print("%s: train loss %f" % (datetime.now(), loss_value))

        output_predict(logits_val, images_val, "predict")

        coord.request_stop()
        coord.join(threads)
        sess.close()


def main(argv=None):
    if not gfile.Exists(COARSE_DIR):
        gfile.MakeDirs(COARSE_DIR)
    if not gfile.Exists(REFINE_DIR):
        gfile.MakeDirs(REFINE_DIR)
    predict()


if __name__ == '__main__':
    tf.app.run()
