#encoding: utf-8

from datetime import datetime
from tensorflow.python.platform import gfile
import numpy as np
import tensorflow as tf
from dataset import DataSet
from dataset import output_predict
import model
import train_operation as op
import pickle

# Option for prediction is to copy this script, change batch size to 1 and put an image we want in there.

MAX_STEPS = 1500;
ITERATIONS = 1000;
LOG_DEVICE_PLACEMENT = False
BATCH_SIZE = 8
TRAIN_FILE = "train_new.csv"
# Validation adaptations initially from https://raw.githubusercontent.com/ImaduddinAMajid/cnn_depth_tensorflow/4aa1846ec56efcbfc7acc55f93471366c65e152a/task.py
VALIDATION_SET = True;
VALIDATION_FILE = "validate_new.csv" 
COARSE_DIR = "coarse"
REFINE_DIR = "refine"

REFINE_TRAIN = True
FINE_TUNE = False

def train():
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        dataset = DataSet(BATCH_SIZE)
        images, depths, invalid_depths = dataset.csv_inputs(TRAIN_FILE)
        keep_conv = tf.placeholder(tf.float32)
        keep_hidden = tf.placeholder(tf.float32)
        
        if(VALIDATION_SET):
            val_dataset = DataSet(BATCH_SIZE)        
            val_images, val_depths, val_invalid_depths = val_dataset.csv_inputs(VALIDATION_FILE)        
#            val_keep_conv = tf.placeholder(tf.float32)
#            val_keep_hidden = tf.placeholder(tf.float32)
        
        if REFINE_TRAIN:
            print("refine train.")
            coarse = model.inference(images, trainable=False) 
            logits = model.inference_refine(images, coarse, keep_conv) 
            
            if(VALIDATION_SET):
                # why was there no val_logits here?
                val_coarse = model.inference(val_images, reuse= True, trainable=False);
                val_logits = model.inference_refine(val_images, val_coarse, keep_conv, reuse=True, trainable=False)
        else:
            print("coarse train.")
            logits = model.inference(images, keep_conv, keep_hidden)  # why these parameters?          
            if(VALIDATION_SET):
                val_logits = model.inference(val_images, reuse=True, trainable=False)
        
        loss = model.loss(logits, depths, invalid_depths)            
        train_op = op.train(loss, global_step, BATCH_SIZE)        

        if(VALIDATION_SET):
            val_loss = model.loss(val_logits, val_depths, val_invalid_depths)
            val_op = op.train(val_loss, global_step, BATCH_SIZE)

        init_op = tf.initialize_all_variables()

        # Session
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=LOG_DEVICE_PLACEMENT))
        sess.run(init_op)    

        # parameters
        coarse_params = {}
        refine_params = {}
        if REFINE_TRAIN:
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
        else:
            for variable in tf.trainable_variables():
                variable_name = variable.name
                print("parameter: %s" %(variable_name))
                if variable_name.find("/") < 0 or variable_name.count("/") != 1:
                    continue
                if variable_name.find('coarse') >= 0:
                    coarse_params[variable_name] = variable
                if variable_name.find('fine') >= 0:
                    refine_params[variable_name] = variable
        # define saver
        print coarse_params
        saver_coarse = tf.train.Saver(coarse_params)
        if REFINE_TRAIN:
            saver_refine = tf.train.Saver(refine_params)
        # fine tune
        if REFINE_TRAIN: 
            coarse_ckpt = tf.train.get_checkpoint_state(COARSE_DIR)
            if coarse_ckpt and coarse_ckpt.model_checkpoint_path:
                print("Pretrained coarse Model Loading.")
                saver_coarse.restore(sess, coarse_ckpt.model_checkpoint_path)
                print("Pretrained coarse Model Restored.")
            else:
                print("No Pretrained coarse Model.")
            if FINE_TUNE:
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
        losses = np.zeros([(MAX_STEPS*ITERATIONS/10)+1, 2]);
        loss_index = 0;
        # later we may use the validation performance as an additional stopping criterion:
        for step in xrange(MAX_STEPS):
            index = 0
            for i in xrange(ITERATIONS):
                _, loss_value, logits_val, images_val = sess.run([train_op, loss, logits, images], feed_dict={keep_conv: 0.8, keep_hidden: 0.5})
                
                if index % 10 == 0:
                    # print("%s: %d[epoch]: %d[iteration]: train loss %f" % (datetime.now(), step, index, loss_value))
                    if(VALIDATION_SET):                
                        _, val_loss_value, val_logits_val, val_images_val = sess.run([val_op, val_loss, val_logits, val_images], feed_dict={keep_conv: 0.8, keep_hidden: 0.5})
                        losses[loss_index, 1] = val_loss_value;
                        print("%s: %d[epoch]: %d[iteration]: train loss %f validation loss %f" % (datetime.now(), step, index, loss_value, val_loss_value))
                    else:
                        print("%s: %d[epoch]: %d[iteration]: train loss %f" % (datetime.now(), step, index, loss_value))      
                        
                    losses[loss_index, 0] = loss_value;
                    loss_index += 1;
                    
                    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                    
                    with open('losses.pkl', 'w') as f:
                        pickle.dump(losses, f)
                    
                if index % 500 == 0:
                    if REFINE_TRAIN:
                        output_predict(logits_val, images_val, "data/predict_refine_%05d_%05d" % (step, i))
                    else:
                        output_predict(logits_val, images_val, "data/predict_%05d_%05d" % (step, i))
                index += 1

            if step % 5 == 0 or (step * 1) == MAX_STEPS:
                if REFINE_TRAIN:
                    refine_checkpoint_path = REFINE_DIR + '/model.ckpt'
                    saver_refine.save(sess, refine_checkpoint_path, global_step=step)
                else:
                    coarse_checkpoint_path = COARSE_DIR + '/model.ckpt'
                    saver_coarse.save(sess, coarse_checkpoint_path, global_step=step)
        coord.request_stop()
        coord.join(threads)
        sess.close()


def main(argv=None):
    if not gfile.Exists(COARSE_DIR):
        gfile.MakeDirs(COARSE_DIR)
    if not gfile.Exists(REFINE_DIR):
        gfile.MakeDirs(REFINE_DIR)
    train()


if __name__ == '__main__':
    tf.app.run()
