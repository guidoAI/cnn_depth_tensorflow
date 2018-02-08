# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 08:24:09 2018

Script that helps calculate the changes needed to apply the script to other input image sizes.

@author: guido
"""

import numpy as np

# original values:
#INPUT_HEIGHT = 228;
#INPUT_WIDTH = 304;
#OUTPUT_HEIGHT = 55;
#OUTPUT_WIDTH = 74;

INPUT_HEIGHT = np.round(375*0.75);
INPUT_WIDTH = np.round(1242*0.75); 
OUTPUT_HEIGHT = 40;
OUTPUT_WIDTH = 128;


height = np.asarray([0.0]*8)
width = np.asarray([0.0]*8)

i = 0;
height[i] = INPUT_HEIGHT;
width[i] = INPUT_WIDTH;
print('H,W = {},{}'.format(height[i], width[i]));

# calculate the sizes throughout Eigen:

# coarse1_conv = conv2d('coarse1', images, [11, 11, 3, 96], [96], [1, 4, 4, 1], padding='VALID', reuse=reuse, trainable=trainable)
i += 1;
filter_size = 11;
stride = 4;
height[i] = np.ceil((height[i-1] - filter_size + 1) / stride);
width[i] = np.ceil((width[i-1] - filter_size + 1) / stride);
print('H,W = {},{}'.format(height[i], width[i]));

# coarse1 = tf.nn.max_pool(coarse1_conv, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')
i += 1;
filter_size = 3;
stride = 2;
height[i] = np.ceil((height[i-1] - filter_size + 1) / stride);
width[i] = np.ceil((width[i-1] - filter_size + 1) / stride);
print('H,W = {},{}'.format(height[i], width[i]));

# coarse2_conv = conv2d('coarse2', coarse1, [5, 5, 96, 256], [256], [1, 1, 1, 1], padding='VALID', reuse=reuse, trainable=trainable)
i += 1;
filter_size = 5;
stride = 1;
height[i] = np.ceil((height[i-1] - filter_size + 1) / stride);
width[i] = np.ceil((width[i-1] - filter_size + 1) / stride);
print('H,W = {},{}'.format(height[i], width[i]));

# coarse2 = tf.nn.max_pool(coarse2_conv, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
# 'SAME' leads to a different formula:
i += 1;
filter_size = 3;
stride = 2;
height[i] = np.ceil(height[i-1] / stride);
width[i] = np.ceil(width[i-1] / stride);
print('H,W = {},{}'.format(height[i], width[i]));

# coarse3 = conv2d('coarse3', coarse2, [3, 3, 256, 384], [384], [1, 1, 1, 1], padding='VALID', reuse=reuse, trainable=trainable)
i += 1;
filter_size = 3;
stride = 1;
height[i] = np.ceil((height[i-1] - filter_size + 1) / stride);
width[i] = np.ceil((width[i-1] - filter_size + 1) / stride);
print('H,W = {},{}'.format(height[i], width[i]));

# coarse4 = conv2d('coarse4', coarse3, [3, 3, 384, 384], [384], [1, 1, 1, 1], padding='VALID', reuse=reuse, trainable=trainable)
i += 1;
filter_size = 3;
stride = 1;
height[i] = np.ceil((height[i-1] - filter_size + 1) / stride);
width[i] = np.ceil((width[i-1] - filter_size + 1) / stride);
print('H,W = {},{}'.format(height[i], width[i]));

# coarse5 = conv2d('coarse5', coarse4, [3, 3, 384, 256], [256], [1, 1, 1, 1], padding='VALID', reuse=reuse, trainable=trainable)
i += 1;
filter_size = 3;
stride = 1;
height[i] = np.ceil((height[i-1] - filter_size + 1) / stride);
width[i] = np.ceil((width[i-1] - filter_size + 1) / stride);
print('H,W = {},{}'.format(height[i], width[i]));

depth = 256;

# coarse6 = fc('coarse6', coarse5, [6*10*256, 4096], [4096], reuse=reuse, trainable=trainable)
# coarse7 = fc('coarse7', coarse6, [4096, 4070], [4070], reuse=reuse, trainable=trainable)
# coarse7_output = tf.reshape(coarse7, [-1, 55, 74, 1])

print('Number of weights in = {} x {} x {} = {}'.format(width[i], height[i], depth, width[i] * height[i] * depth))
print('Number of weights out = {}'.format(OUTPUT_HEIGHT*OUTPUT_WIDTH))



