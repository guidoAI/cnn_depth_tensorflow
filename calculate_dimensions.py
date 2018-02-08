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

INPUT_HEIGHT = 168;
INPUT_WIDTH = 520;
OUTPUT_HEIGHT = 40;
OUTPUT_WIDTH = 128;

print('TARGET OUTPUT = {} x {} = {} pixels'.format(OUTPUT_HEIGHT, OUTPUT_WIDTH, OUTPUT_HEIGHT * OUTPUT_WIDTH));

height = np.asarray([0.0]*8)
width = np.asarray([0.0]*8)

print('COARSE');

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
# SAME:
height[i] = np.ceil(height[i-1] / stride);
width[i] = np.ceil(width[i-1] / stride);
# VALID:
#height[i] = np.ceil((height[i-1] - filter_size + 1) / stride);
#width[i] = np.ceil((width[i-1] - filter_size + 1) / stride);
print('H,W = {},{}'.format(height[i], width[i]));

depth = 256;

# coarse6 = fc('coarse6', coarse5, [6*10*256, 4096], [4096], reuse=reuse, trainable=trainable)
# coarse7 = fc('coarse7', coarse6, [4096, 4070], [4070], reuse=reuse, trainable=trainable)
# coarse7_output = tf.reshape(coarse7, [-1, 55, 74, 1])

print('Number of weights in = {} x {} x {} = {}'.format(width[i], height[i], depth, width[i] * height[i] * depth))
print('Number of weights out = {}'.format(OUTPUT_HEIGHT*OUTPUT_WIDTH))



print('FINE');

i = 0;
height[i] = INPUT_HEIGHT;
width[i] = INPUT_WIDTH;
print('H,W = {},{}'.format(height[i], width[i]));

#fine1_conv = conv2d('fine1', images, [9, 9, 3, 63], [63], [1, 2, 2, 1], padding='VALID', reuse=reuse, trainable=trainable)
i += 1;
filter_size = 9;
stride = 2;
height[i] = np.ceil((height[i-1] - filter_size + 1) / stride);
width[i] = np.ceil((width[i-1] - filter_size + 1) / stride);
print('H,W = {},{}'.format(height[i], width[i]));

#fine1 = tf.nn.max_pool(fine1_conv, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='fine_pool1')
i += 1;
filter_size = 3;
stride = 2;
height[i] = np.ceil(height[i-1] / stride);
width[i] = np.ceil(width[i-1] / stride);
print('H,W = {},{}'.format(height[i], width[i]));

#fine1_dropout = tf.nn.dropout(fine1, keep_conv)
#fine2 = tf.concat([fine1_dropout, coarse7_output], 3) # how are they conacetanted?

#fine3 = conv2d('fine3', fine2, [5, 5, 64, 64], [64], [1, 1, 1, 1], padding='SAME', reuse=reuse, trainable=trainable)
i += 1;
filter_size = 5;
stride = 1;
height[i] = np.ceil(height[i-1] / stride);
width[i] = np.ceil(width[i-1] / stride);
print('H,W = {},{}'.format(height[i], width[i]));

#fine3_dropout = tf.nn.dropout(fine3, keep_conv)

# OLD:
#fine4 = conv2d('fine4', fine3_dropout, [5, 5, 64, 1], [1], [1, 1, 1, 1], padding='SAME', reuse=reuse, trainable=trainable)
i += 1;
filter_size = 5;
stride = 1;
height[i] = np.ceil(height[i-1] / stride);
width[i] = np.ceil(width[i-1] / stride);
print('H,W = {},{}'.format(height[i], width[i]));

## NEW:
##fine4 = conv2d('fine4', fine3_dropout, [5, 5, 64, 1], [1], [1, 1, 1, 1], padding='VALID', reuse=reuse, trainable=trainable)
#i += 1;
#filter_size = 7;
#stride = 1;
#height[i] = np.ceil((height[i-1] - filter_size + 1) / stride);
#width[i] = np.ceil((width[i-1] - filter_size + 1) / stride);
#print('H,W = {},{}'.format(height[i], width[i]));


print('Number of weights out = {}'.format(height[i]*width[i]))
