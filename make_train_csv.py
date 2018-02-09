# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 17:14:09 2018

Create a csv file with images in column 1 and depth images in column 2.

@author: guido
"""

import os
import sys
import time
import numpy as np
import cv2
import pdb

SANITY_CHECK = True;

# function for changing the name:
pattern1 = 'sync/';
lp1 = len(pattern1);
pattern2 = '2/data/';
lp2 = len(pattern2);
pattern3 = '.png';
lp3 = len(pattern3);
disp_dirname = 'disp/';
disp_suffix = '_disparity';

def get_disparity_name(image_name):
    i1 = image_name.find(pattern1);
    i1 += lp1;
    i2 = image_name.find(pattern2);
    i2 += lp2;
    i3 = image_name.find(pattern3);
    
    disparity_name = image_name[:i1] + disp_dirname + image_name[i2:i3] + disp_suffix + image_name[i3:];
    return disparity_name;



fname_left = 'train_images2.txt';
with open(fname_left) as f:
    images_left = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
images_left = [x.strip() for x in images_left] 

fname_target = 'train_new.csv'

f = open(fname_target, 'w');
for idx in range(len(images_left)):

    if(np.mod(idx, np.round(len(images_left)/10)) == 0):
        print('.');
        time.sleep(0.1);

    image_name = images_left[idx];
    disparity_name = get_disparity_name(image_name);
    f.write('{},{}\n'.format(image_name, disparity_name));
    
    if(SANITY_CHECK):
        im = cv2.imread(image_name);
        disp = cv2.imread(disparity_name);
        mean_im = np.mean(im[:]);
        mean_disp = np.mean(disp[:]);
        print('{}/{}: Mean im = {}, disp = {}.'.format(idx, len(images_left), mean_im, mean_disp));
        if(mean_disp > 250.0 or mean_im > 250.0):
            print('filename: {}'.format(image_name));
            pdb.set_trace();        
    
f.close();
