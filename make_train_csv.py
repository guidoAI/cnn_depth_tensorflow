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

KITTI = True;
CONFIDENCE_MAP = True;

if(KITTI):
    SANITY_CHECK = False;
    WRITE = True;
    
    # function for changing the name:
    pattern1 = 'sync/';
    lp1 = len(pattern1);
    pattern2 = '2/data/';
    lp2 = len(pattern2);
    pattern3 = '.png';
    lp3 = len(pattern3);
    disp_dirname = 'disp/';
    disp_suffix = '_disparity';
    conf_dirname = 'conf/';
    conf_suffix = '_confidence';
    
    def get_disparity_name(image_name):
        i1 = image_name.find(pattern1);
        i1 += lp1;
        i2 = image_name.find(pattern2);
        i2 += lp2;
        i3 = image_name.find(pattern3);
        
        disparity_name = image_name[:i1] + disp_dirname + image_name[i2:i3] + disp_suffix + image_name[i3:];
        return disparity_name;
    
    def get_confidence_name(image_name):
        i1 = image_name.find(pattern1);
        i1 += lp1;
        i2 = image_name.find(pattern2);
        i2 += lp2;
        i3 = image_name.find(pattern3);
        
        confidence_name = image_name[:i1] + conf_dirname + image_name[i2:i3] + conf_suffix + image_name[i3:];
        return confidence_name;
    
    
#    fname_left = 'train_images2.txt';
#    fname_target = 'train_new.csv'; 
    
    fname_left = 'val_images2.txt';
    fname_target = 'validate_new.csv'
    
    with open(fname_left) as f:
        images_left = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    images_left = [x.strip() for x in images_left] 
    
    if(WRITE):
        f = open(fname_target, 'w');
    
    invalid_disp = 0;
        
    for idx in range(len(images_left)):
    
        if(np.mod(idx, np.round(len(images_left)/10)) == 0):
            print('.');
            time.sleep(0.1);
    
        image_name = images_left[idx];
        disparity_name = get_disparity_name(image_name);
        if(CONFIDENCE_MAP):
            confidence_name = get_confidence_name(image_name);
        if(WRITE):
            if(not CONFIDENCE_MAP):
                f.write('{},{}\n'.format(image_name, disparity_name));
            else:
                f.write('{},{},{}\n'.format(image_name, disparity_name, confidence_name));
        
        if(SANITY_CHECK):
            im = cv2.imread(image_name);
            disp = cv2.imread(disparity_name);
            mean_im = np.mean(im[:]);
            mean_disp = np.mean(disp[:]);
            print('{}/{}: Mean im = {}, disp = {}.'.format(idx, len(images_left), mean_im, mean_disp));
            if(mean_disp > 250.0 or mean_im > 250.0):
                print('filename: {}'.format(image_name));
                invalid_disp += 1;        
                
    if(SANITY_CHECK):   
        print('Number of invalid disparity maps: {}'.format(invalid_disp));
    
    if(WRITE):
        f.close();
else:
    # based on two directories, one with images, the other with depth maps:
    dir_name_images = '/home/guido/KITTI/2011_09_26/2011_09_26_drive_0002_sync/image_02/data/'; #'./test_kitti/'; #'./data/learn_images/';
    dir_name_depths = '/home/guido/KITTI/2011_09_26/2011_09_26_drive_0002_sync/GT_disp/';#'./test_kitti/'; #'./data/dense_GT_images/';

    fname_target = 'test_kitti.csv'; #'train_nuovo.csv'
    f = open(fname_target, 'w');

    for file in os.listdir(dir_name_images):
        if file.endswith(".png"):
            file_name_image = dir_name_images + file;
            file_name_depth = dir_name_depths + file;
            f.write('{},{}\n'.format(file_name_image, file_name_depth));
    
    f.close();
