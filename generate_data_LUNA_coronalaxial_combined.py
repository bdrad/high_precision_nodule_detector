#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 17:49:06 2019

@author: Yesh
"""

import os
import numpy as np
import pandas as pd
import sys
import cv2
import time
import argparse

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.rcParams['figure.figsize'] = [12, 8]

import Mip_Utility as mip_utils # utils for LUNA data

np.random.seed(42)



def bbox(mask):
    # get bbox from numpy mask
    # src: https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if (np.any(rows) == False and np.any(cols) == False):
        return 0,0,0,0

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    if rmin == rmax:
        rmax += 1

    if cmin == cmax:
        cmax += 1

    return rmin-1, rmax+1, cmin-1, cmax+1

def convert_to_coronal(img_a):
    # img_a is a numpy array of axial slices of shape (n_a_slices, x, y)
    # img_c is returned which is numpy array of coronal slices of shape (x, n_a_slices, y), where x = n_coronal_slices
    slices, x, y = img_a.shape
    img_c = np.zeros([x, slices, y])
    for i, s in enumerate(img_a):
        img_c[:,i,:] = s
    return img_c


def convert_to_3channels(img, channel_space):
    n, h, w = img.shape
    img_3c = np.zeros((n,h,w,3))
    for i, _ in enumerate(img_3c):
        img_3c[i,:,:,0] = img[np.max([i-channel_space, 0])]
        img_3c[i,:,:,1] = img[i]
        img_3c[i,:,:,2] = img[np.min([i+channel_space, len(img)-1])]

    return img_3c

def generate_data(mhd_ids, split, save_dir, luna_data_dir, annotations_df, mip=25, no_masking=False,
                  positives_only=False, min_bbox_area=0, channel_space=0):
    # Initialize array for CSV that will contain bboxes
    csv = []

    for idx, mhd_id in enumerate(mhd_ids):
        print('-'*50)
        print('N = {}/{}'.format(idx, len(mhd_ids)))
        print(mhd_id)
        time_mhd0 = time.time()
        mhd_id_fp = mhd_id + '.mhd'
        path = os.path.join(luna_data_dir, mhd_id_fp)

        try:
            # these are axial sections by default
            lung_img_512_axial, lung_mask_512_axial, nodule_mask_512_axial = mip_utils.create_nodule_mask(path, annotations_df[annotations_df['seriesuid'] == mhd_id])
        except Exception:
            continue


        for section in ['axial', 'coronal']:

            if section == 'coronal':
                lung_img_512 = convert_to_coronal(lung_img_512_axial)
                lung_mask_512 = convert_to_coronal(lung_mask_512_axial)
                nodule_mask_512 = convert_to_coronal(nodule_mask_512_axial)
            elif section=='axial':
                lung_img_512 = lung_img_512_axial
                lung_mask_512 = lung_mask_512_axial
                nodule_mask_512 = nodule_mask_512_axial

            if no_masking:
                # we dont want just the lung masks so then just make the lung_mask object the lung_img
                lung_mask_512 = lung_img_512

            if mip:
                lung_mask_512 = mip_utils.createMIP(lung_mask_512, slices_num=mip)


            # Flipping the images upside down
            lung_img_512_new = lung_img_512.copy()
            lung_mask_512_new = lung_mask_512.copy()
            nodule_mask_512_new = nodule_mask_512.copy()
            for i in range(len(lung_img_512)):
                lung_img_512_new[i] = np.flipud(lung_img_512[i])
                lung_mask_512_new[i] = np.flipud(lung_mask_512[i])
                nodule_mask_512_new[i] = np.flipud(nodule_mask_512[i])
            lung_img_512, lung_mask_512, nodule_mask_512 = lung_img_512_new, lung_mask_512_new, nodule_mask_512_new


            lung_img_512 = convert_to_3channels(lung_img_512, channel_space)
            lung_mask_512 = convert_to_3channels(lung_mask_512, channel_space)

            lung_mask_512 = ((lung_mask_512 - lung_mask_512.min()) * (1/(lung_mask_512.max() - lung_mask_512.min()) * 255)) # scale to 0-255
            pos_counter = 0

            for i, s in enumerate(nodule_mask_512):
                filename = mhd_id + '-' + str(i) + '_' + section + '.npz'
                saved_fp = os.path.join(os.path.abspath(save_dir), filename)

                bbox_too_small = False
                y1, y2, x1, x2 = bbox(s)

                # check if bbox is too small or no bbox - need the second or statement b/c of the "+1" in previous statement!
                if np.abs((y2-y1+1) * (x2-x1+1) <= min_bbox_area) or (x1 == 0 and y1 == 0 and x2 ==0  and y2 == 0):
                    bbox_too_small = True

                if not bbox_too_small:
                    csv.append([saved_fp, x1,y1,x2,y2,'cancer'])
                    
                    np.savez_compressed(saved_fp, lung_mask_512[i])
                    
                    pos_counter += 1

                    if idx==0: # plot
                        print(i)
                        print(np.count_nonzero(s))
                        print(x1, x2, y1, y2)

                        fig,ax = plt.subplots(1)
                        ax.imshow(lung_mask_512[i,:,:]/255)
                        #ax.imshow(s, alpha=0.3, cmap='Reds')
                        #ax.axis([120,395,395, 120]) # zooom in
                        rect = patches.Rectangle((x1,y1),x2-x1,y2-y1,linewidth=0.5,edgecolor='r',facecolor='none')
                        ax.add_patch(rect)
                        fig.savefig('./generate_data_LUNA_test_images/{}-{}-{}.png'.format(mhd_id, i, section))
                        # fig.show()
                        plt.close(fig)

                        # fig,ax = plt.subplots(1)
                        # ax.imshow(lung_img_512[i,:,:]/255)
                        # #ax.axis([120,395,395, 120]) # zooom in
                        # fig.show()



                else:
                    if not positives_only:
                        np.savez_compressed(saved_fp, lung_mask_512[i])
    
                        csv.append([saved_fp, '','','','',''])

        print('MHD Data Gen Time: {} min'.format(round((time.time()-time_mhd0)/60, 2)))    
        print(pos_counter) # counts how many positive slices in each mhd_id - used to debug the running out of memory issue

    # save csv
    df_csv = pd.DataFrame(csv)
    df_csv.to_csv(os.path.join(save_dir + '/annotations_{}.csv'.format(split)),
                  index=False, header=False)




def main(args=None):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--mip', default=25, type=int)
    parser.add_argument('--no_masking', action='store_true')
    parser.add_argument('--positives_only', action='store_true')
    parser.add_argument('--LUNA_DATA_DIR', default="/media/bdrad1/Backup Plus/datasets/original_whole_LUNA/LUNA_Total")
    parser.add_argument('--LUNA_ANNS', default="/media/bdrad1/Backup Plus/datasets/original_whole_LUNA/annotations.csv")
    parser.add_argument('--SAVE_DIR')
    parser = parser.parse_args()
    print(parser.SAVE_DIR)
    os.mkdir(parser.SAVE_DIR)

    annotations_df = pd.read_csv(parser.LUNA_ANNS)

    # create classes anns
    pd.DataFrame([['cancer',0]]).to_csv(parser.SAVE_DIR + '/classes.csv', header=False, index=False)

    # Create Train/Val/Test Split
    mhd_ids = [file.split('.mhd')[0] for file in os.listdir(parser.LUNA_DATA_DIR) if file.endswith('.mhd')]
    np.random.shuffle(mhd_ids)

    mhd_ids_train = mhd_ids[:int(0.7*len(mhd_ids))]
    mhd_ids_val = mhd_ids[int(0.7*len(mhd_ids)):int(0.7*len(mhd_ids))+int(0.15*len(mhd_ids))]
    mhd_ids_test = mhd_ids[int(0.7*len(mhd_ids))+int(0.15*len(mhd_ids)):]

    generate_data(mhd_ids_train,
                    luna_data_dir=parser.LUNA_DATA_DIR,
                    annotations_df=annotations_df,
                    split='train', 
                    mip=parser.mip,
                    no_masking=parser.no_masking, 
                    save_dir=parser.SAVE_DIR,
                    positives_only=parser.positives_only, 
                    min_bbox_area=0, 
                    channel_space=0)

    generate_data(mhd_ids_val, 
                    luna_data_dir=parser.LUNA_DATA_DIR,
                    annotations_df=annotations_df,
                    split='val', 
                    mip=parser.mip,
                    no_masking=parser.no_masking, 
                    save_dir=parser.SAVE_DIR,
                    positives_only=parser.positives_only, 
                    min_bbox_area=0, 
                    channel_space=0)

    generate_data(mhd_ids_test, 
                    luna_data_dir=parser.LUNA_DATA_DIR,
                    annotations_df=annotations_df,
                    split='test', 
                    mip=parser.mip,
                    no_masking=parser.no_masking, 
                    save_dir=parser.SAVE_DIR,
                    positives_only=parser.positives_only, 
                    min_bbox_area=0, 
                    channel_space=0)
  

if __name__ == '__main__':
    time_total0 = time.time()
    
    main()
    
    print('-'*30)
    print('Total Time: {} min'.format(round((time.time()-time_total0)/60, 2)))
 

