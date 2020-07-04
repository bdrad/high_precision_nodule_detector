#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze NLST

Created on Thu Dec 12 22:41:52 2019

@author: Yesh
"""

import csv
import numpy as np
import pandas as pd
import time
import os
#os.chdir('/home/bdrad1/chest_ct_projects/pytorch-retinanet')
#os.chdir('/Users/Yesh/Documents/BDRAD/chest_ct_projects/pytorch-retinanet')

import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataloader_NLST import NLSTDataset, Normalizer, Resizer


print('CUDA available: {}'.format(torch.cuda.is_available()))

def listdir_nohidden(path):
        for f in os.listdir(path):
            if not f.startswith('.'):
                yield f


def load_csv(results_dir: str, csv_path: str, data_dir: str, pids):
    """
    Loads csv given path to sctabn csv file and path to where results csvs are written.
    Removes examples from sctabn csv which aren't in local dataset, and removes examples for which we've already done inference.

    Arguments: 
        1) results_dir: Path to where results csvs are written
        2) csv_path: Path to sctabn csv
        3) data_dir: Path to directory containing all data
        4) pids: List of PIDs to perform inference on, None if we want to do
            inference on all examples locally stored. 
    Returns: 
        anns: pandas.DataFrame object with annotations
    """

    save_path = ""
    for x in csv_path.split("/")[:-1]:
        save_path = os.path.join(save_path, x)
    save_path = os.path.join(save_path, "sctabn_massive_annotation.csv")  # Location new csv will be saved, this avoids repetitive computation

    if not os.path.exists(save_path):
        anns = pd.read_csv(csv_path)
    
        remove_idx = []
        for index in range(len(anns)):
            ann = anns.iloc[index]
            pid = ann['pid']
            study_yr = ann['STUDY_YR']
            pid_dir = os.path.join(data_dir,str(pid))
            if os.path.isdir(pid_dir) == False:  # If we don't have the PID
                remove_idx.append(index)
                continue
            timestamps = sorted(listdir_nohidden(pid_dir))

            study_yr = 'T'+str(study_yr)
            if study_yr not in timestamps:  # If we have the PID but not the specific study year
                remove_idx.append(index)

        for index in remove_idx[::-1]:  # Remove in reverse order so index to remove don't move around unexpectedly
            anns.drop(index, axis=0, inplace=True)
        anns.to_csv(save_path, index=False)
    else:
        anns = pd.read_csv(save_path) 



    # If directory exists, continue annotations from there
    # - by removing already inferenced anns from the dataloader
    if os.path.exists(results_dir):
        # get list of all pid-study_yrs already inferenced - will be used to drop pid_yr from "anns"
        df_preds = pd.DataFrame(columns=['pid', 'study_yr'])
        for f in sorted(os.listdir(results_dir)):
            if f.endswith('.csv'):
                df_temp = pd.read_csv(os.path.join(results_dir, f), header=0)
                # just get unique pid-study_yr to prevent dataframe from getting too large
                df_temp = df_temp.drop_duplicates(subset=['pid', 'study_yr'])
                df_preds = df_preds.append(df_temp[['pid', 'study_yr']])
                last_f = f

        # remove last pid_yr in case incomplete from BOTH the last csv and the df_preds so it will reinference this pid
        last_pid = df_preds.tail(1)['pid'].values[0]
        last_yr = df_preds.tail(1)['study_yr'].values[0]
        df_last_preds = pd.read_csv(os.path.join(results_dir, last_f), header=0)

        ## drop from last pid_yr and save the last results csv
        df_last_preds = df_last_preds[~((df_last_preds['pid'] == last_pid) & (df_last_preds['study_yr'] == last_yr))]
        df_last_preds.to_csv(os.path.join(results_dir, last_f), index=False)

        ## drop from df_preds so you won't drop it from anns
        df_preds = df_preds[~((df_preds['pid'] == last_pid) & (df_preds['study_yr'] == last_yr))]

        # drop those already in preds
        anns = anns[~((anns['pid'].isin(df_preds['pid'])) & (anns['STUDY_YR'].isin(df_preds['study_yr'])))]
    else:
        os.mkdir(results_dir)
    
    if isinstance(pids, int):
        anns = anns[:pids]
    elif isinstance(pids, list):
        anns = anns[anns['pid'].isin(pids)]

    return anns


def main(args=None):
    parser = argparse.ArgumentParser(description='NLST inference script for retinanet.')
    parser.add_argument('--datadir', help='Path to NLST Folder')
    parser.add_argument('--anns', help='Path to NLST sctabn.csv Annotations')
    parser.add_argument('--model', help='Path to retinanet model')
    parser.add_argument('--pids', help='load an array of specific PIDs (optional)', type=int, nargs='+', default=None) # test pids with large nodules: [207375, 210581] 
    parser.add_argument('--mip', help='apply maximum intensity projects', type=int, default=25)
    parser.add_argument('--nodule_cts_only', help='Apply inference to only use CT scans with a known nodule', action="store_true")
    parser.add_argument('--no_voxel_transform', help='Do not applu 1mm transformation to voxels', action="store_true")
    parser.add_argument('--results_dir', help='Dir to save results csv or director from which to continue')
    parser.add_argument('--no_mask', help='do NOT apply lung mask (i.e. do not segment out the lungs with manual thresholding)',  action="store_true")
    parser = parser.parse_args()

    np.random.seed(10)

    results_dir = parser.results_dir

    anns = load_csv(results_dir=results_dir, csv_path=parser.anns, 
        data_dir=parser.datadir, pids=parser.pids)

    # Load dataloader
    nlst = NLSTDataset(fp_anns=anns,
                       data_dir=parser.datadir,
                       nodule_cts_only=parser.nodule_cts_only,
                       transform=transforms.Compose([Normalizer(), Resizer()]),
                       mip=parser.mip,
                       voxel_transform = not parser.no_voxel_transform,
                       mask = not parser.no_mask)
    dataloader = DataLoader(nlst, shuffle=False)

    print('Remaining PID-YRs: {}'.format(nlst.anns.shape[0]))

    if parser.pids:
        print('Inferencing: {}'.format(np.unique(nlst.anns['pid'])))

    # Load Model
    assert(torch.cuda.is_available())
    retinanet = torch.load(parser.model)
    retinanet = retinanet.cuda()
    retinanet.eval()

    
    # index for the results files
    results_file_index = round(time.time())
    cts_per_results_file = 100 # how many pid-year combos per ct

    total_time0 = time.time()
    preprocess_time_0 = time.time()
    for idx, data in enumerate(dataloader):        
        if bool(data) == False:
            continue

        preprocess_time = time.time() - preprocess_time_0
        print('preprocessing time: {:.2f}s'.format(preprocess_time))

        # intiailize csv filename
        if idx % cts_per_results_file == 0:
            results_file_index = round(time.time())
        
        results_filename = 'NLST_inferences_' + str(results_file_index)  + '.csv'
        results_csv_path = os.path.join(results_dir, results_filename)
        
        for kernel, kernel_data in data.items():

            # iterate through axial and coronal views 
            for view, item in kernel_data.items():
                ct = item['img_array']
                file_path = item['file_path'][0]

                scale = float(item['scale'])
                axial_conversion = float(item['axial_conversion'])
                saggital_conversion = float(item['saggital_conversion'])
                coronal_conversion = float(item['coronal_conversion'])


                pid = int(item['pid'])
                study_yr = int(item['study_yr'])
    
                print('Inferencing {}_{}_{}_{}'.format(pid, study_yr, kernel, view))
                
                inference_time_list = []
                for slice_idx, ct_slice in enumerate(ct[0]): # NOTE: this loop only works with inferencing a batch size of 1 currently    
                    with torch.no_grad():
                        h, w = ct_slice.shape[:2]
                        ct_input = ct_slice.unsqueeze(0)
                        ct_input = ct_input.permute(0,3,1,2)
                        ct_input = torch.cat((ct_input, ct_input, ct_input), dim=1)
                        
                        inference_time_0 = time.time()
                        scores, classifications, transformed_anchors = retinanet(ct_input.cuda().float())
                        inference_time = time.time() - inference_time_0
                        inference_time_list.append(inference_time)
    
                        for score, classification, transformed_anchor in zip(scores, classifications, transformed_anchors):
                            bbox = transformed_anchor.cpu().numpy()
                            x1 = int(bbox[0])
                            y1 = int(bbox[1])
                            x2 = int(bbox[2])
                            y2 = int(bbox[3])
                            confidence = float(score.cpu())

                            inference_time_per_prediction = inference_time / len(scores)                            
                            
                            # initialize csv headers
                            if not os.path.exists(results_csv_path):
                                with open(results_csv_path,'w',newline='') as csvfile:
                                    writer = csv.writer(csvfile, delimiter=',')
                                    writer.writerow(['pid', 'study_yr', 'kernel', 'view', 'pred_slice_num', 
                                                     'scale', 'axial_conversion', 'saggital_conversion', 'coronal_conversion',
                                                     'x1', 'y1', 'x2', 'y2', 'confidence', 'file_path', 'mip', 
                                                     'inference_time', 'preprocess_time', 'model_fp', 'height','width', 'timestamp',
                                                     'voxel_transform'])
                            # write to csv
                            with open(results_csv_path,'a', newline='') as csvfile:
                                csv_array = [pid, study_yr, kernel, view, slice_idx, 
                                            scale, axial_conversion, saggital_conversion, coronal_conversion, 
                                            x1, y1, x2, y2, confidence, file_path, parser.mip, 
                                            inference_time_per_prediction, preprocess_time, parser.model, h, w, time.time(),
                                            not parser.no_voxel_transform]
                                writer = csv.writer(csvfile, delimiter=',')
                                writer.writerow(csv_array)
                
                print('- inference time: {:.2f}s'.format(sum(inference_time_list)))

        # reset preprocessing timer
        preprocess_time_0 = time.time()

        # estimate total inference time remaining = (number of PID-YRs reamining) * (total_time / pidyrs inferenced)
        remaining_pidyrs = len(nlst.anns) - (idx+1)
        time_remaining_estimate_seconds = (remaining_pidyrs) *  ((time.time() - total_time0) / (idx+1))
        print('PID-YRs Remaining: {}'.format(remaining_pidyrs))
        print('- time remaining estimate: {:.2f} hours'.format(time_remaining_estimate_seconds / 60 / 60))

        # cleanup big items to reduce memory usage
        del data, kernel_data, item, ct, ct_input, 




if __name__ == '__main__':
    main()
