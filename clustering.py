#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 11:17:08 2020

@author: Yesh
"""

import os
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from stl import mesh
import time
import argparse
import matplotlib.pyplot as plt
import joblib

from dataloader_NLST import NLSTDataset, Normalizer, Resizer
from torchvision import transforms

plt.rcParams['figure.figsize'] = [12, 6]

"""
Script to cluster slice inferences into discrete nodules.
 - saves nodules predictions into CSV
 - creates Mesh file to visualize the nodule using Slicer3d
"""

def createMeshCube(corner, dx, dy, dz):
    """
    Returns a mesh cube object with the above diameters for x, y, z axis with respect to an axial slice orientation
     - corner is the posterior, inferior, left corner - I think
    """
    
    # vertices of square
    vertices = np.array([
        corner,
        corner + np.array([dx,  0,  0]),
        corner + np.array([dx, dy,  0]),
        corner + np.array([ 0, dy,  0]),
        corner + np.array([ 0,  0, dz]),
        corner + np.array([dx,  0, dz]),
        corner + np.array([dx, dy, dz]),
        corner + np.array([ 0, dy, dz]),])

    # Define the 12 triangles composing the cube
    faces = np.array([
        [0,3,1],
        [1,3,2],
        [0,4,7],
        [0,7,3],
        [4,5,6],
        [4,6,7],
        [5,1,2],
        [5,2,6],
        [2,3,6],
        [3,7,6],
        [0,1,5],
        [0,5,4]])

    # Create the mesh
    cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            cube.vectors[i][j] = vertices[f[j],:]
    return cube
    
def convertToWorldCoords(df_pid_yr, dcms, origin, mip):
    # - Axial inferences
    df_axial = df_pid_yr[df_pid_yr['view'] == 'axial'].copy()
    
    df_axial['axial_slice'] = (df_axial['pred_slice_num'] - mip/2)/ df_axial['axial_conversion']
    df_axial['axial_world_coord'] = [dcms[int(slice_idx)].ImagePositionPatient[2] for slice_idx in df_axial['axial_slice'].values]
    
    df_axial['saggital_world_coord'] = -origin[0] - (df_axial['x'] / df_axial['scale'])
    df_axial['coronal_world_coord'] = -origin[1] - (df_axial['y'] / df_axial['scale'])
    
    
    # - Coronal inferences
    df_cor = df_pid_yr[df_pid_yr['view'] == 'coronal'].copy()
    
    df_cor['axial_slice'] = df_cor['y'] / df_cor['scale'] / df_cor['axial_conversion']
    df_cor['axial_world_coord'] = [dcms[int(slice_idx)].ImagePositionPatient[2] for slice_idx in df_cor['axial_slice'].values]

    df_cor['saggital_world_coord'] = -origin[0] - (df_cor['x'] / df_cor['scale'])
    df_cor['coronal_world_coord'] = -origin[1] - (df_cor['pred_slice_num'] - mip/2)

    # - combine axial and coronal
    df_cluster = pd.concat([df_cor, df_axial], axis=0)
    return df_cluster
            


def main(args=None):
    parser = argparse.ArgumentParser(description='Clustering on NLST Inference By pid and study_yr')
    parser.add_argument('--inference_dir', help='Path to Directory containing NLST Inferences', default='')
    parser.add_argument('--save_dir', help='Where to Save Nodule Candidates CSV & Mesh Segmentation Files', default='')
    parser.add_argument('--conf_thresh', help='Confidence Threshold to drop inferences', default=0.10, type=float)
    parser.add_argument('--nlst_dir', help='Directory of NLST data', default='../data/nlst-ct')
    parser.add_argument('--nlst_anns', help='Path to NLST annotations', default='../data/nlst_annotations/all_14_datasets/cdas_delivery_csv/sctabn.csv')
    parser.add_argument('--plot', help='Plot Predictions for debugging', action='store_true')
    parser.add_argument('--fp_reducer', help='Path To False Positive Reducer', default=None)
    parser = parser.parse_args()
    
    nodules_fn = 'nodules.csv'

    if parser.fp_reducer:
        # load fp_reducer
        fp_reducer = joblib.load(parser.fp_reducer)
    
    # initialize or load already existing df_nodules
    if os.path.exists(parser.save_dir):
        df_nodules = pd.read_csv(os.path.join(parser.save_dir, nodules_fn))
        completed_pid_yrs = df_nodules['pid'].astype(str) + '_' + df_nodules['study_yr'].astype(str)
        completed_pid_yrs = completed_pid_yrs.values
    else:
        os.mkdir(parser.save_dir)
        df_nodules = pd.DataFrame()
        completed_pid_yrs = []
        
    # get all csvs & sort in scending timestamp
    csvs = [f for f in os.listdir(parser.inference_dir) if f.endswith('.csv')]
    csvs.sort()
    
    # loop through csvs
    for csv in csvs:
        time_csv0 = time.time()
        print('-'*30)
        print(csv)
        csv_base = csv.split('.')[0]
        
        # read csv
        df_csv = pd.read_csv(os.path.join(parser.inference_dir, csv), header=0)
        
        # create a pid_study_yr convenience index
        df_csv['pid_yr'] = df_csv['pid'].astype(str) + '_' + df_csv['study_yr'].astype(str)
        
        # loop through unique pid_study 
        for pid_yr in np.unique(df_csv['pid_yr']):
            # skip if already clustered
            if pid_yr in completed_pid_yrs:
                continue
            
            time_pid_yr0 = time.time()
            
            # get subset matching pid_yr
            df_pid_yr = df_csv[df_csv['pid_yr'] == pid_yr].copy()
            pid = df_pid_yr['pid'].values[0]
            study_yr = df_pid_yr['study_yr'].values[0]
            mip = df_pid_yr['mip'].values[0]

            # apply conf_thresh
            conf_thresh = parser.conf_thresh
            df_pid_yr = df_pid_yr[df_pid_yr['confidence'] > conf_thresh]
            
            # if no rows remaining, save empty row to final dataframe
            if len(df_pid_yr) == 0:
                df_nodules = df_nodules.append(pd.DataFrame({'pid': [pid], 'study_yr': [study_yr]}), sort=False)
                df_nodules = df_nodules.reset_index(drop=True)
                continue 
            
            # use only kernel with highest mean confidence score
            kernel = df_pid_yr.groupby('kernel').confidence.max().idxmax()
            df_pid_yr = df_pid_yr[df_pid_yr['kernel'] == kernel]

    
            # get origin from NLSTDataset
            try:
                nlst = NLSTDataset(fp_anns=parser.nlst_anns,
                       data_dir=parser.nlst_dir,
                       transform=transforms.Compose([Normalizer(), Resizer()]))
                
                res = nlst.load_NLST_images(pid, study_yr, load_dcms=True)
                dcms = res[kernel]['dcms']
                origin = res[kernel]['image_position_patient']
            except:
                print('Unable to load NLST data for {}. Check if you have the dicom data locally'.format(pid_yr))
                continue
            
            #  calculate diameter
            df_pid_yr['nod_width'] = np.abs(df_pid_yr['x1'] - df_pid_yr['x2']) / df_pid_yr['scale']
            df_pid_yr['nod_height'] = np.abs(df_pid_yr['y1'] - df_pid_yr['y2']) / df_pid_yr['scale']
            df_pid_yr['diameter'] = np.max(df_pid_yr[['nod_width', 'nod_height']], axis=1)
            
            # calc mean nodule position
            df_pid_yr['x'] = (df_pid_yr['x1'] + df_pid_yr['x2'])/2
            df_pid_yr['y'] = (df_pid_yr['y1'] + df_pid_yr['y2'])/2
            
            # Convert to World Coordinates
            try:
                df_cluster = convertToWorldCoords(df_pid_yr, dcms, origin, mip)
            except IndexError as e:
                print(' - {} Error: {}. May be missing some dcms files if you transferred dicoms to another computer'.format(pid_yr, e))
                
                continue
                
            # convenience: binary view to perform calculation easily
            df_cluster['is_coronal'] = np.array(df_cluster['view'] == 'coronal').astype(int)

            
            # plot axial vs coronal
            if parser.plot:
                f, ax = plt.subplots()
                points = ax.scatter(df_cluster['saggital_world_coord'], df_cluster['axial_world_coord'],
                                    s=(df_cluster['confidence']*20)**3.5,
                                    c=df_cluster['is_coronal'],
                                    alpha=0.1)
                f.colorbar(points)
                points
            
            # 3d plot
            if parser.plot:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')

                # plot coronal
                df_sub_cor = df_cluster[df_cluster['is_coronal'] == 1]
                ax.scatter(df_sub_cor['saggital_world_coord'],  df_sub_cor['axial_world_coord'],  df_sub_cor['coronal_world_coord'],
                           s=(df_sub_cor['confidence']*25)**3,
                           label='coronal inferences',
                           c='#003f5c',
                           alpha=0.1)

                # plot axial
                df_sub_ax = df_cluster[df_cluster['is_coronal'] == 0]
                ax.scatter(df_sub_ax['saggital_world_coord'],  df_sub_ax['axial_world_coord'],  df_sub_ax['coronal_world_coord'],
                           s=(df_sub_ax['confidence']*25)**3,
                           label='axial inferences',
                           c='#ff6361',
                           alpha=0.1)
                
                ax.set_xlabel('Saggital Position')
                ax.set_ylabel('Axial Position')
                ax.set_zlabel('Coronal Position')
                ax.legend()
                plt.savefig('inferences_3d_scatter.pdf', dpi=300)
                
            
            # CLUSTERING
            df_cluster['cluster'] = DBSCAN(eps=10, min_samples=4).fit_predict(X=df_cluster[['axial_world_coord', 'saggital_world_coord', 'coronal_world_coord']])
            
            # remove all unclustered inferences
            df_cluster = df_cluster[df_cluster['cluster'] != -1]
            
            
            # if no rows remaining, save empty row to final dataframe
            if len(df_cluster) == 0:
                df_nodules = df_nodules.append(pd.DataFrame({'pid': [pid], 'study_yr': [study_yr]}), sort=False)
                df_nodules = df_nodules.reset_index(drop=True)
                continue 
            
            
            # plot clusters
            if parser.plot:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                points = ax.scatter(df_cluster['saggital_world_coord'], df_cluster['axial_world_coord'], df_cluster['coronal_world_coord'],
                                    s=(df_cluster['confidence']*20)**4,
                                    c=df_cluster['cluster'],
                                    alpha=0.1)
                fig.colorbar(points)
                points
                
                
            # plot clusters 3d
            if parser.plot:
                plt.style.use('seaborn')
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')

                df_sub = df_cluster
                ax.scatter(df_sub['saggital_world_coord'],  df_sub['axial_world_coord'],  df_sub['coronal_world_coord'],
                           s=(df_sub['confidence']*25)**3,
                           c=df_sub['cluster'].values,
                           label=df_sub['cluster'].values,
                           alpha=0.1)
                    
                ax.set_xlabel('Saggital Position')
                ax.set_ylabel('Axial Position')
                ax.set_zlabel('Coronal Position')
                
                plt.savefig('inferences_clustered_3d_scatter.pdf', dpi=300)

         
            # Group inferences by cluster
            df_cluster = df_cluster.groupby('cluster').agg({
                                              'confidence':['max', 'mean', 'std'],
                                              'axial_world_coord': ['mean'],
                                              'saggital_world_coord': ['mean'],
                                              'coronal_world_coord': ['mean'],
                                              'pred_slice_num': ['count'],
                                              'is_coronal': ['mean'],
                                              'diameter': ['max']})

            # collapse multiindex columns b/c easier to work with
            df_cluster.columns = list(map('_'.join, df_cluster.columns.values))            
            
            df_cluster['is_seen_axial_and_coronal'] = np.where(df_cluster['is_coronal_mean'].isin([0,1]), 0, 1)
            df_cluster['top_world_coord'] = dcms[0].ImagePositionPatient[2]
            df_cluster['bottom_world_coord'] = dcms[-1].ImagePositionPatient[2]
            df_cluster['dist_to_top'] = np.abs(df_cluster['axial_world_coord_mean'] - df_cluster['top_world_coord'])
            df_cluster['dist_to_bottom'] = np.abs(df_cluster['axial_world_coord_mean'] - df_cluster['bottom_world_coord'])
  
            
            if parser.fp_reducer:
                # apply fp_reducer
                variables = ['is_seen_axial_and_coronal', 'pred_slice_num_count', 'confidence_max', 
                             'confidence_mean', 'diameter_max', 'dist_to_top', 'dist_to_bottom']
                df_cluster['nodule_proba'] = fp_reducer.predict_proba(df_cluster[variables])[:,1]
                df_cluster = df_cluster[df_cluster['nodule_proba'] >= 0.2]

                df_cluster = df_cluster.reset_index(drop=True)
            
            # if no rows remaining, save empty row to final dataframe
            if len(df_cluster) == 0:
                df_nodules = df_nodules.append(pd.DataFrame({'pid': [pid], 'study_yr': [study_yr]}), sort=False)
                df_nodules = df_nodules.reset_index(drop=True)
                continue 
            
            # create cubes & save cubes
            cubes = []
            for nod, row in df_cluster.iterrows():
                dx, dy, dz = row['diameter_max'], row['diameter_max'], row['diameter_max']
                
                corner = np.array([0,0,0])
                corner[0] = row['saggital_world_coord_mean'] - dx/2
                corner[1] = row['coronal_world_coord_mean'] - dy/2
                corner[2] = row['axial_world_coord_mean'] - dz/2
    
    
                cube = createMeshCube(corner, dx, dy, dz)
                cubes.append(cube)
    
            cubes = mesh.Mesh(np.concatenate([c.data for c in cubes]))
            
            # save cube to directory, grouped into folders with same name as the csv
            cube_dir = os.path.join(parser.save_dir, csv_base)
            if not os.path.exists(cube_dir):
                os.mkdir(cube_dir)
                
            cubes.save(os.path.join(cube_dir,  '{}_{}_{}.stl'.format(pid, study_yr, kernel, len(df_cluster), mip)))
            
            # add in metadata
            df_cluster['pid'] = pid
            df_cluster['study_yr'] = study_yr
            df_cluster['kernel'] = kernel
            df_cluster['cluster'] = df_cluster.index.values
            df_cluster['conf_thresh'] = conf_thresh
            

            df_nodules = df_nodules.append(df_cluster, sort=False)
            df_nodules = df_nodules.reset_index(drop=True)
            
            # save on each loop in case script crashes
            df_nodules.to_csv(os.path.join(parser.save_dir, nodules_fn), index=False)
            
            print(' - {} clusters: {}. time: {} secs'.format(pid_yr, len(df_cluster), round((time.time()-time_pid_yr0), 2)))


        print('CSV total time: {} min'.format(round((time.time()-time_csv0)/60, 2)))


if __name__ == '__main__':
    time_total0 = time.time()
    
    main()
    
    print('-'*30)
    print('Total Time: {} min'.format(round((time.time()-time_total0)/60, 2)))
    
    
    
    