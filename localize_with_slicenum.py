#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 16:31:45 2020

@author: Yesh
"""

import os
#os.chdir('/Users/Yesh/Documents/BDRAD/chest_ct_projects/lung_nodule_slice_localizer')
import numpy as np
import pandas as pd

import time
import argparse
from stl import mesh

import sys
sys.path.append('../pytorch-retinanet')
from dataloader_NLST import NLSTDataset, Normalizer, Resizer
from torchvision import transforms

np.random.seed(42)

def get_inferences(datadir):
    data = pd.DataFrame()
    csvs = [f for f in os.listdir(datadir) if f.endswith('.csv')]

    for csv in csvs:
        d = pd.read_csv(os.path.join(datadir,csv))
        data = data.append(d, ignore_index=True)

    return data


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


def main(args=None):
    parser = argparse.ArgumentParser(description='Identify nodule using axial slice number')
    parser.add_argument('--inferences_dir', help='Path to Directory containing NLST raw inferences', default='../data/NLST_Massive_Annotation_No_Masking_Results_2020_06_07_CoronalAxial')
    parser.add_argument('--nodules_filepath', help='Path to file containing NLST nodule candidates', default='../data/NLST_Clustered_Massive_Annotation_No_Masking_Results_2020_06_07_CoronalAxial_fp_reducer/nodules.csv')
    parser.add_argument('--nlst_dir', help='Directory of NLST data', default='/Volumes/Yesh2020/nlst-ct')
    parser.add_argument('--nlst_anns', help='Path to NLST annotations', default='../data/nlst_annotations/all_14_datasets/cdas_delivery_csv/sctabn.csv')
    parser = parser.parse_args()


    # reads annotations
    anns = pd.read_csv(parser.nlst_anns)
    anns['pidyr'] = anns['pid'].astype(str) + '_' + anns['STUDY_YR'].astype(str)
    anns = anns.dropna(subset=['sct_slice_num'])
    anns = anns[anns['pid'] > 100380] # pids before this were used with fp_reducer and must be excluded


    # get predicted nodules
    nodules = pd.read_csv(parser.nodules_filepath)
    nodules['pidyr'] = nodules['pid'].astype(str) + '_' + nodules['study_yr'].astype(str)
    nodules = nodules.dropna(subset=['confidence_max'])

    # keep only anns with pidyrs that we've clustered (including nodules with no clusters)
    anns = anns[anns['pidyr'].isin(nodules['pidyr'])]


    # get test set
    saved_test_pids = 'anns_test_pids.csv'
    if os.path.exists(saved_test_pids):
        anns = pd.read_csv(saved_test_pids)
    else:
        # randomize by pid & only select 1 year per pid
        test_pidyrs = anns.sample(3).groupby('pid').first()['pidyr'].values
        anns = anns[anns['pidyr'].isin(test_pidyrs)]
        
        assert(os.path.exists(saved_test_pids) == False)
        anns.to_csv(saved_test_pids, index=False)


    # get raw inferences
    inferences = get_inferences(parser.inferences_dir)
    inferences['pidyr'] = inferences['pid'].astype(str) + '_' + inferences['study_yr'].astype(str)

    # get axial conversion value
    inference_grp = inferences.groupby(['pidyr', 'axial_conversion']).size().reset_index().rename(columns={0:'count'})
    del inference_grp['count']

    # merge to axial_conversion information
    nodules = nodules.merge(inference_grp, on='pidyr')


    nlst = NLSTDataset(fp_anns=anns,
           data_dir=parser.nlst_dir,
           transform=transforms.Compose([Normalizer(), Resizer()]))


    df_final_nodules = pd.DataFrame()
    for pidyr in np.unique(anns['pidyr']):
        print(pidyr)
        ann = anns[anns['pidyr'] == pidyr]
        nodule_cands = nodules[nodules['pidyr'] == pidyr]

        if len(nodule_cands) == 0:
                df_final_nodules = df_final_nodules.append(pd.DataFrame({'pidyr': ann['pidyr'],
                                                                         'gt_slice_num': ann['sct_slice_num'],
                                                                         'gt_abn_num': ann['SCT_AB_NUM']
                                                                         }), sort=False)
                continue


        pid = ann['pid'].values[0]
        study_yr = ann['STUDY_YR'].values[0]
        kernel = nodule_cands['kernel'].values[0]

        res = nlst.load_NLST_images(pid, study_yr, load_dcms=True)
        dcms = res[kernel]['dcms']

        slice_nums = np.array([int(dcm.InstanceNumber) for dcm in dcms])
        axial_positions = np.array([float(dcm.ImagePositionPatient[2]) for dcm in dcms])

        used_nodules_clusters = []
        cubes = []
        df_pidyr_nodules = pd.DataFrame()

        for idx, row in ann.iterrows():
            # drop already identified nodules in this pidyr
            nods = nodule_cands[~nodule_cands['cluster'].isin(used_nodules_clusters)]

            # get axial world position for GT slice
            slice_num_gt = int(row['sct_slice_num'])
            gt_abn_num = row['SCT_AB_NUM']

            try:
                axial_gt = axial_positions[np.where(slice_nums == slice_num_gt)[0]][0]
            except IndexError as e:
                print(e)
                df_pidyr_nodules = df_pidyr_nodules.append(pd.DataFrame({'pidyr': [pidyr],
                                                                         'gt_slice_num': [slice_num_gt],
                                                                         'gt_axial_world_coord': [np.nan],
                                                                         'comment': ['IndexError when trying to get gt_axial_world_coord. Check sct_slice_num manually'],
                                                                         'gt_abn_num': gt_abn_num,
                                                                         }), sort=False)
                continue


            gt_location = None
            if row['SCT_EPI_LOC'] == 1:
                gt_location = 'Right Upper Lobe'
            elif row['SCT_EPI_LOC'] == 2:
                gt_location = 'Right Middle Lobe'
            elif row['SCT_EPI_LOC'] == 3:
                gt_location = 'Right Lower Lobe'
            elif row['SCT_EPI_LOC'] == 4:
                gt_location = 'Left Upper Lobe'
            elif row['SCT_EPI_LOC'] == 5:
                gt_location = 'Lingula'
            elif row['SCT_EPI_LOC'] == 6:
                gt_location = 'Left Lower Lobe'



            # get the closest nodules
            nods.loc[:,'dist_to_label'] = np.abs(nods['axial_world_coord_mean'] - axial_gt)

            # nodules must be within 20mm
            nods = nods[nods['dist_to_label'] <= 20]

            # get X highest proba nodule to the axial position, change 1 to N to get N largest
            nods = nods[nods['dist_to_label'].isin(nods['dist_to_label'].nsmallest(1))]
            nods = nods.sort_values(by='dist_to_label')



            if len(nods) == 0:
                df_pidyr_nodules = df_pidyr_nodules.append(pd.DataFrame({'pidyr': [pidyr],
                                                                         'gt_slice_num': [slice_num_gt],
                                                                         'gt_axial_world_coord': [axial_gt],
                                                                         'gt_location': [gt_location],
                                                                         'gt_abn_num': gt_abn_num,
                                                                         }), sort=False)
                continue

            nods['gt_slice_num'] = slice_num_gt
            nods['gt_axial_world_coord'] = axial_gt
            nods['gt_location'] = gt_location
            nods['gt_abn_num'] = gt_abn_num 

            df_pidyr_nodules = df_pidyr_nodules.append(nods, sort=False)
            used_nodules_clusters += [int(cluster) for cluster in list(nods['cluster'])]


            # create mesh cubes for visualization on slicer - loop b/c if we want to predict "top 3" then we need to loop
            for n, nod in nods.iterrows():
                dx, dy, dz = nod['diameter_max'], nod['diameter_max'], nod['diameter_max']

                corner = np.array([0,0,0])
                corner[0] = nod['saggital_world_coord_mean'] - dx/2
                corner[1] = nod['coronal_world_coord_mean'] - dy/2
                corner[2] = nod['axial_world_coord_mean'] - dz/2


                cube = createMeshCube(corner, dx, dy, dz)
                cubes.append(cube)


        # save cube to directory, grouped into folders with same name as the csv
        cube_dir = 'Nodule_Masks'
        if not os.path.exists(cube_dir):
            os.mkdir(cube_dir)

        if len(cubes) != 0:
            cubes = mesh.Mesh(np.concatenate([c.data for c in cubes]))

            cubes.save(os.path.join(cube_dir,  '{}_{}_{}.stl'.format(pid, study_yr, kernel)))


        # add to final df and used nodules
        df_final_nodules = df_final_nodules.append(df_pidyr_nodules, sort=False)
        df_final_nodules.to_csv(os.path.join(cube_dir, 'nodules_test_set.csv'), index=False)



if __name__ == '__main__':
    time_total0 = time.time()

    main()

    print('-'*30)
    print('Total Time: {} min'.format(round((time.time()-time_total0)/60, 2)))
