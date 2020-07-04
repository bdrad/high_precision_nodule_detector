from __future__ import print_function, division
import sys
import os, os.path
import torch
import numpy as np
import pydicom
import pandas as pd
import math

from torch.utils.data import Dataset

import skimage.io
import skimage.transform
import skimage.color
import skimage

local_mip_utils_path = '/Users/Yesh/Documents/BDRAD/chest_ct_projects/lung-nodule-tracker/utils'
if os.path.exists(local_mip_utils_path):
    sys.path.insert(1, local_mip_utils_path)
else:
    sys.path.insert(1, '/home/bdrad1/chest_ct_projects/lung-nodule-tracker/utils')

import Mip_Utility as mip_utils # utils for LUNA data



"""
The building block for NLST dataloader, it contains the annotations and the
corresponding patient CT.
The CTs for different patient and study year are loaded upon request and
the annotations are loaded only once.
- Returns a dict containing sboth the axial and coronal MIP projects for each kernel
    General Schema for Return Dict: kernel: {view: {scan_with_mip_project, details}}
"""
class NLSTDataset(Dataset):
    def __init__(self, fp_anns, data_dir, mip=25, nodule_cts_only=False, transform=None, all_kernels=True, voxel_transform=True,mask=True):
        """
        fp_anns = filepath to 'sctabn.csv' annotations file, or pandas dataframe object itself
        data_dir = directory to NLST folder containing pids & CT images
        pids = Optional. Integer to specify first N pids OR array of specific PIDs you want to test
                (e.g. pids=100 OR pids=[100002, 100004, 100005, ...])
        mip = Apply Maximum Intensity Projections, units are in 1mm (i.e. mip=25 -> 25mm MIP applied)
        nodule_cts_only = Use only CTs with a nodule.
        no_voxel_transform = Apply 1mm voxel transformation to speed up preprocessing
        mask = Apply lung mask to segment out lungs using voxel thresholding
        """
        self.all_kernels = all_kernels
        self.voxel_transform = voxel_transform
        self.mask=mask

        if isinstance(fp_anns, str):
            self.anns = pd.read_csv(fp_anns)
        elif isinstance(fp_anns, pd.DataFrame):
            self.anns = fp_anns
        else:
            raise ValueError("fp_anns parameter must be the path to the csv (str) or a pandas dataframe object")

        # TODO: After moving the code for removing examples which aren't present in the local dataset, this code may break
        # double check this code when running analysis script
        self.nodule_cts_only = nodule_cts_only
        if nodule_cts_only:
            self.anns = self.get_nodule_cts_only(self.anns)

        # remove duplicates from annotations b/c we use this to loop through studies
        self.anns = self.anns.drop_duplicates(subset=['pid', 'STUDY_YR'])

        self.data_dir = data_dir
        self.mip = mip
        self.transform = transform


    def __len__(self):
        return len(self.anns)

    def get_nodule_cts_only(self, ann):
        return ann[(ann['sct_slice_num'] > 0) & (ann['sct_slice_num'] != 999)]


    def convert_to_coronal(self, img_a):
        # img_a is a numpy array of axial slices of shape (n_axial_slices, x, y)
        # img_c is returned which is numpy array of coronal slices of shape (x, n_axial_slices, y), where x = n_coronal_slices
        slices, x, y = img_a.shape
        img_c = np.zeros([x, slices, y])
        for i, s in enumerate(img_a):
            img_c[:,i,:] = s
        return img_c


    def load_NLST_images(self, pid, study_yr, load_dcms=False):
        """
        pid = patient id
        study_yr = 0, 1, 2, ...
        load_dcms = True if you want func to return pydicom objects along with img_arrays

        Returns:
            1: an array of numpy arrays containing the patient PID's scan in
            study_yr
            2: spcing for adjusting slice thickness in mip
        """
        data_dir = self.data_dir

        pid_dir = os.path.join(data_dir,str(pid))
        if os.path.isdir(pid_dir) == False:
            return {}

        study_yr = 'T' + str(study_yr)
        study_dir = os.path.join(pid_dir,study_yr)

        """
        Return a map of all images with kenel name, so far
        the function could potentially be returning different datastructures
        but could optimize a bit later.
        """

        mapping = {}
        for kernel_foldername in self.listdir_nohidden(study_dir):
            dcms = []
            file_paths = []  # May just want to append kernel_dir instead
            kernel = None
            kernel_dir = os.path.join(study_dir, kernel_foldername)

            # check if the folder contains scout images
            if len([name for name in os.listdir(kernel_dir)]) > 10:
                for filename in self.listdir_nohidden(kernel_dir):
                    dcm_path = os.path.join(kernel_dir, filename)
                    ds = pydicom.dcmread(dcm_path, force = True)
                    try:
                        kernel = ds.ConvolutionKernel
                    except Exception as e:
                        print('-'*30)
                        print('pid: {} study_yr: {}'.format(pid, study_yr))
                        print('Kernel foldername: {}'.format(kernel_foldername))
                        print(e)
                    kernel = ds.ConvolutionKernel
                    dcms.append(ds)
                    file_paths.append(dcm_path)

                spacing = np.array([float(ds.SliceThickness), float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1])])

                # reorder dcms by patient position
                dcms.sort(key = lambda dcm : -dcm.ImagePositionPatient[2]) # reorder dicoms along z (axial) axis

                # add to return object with kernel as key
                # - if multiple versions of the same kernel, return only the one with the smallest SliceThickness

                # check if new kernel with same the kernel_name has a larger sliceThickness. If so, skip adding this kernel
                if kernel.lower() in mapping.keys():
                    if spacing[0] > mapping[kernel.lower()]['spacing'][0]:
                        continue

                mapping[kernel.lower()] = {}
                mapping[kernel.lower()]['file_path'] = kernel_dir
                mapping[kernel.lower()]['image_position_patient'] = dcms[0].ImagePositionPatient
                mapping[kernel.lower()]['spacing'] = spacing
                mapping[kernel.lower()]['img_array'] = np.array([dcm.pixel_array for dcm in dcms])

                if load_dcms:
                    mapping[kernel.lower()]['dcms'] = dcms


        return mapping


    def __getitem__(self, index):
        """
        Returns mapping = {
                kernel_name1: {
                        img_array: all ct slices preprocessed,
                        slice_num_new: slice number of nodule after preprocessing,
                        slice_num_orig: slice number of nodule before preprocessing,
                        pid: patient ID,
                        study_yr: Study Year,
                },
                kernel_name2: {...},
                ...
        }

                2. adjusted slice
            3. original slice number of nodule
        """
        # select sample
        mip = self.mip
        ann = self.anns.iloc[index]
        pid = ann['pid']
        study_yr = ann['STUDY_YR']

        print('-'*30)
        print('Preprocessing: {}_{}'.format(pid, study_yr))

        mapping = self.load_NLST_images(pid, study_yr)

        output = {}

        # inference on preferenced kernel, otherwise inference on all kernels
        # - preference list in order from highest to lower preference
        preferenced_kernels = ['standard', 'bone', 'b50f', 'b30f', 'd', 'c']

        # get the kernel that matches highest on the preference list
        kernels =  next(([k] for k in preferenced_kernels if k in mapping.keys()), list(mapping.keys()))

        for kernel in kernels:
            item = mapping[kernel]
            img_array = item['img_array']
            spacing = item['spacing']
            file_path = item['file_path']

            # Resize slice (this step takes a long time, about 70 seconds)
            if self.voxel_transform:
                img_array, new_spacing = mip_utils.resize_image(img_array, spacing)
                axial_conversion = (spacing/new_spacing)[0]
                saggital_conversion = (spacing/new_spacing)[1]
                coronal_conversion = (spacing/new_spacing)[2]
            else:
                axial_conversion = spacing[0]
                saggital_conversion = spacing[1]
                coronal_conversion = spacing[2]

            # segment lungs
            if self.mask:
                img_array = mip_utils.segment_lung_from_ct_scan(img_array)

            # apply transformations to both axial and coronal views
            view_output = {}
            views = ['axial', 'coronal']
            for view in views:
                # switch axis to coronal
                if view == 'axial':
                    img_array_view = img_array
                elif view == 'coronal':
                    img_array_view = self.convert_to_coronal(img_array)

                # scale mip to appropriate value if not transforming to 1mm voxels (i.e. mip_utils.resize_image function)
                if not self.voxel_transform:
                    if view == 'axial':
                        mip = int(mip/axial_conversion)
                    elif view == 'coronal':
                        mip = int(mip/coronal_conversion)

                # Apply MIP - must be run after mip_utils.resize_image()!!!
                img_array_view = mip_utils.createMIP(img_array_view, slices_num=mip)

                # Convert to 3 channels
                # img_array= np.stack((img_array,) * 3, axis=-1)
                img_array_view = np.expand_dims(img_array_view, axis=-1)

                # scale to 0-1
                img_array_view = ((img_array_view - img_array_view.min()) * (1/(img_array_view.max() - img_array_view.min())))

                # transforms
                if not (self.transform is None):
                    # Transforms are applied to each slice
                    transformed_img_array = []

                    for img in img_array_view:
                        sample = {'img': img}
                        sample = self.transform(sample)
                        scale = sample['scale']
                        x = torch.unsqueeze(sample['img'], 0)

                        transformed_img_array.append(x)

                    transformed_img_array = torch.cat(transformed_img_array, 0)
                    img_array_view = transformed_img_array

                view_output[view] = {
                        'scale' : scale,
                        'img_array': img_array_view,
                        'axial_conversion': axial_conversion,
                        'saggital_conversion': saggital_conversion,
                        'coronal_conversion': coronal_conversion,
                        'pid': pid,
                        'study_yr': study_yr,
                        'file_path': file_path}

                # cleanup to reduce CPU RAM use
                del img_array_view
            del img_array

            output[kernel] = view_output
        return output


    def listdir_nohidden(self, path):
        for f in os.listdir(path):
            if not f.startswith('.'):
                yield f


def collater(data):

    imgs = [s['img'] for s in data]
    scales = [s['scale'] for s in data]

    widths = [int(s.shape[0]) for s in imgs]
    heights = [int(s.shape[1]) for s in imgs]
    batch_size = len(imgs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)

    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img


    padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    return {'img': padded_imgs, 'scale': scales}


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, min_side=608, max_side=1024):
        image = sample['img']

        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        image = skimage.transform.resize(image, (int(round(rows*scale)), int(round((cols*scale)))),
                                         mode='constant',
                                         anti_aliasing=None)
        rows, cols, cns = image.shape

        pad_w = 32 - rows%32
        pad_h = 32 - cols%32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)


        return {'img': torch.from_numpy(new_image), 'scale': scale}



class Normalizer(object):
    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):
        """
        Normalizes sample based on provided mean and std statistics
        """
        image = sample['img']
        num_channels = image.shape[2]
        if num_channels == 1:
            mean = self.mean[:, :, 0]
            std = self.std[:, :, 0]

        return {'img':((image.astype(np.float32)-mean)/std)}

class UnNormalizer(object):
    def __init__(self, mean=None, std=None):
        if mean == None:
            self.mean = [0.485, 0.456, 0.406]
        else:
            self.mean = mean
        if std == None:
            self.std = [0.229, 0.224, 0.225]
        else:
            self.std = std

    def __call__(self, tensor):
        """
        Undoes normalization from Normalizer transformation
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Un-Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor
