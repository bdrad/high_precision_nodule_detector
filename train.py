import time
import os
import copy
import argparse
import pdb
import collections
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import torchvision

from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import matplotlib.pyplot as plt
import matplotlib.patches as patches


import model
from anchors import Anchors
import losses
from dataloader import CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import csv_eval

#assert torch.__version__.split('.')[1] == '4'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    """
    In current implementation, if test csv is provided, we use that as validation set and combine the val and train csv's 
    as the csv for training.

    If train_all_labeled_data flag is use, then we combine all 3 (if test is provided) for training and use a prespecified learning rate step schedule.
    """

    parser     = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)', default=None)
    parser.add_argument('--csv_test', 
                        help='Path to file containing test annotations (optional, if provided, train & val will be combined for training and test will be used for evaluation)', 
                        default=None)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=101)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=25)
    parser.add_argument('--model_output_dir', type=str, default='models')
    parser.add_argument('--train_all_labeled_data', 
                        help='Combine train, val, and test into 1 training set. Will use prespecified learning rate scheduler steps',
                        action='store_true')
    parser.add_argument('--resnet-backbone-normalization', choices=['batch_norm', 'group_norm'], type=str, default='batch_norm')

    parser = parser.parse_args(args)

    print('Learning Rate: {}'.format(parser.lr))
    print("Normalization: ", parser.resnet_backbone_normalization)


    # Create folder - will raise error if folder exists
    assert(os.path.exists(parser.model_output_dir) == False)
    os.mkdir(parser.model_output_dir)

    if parser.csv_train is None:
        raise ValueError('Must provide --csv_train when training,')

    if parser.csv_classes is None:
        raise ValueError('Must provide --csv_classes when training,')

    if not parser.csv_val and parser.csv_test:
        raise ValueError("Cannot specify test set without specifying validation set")

    if parser.train_all_labeled_data:
        csv_paths = [parser.csv_train, parser.csv_val, parser.csv_test]
        train_csv = []
        for path in csv_paths:
            if isinstance(path, str):
                train_csv.append(path)
        val_csv = None
    else:
        if parser.csv_train and parser.csv_val and parser.csv_test:
            train_csv = [parser.csv_train, parser.csv_val]  # Combine train and val sets for training
            val_csv = parser.csv_test
        else:
            train_csv = parser.csv_train
            val_csv = parser.csv_val

    print('loading train data')
    print(train_csv)
    dataset_train = CSVDataset(train_file=train_csv, class_list=parser.csv_classes,
                        transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
    print(dataset_train.__len__())

    if val_csv is None:
        dataset_val = None
        print('No validation annotations provided.')
    else:
        dataset_val = CSVDataset(train_file=val_csv, class_list=parser.csv_classes,
                        transform=transforms.Compose([Normalizer(), Resizer()]))

    print('putting data into loader')
    sampler = AspectRatioBasedSampler(dataset_train, batch_size=2, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)

    # Create the model
    print('creating model')
    if parser.depth == 18:
        retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True, normalization=parser.resnet_backbone_normalization)
    elif parser.depth == 34:
        retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True, normalization=parser.resnet_backbone_normalization)
    elif parser.depth == 50:
        retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True, normalization=parser.resnet_backbone_normalization)
    elif parser.depth == 101:
        retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True, normalization=parser.resnet_backbone_normalization)
    elif parser.depth == 152:
        retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True, normalization=parser.resnet_backbone_normalization)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    use_gpu = True

    if use_gpu:
        retinanet = retinanet.cuda()

    retinanet = torch.nn.DataParallel(retinanet).cuda()

    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=parser.lr)

    lr_factor = 0.3
    if not parser.train_all_labeled_data:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=lr_factor, verbose=True)
    else:
        # these milestones are for when using the lung masks - not for unmasked lung data
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[12, 16, 20, 24], gamma=lr_factor) # masked training
        #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[14, 18, 22, 26], gamma=lr_factor)

    loss_hist = collections.deque(maxlen=500)

    retinanet.train()
    retinanet.module.freeze_bn()


    #initialize tensorboard
    writer = SummaryWriter(comment=parser.model_output_dir)

    # Augmentation
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.Affine(
            scale={"x": (1.0, 1.2), "y": (1.0, 1.2)},
            rotate=(-20, 20),
            shear=(-4, 4)
        )
    ], random_order=True)


    def augment(data, seq):
        for n, img in enumerate(data['img']):
            # imgaug needs dim in format (H, W, C)
            image = data['img'][n].permute(1, 2, 0).numpy()

            bbs_array = []
            for ann in data['annot'][n]:
                x1, y1, x2, y2, _ = ann
                bbs_array.append(BoundingBox(x1=x1,y1=y1,x2=x2,y2=y2))

            bbs = BoundingBoxesOnImage(bbs_array, shape=image.shape)
            image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)

            # save augmented image and chage dims to (C, H, W)
            data['img'][n] = torch.tensor(image_aug.copy()).permute(2, 0, 1)

            # save augmented annotations
            for i, bbox in enumerate(bbs_aug.bounding_boxes):
                x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
                obj_class = data['annot'][n][i][-1]
                data['annot'][n][i] = torch.tensor([x1, y1, x2, y2, obj_class])

        return data

    print('Num training images: {}'.format(len(dataset_train)))
    dir_training_images = os.path.join(os.getcwd(), writer.log_dir, 'training_images')
    os.mkdir(dir_training_images)

    best_validation_loss = None
    best_validation_map = None

    for epoch_num in range(parser.epochs):

        writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], epoch_num)
        
        retinanet.train()
        retinanet.module.freeze_bn()

        epoch_loss = []

        for iter_num, data in enumerate(dataloader_train):
            try:
                optimizer.zero_grad()

                data = augment(data, seq)

                # save a few training images to see what augmentation looks like
                if iter_num % 100 == 0 and epoch_num == 0:
                    x1, y1, x2, y2, _ = data['annot'][0][0]

                    fig, ax = plt.subplots(1)
                    ax.imshow(data['img'][0][1])
                    rect = patches.Rectangle((x1,y1),x2-x1,y2-y1,linewidth=1,edgecolor='r',facecolor='none', alpha=1)
                    ax.add_patch(rect)
                    fig.savefig(os.path.join(dir_training_images, '{}.png'.format(iter_num)))
                    plt.close()


                classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])

                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()

                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue

                loss.backward()

                if parser.resnet_backbone_normalization == 'batch_norm':
                    torch.nn.utils.clip_grad_norm_(parameters=retinanet.parameters(), max_norm=0.1)
                else:
                    torch.nn.utils.clip_grad_norm_(parameters=retinanet.parameters(), max_norm=0.01)  # Decrease norm to reduce risk of exploding gradients

                optimizer.step()

                loss_hist.append(float(loss))

                epoch_loss.append(float(loss))

                print('Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))

                del classification_loss
                del regression_loss
            except Exception as e:
                print(e)
                continue

        writer.add_scalar('Train/Loss', np.mean(epoch_loss), epoch_num)

        if not parser.train_all_labeled_data:
            print('Evaluating Validation Loss...')
            with torch.no_grad():
                retinanet.train()
                val_losses, val_class_losses, val_reg_losses = [],[],[]
                for val_iter_num, val_data in enumerate(dataloader_val):
                    try:
                        val_classification_loss, val_regression_loss = retinanet([val_data['img'].cuda().float(), val_data['annot']])
                        val_losses.append(float(val_classification_loss)+float(val_regression_loss))
                        val_class_losses.append(float(val_classification_loss))
                        val_reg_losses.append(float(val_regression_loss))
                        del val_classification_loss, val_regression_loss
                    except Exception as e:
                        print(e)
                        continue
                print('VALIDATION Epoch: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Total loss: {:1.5f}'.format(epoch_num, np.mean(val_class_losses), np.mean(val_reg_losses), np.mean(val_losses)))

                # Save model with best validation loss
                if best_validation_loss is None:
                    best_validation_loss = np.mean(val_losses)
                if best_validation_loss >= np.mean(val_losses):
                    best_validation_loss = np.mean(val_losses)
                    torch.save(retinanet.module, parser.model_output_dir + '/best_result_valloss.pt')

                writer.add_scalar('Validation/Loss', np.mean(val_losses), epoch_num)

                # Calculate Validation mAP
                print('Evaluating validation mAP')
                mAP = csv_eval.evaluate(dataset_val, retinanet)
                print("Validation mAP: " + str(mAP[0][0]))
                if best_validation_map is None:
                    best_validation_map = mAP[0][0]
                elif best_validation_map < mAP[0][0]:
                    best_validation_map = mAP[0][0]
                    torch.save(retinanet.module, parser.model_output_dir + '/best_result_valmAP.pt')

                writer.add_scalar('Validation/mAP', mAP[0][0], epoch_num)

        if not parser.train_all_labeled_data:
            scheduler.step(np.mean(val_losses))
        else:
            scheduler.step()

        torch.save(retinanet.module, parser.model_output_dir + '/retinanet_{}.pt'.format(epoch_num))

    retinanet.eval()

    torch.save(retinanet, parser.model_output_dir + '/model_final.pt')


if __name__ == '__main__':
 main()
