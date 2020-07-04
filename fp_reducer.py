#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 15:12:29 2020

@author: Yesh
"""


import os # os.chdir('/Users/Yesh/Documents/BDRAD/chest_ct_projects/pytorch-retinanet')
import numpy as np
import pandas as pd
import time
import argparse
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, classification_report
from joblib import dump


plt.rcParams['figure.figsize'] = [12, 6]

"""
Train ML model to reduce false positives using clustering metadata
Ideas:
     - could improve by including a parameter that measures distance from top in mm to remove the lesions located in stomach
"""

def hist_compare(data, var):
    # helper function to compare corrects vs incorrects for visual comparison
    plt.hist([data[data['correct_pred'] == 0][var],
              data[data['correct_pred'] == 1][var]], 
             label=['incorrect', 'correct'],)

def main(args=None):
    parser = argparse.ArgumentParser(description='Clustering on NLST Inference By pid and study_yr')
    parser.add_argument('--annotated_nodules', help='Path to Directory containing annotated clustered nodules', default='../../../../Google Drive/BDRAD/chest_ct_projects/nodules_annotations_yesh_for_fp_reducer_UnMaskedLungs_2020-06-07.csv')
#    parser.add_argument('--annotated_nodules', help='Path to Directory containing annotated clustered nodules', default='../../../../Google Drive/BDRAD/chest_ct_projects/nodules_annotated_yesh_for_fpreducer_MaskedLungs_2020-05-30.csv')
    parser.add_argument('--model', help='model to use', default='xgb')
    parser = parser.parse_args()
    
    seed = 42
    test_size = 0.15
    thresh = 0.5

    save_fp = './fp_reducer_unmasked.joblib'
    print('Saving FP Reducer at: {}'.format(save_fp))
    
    df = pd.read_csv(parser.annotated_nodules, header=0)
    df = df[df['correct_pred'].isin(['0',0,'1',1])]
    
    pidyr = df['pid'].astype(str) + '_' + df['study_yr'].astype(str)
    print('N Patients: {}'.format(len(np.unique(df['pid']))))
    print('N CTs: {}'.format(len(np.unique(pidyr))))
    print('N nodules: {}'.format(len(df)))
    
    df['correct_pred'] = df['correct_pred'].astype(int)
    df['correct_pred'].mean()
        
    # a bit of preprocessing
    df['is_seen_axial_and_coronal'] = np.where(df['is_coronal_mean'].isin([0,1]), 0, 1)
    df['dist_to_top'] = np.abs(df['axial_world_coord_mean'] - df['top_world_coord'])
    df['dist_to_bottom'] = np.abs(df['axial_world_coord_mean'] - df['bottom_world_coord'])
    
    # split into train and test by PID!!!
    train_pids, test_pids = train_test_split(np.unique(df['pid']), test_size=test_size, random_state=seed)
    df_train = df[df['pid'].isin(train_pids)].copy()
    df_test = df[df['pid'].isin(test_pids)].copy()
    
    
    # some data exploration
    #hist_compare(df_train, 'confidence_max')
    
    # select train vars
    train_variables = ['is_seen_axial_and_coronal', 'pred_slice_num_count', 'confidence_max', 
                       'confidence_mean','diameter_max', 'dist_to_top', 'dist_to_bottom']

    
    # Train on best params
    if parser.model == 'xgb':
        # Grid Search, previously attempted parameters in comments
        
        pos_ratio = sum(df_train['correct_pred']==0) / sum(df_train['correct_pred']==1)
        grid_params = grid_params = {'n_estimators': [75, 150, 200], # 1, 2, 4, 5, 6, 7,8,9,10,11,12, 14, 100, 1000
                                     'learning_rate': [0.01, 0.05,0.1,0.5], # 0.01, 0.05, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 1
                                     'max_depth': [2,4,6], # 1-4, 6, 8, 10
                                     'subsample': [0.6,0.8,1], # 0.8, 0.9, 1
                                     'colsample_bytree': [0.6,0.8,1], # 0.7, 1 
                                     'gamma': [0,1], # 0,1,5
                                     'max_delta_step': [0, 0.1, 0.3], #  0.01, 0.1, 0.3
                                     'scale_pos_weight': [1, np.sqrt(pos_ratio), pos_ratio],
                                     }
        
        model = XGBClassifier(random_state=seed)
        model_gridsearch = GridSearchCV(model, grid_params,
                                        cv=StratifiedKFold(n_splits=5, random_state=seed, shuffle=True),
                                        scoring = 'f1', verbose=1, n_jobs=-1)
        
        model_gridsearch.fit(X=df_train[train_variables],
                             y= df_train['correct_pred'])
    
        params = {**model_gridsearch.best_params_}
        print('Best GridSearchCV Score: {}'.format(model_gridsearch.best_score_))
        print(params)
        
        model = XGBClassifier(**params)
        model.fit(df_train[train_variables], df_train['correct_pred'])
        
        model_full = XGBClassifier(**params)
        model_full.fit(df[train_variables], df['correct_pred'])

    elif parser.model == 'logreg':
        grid_params = {'max_iter':[9999],
                       'class_weight': [None, 'balanced'],
                       'penalty':['l2', 'l1']}
        
        model = LogisticRegression()
        model = GridSearchCV(model, 
                             grid_params, 
                             cv=StratifiedKFold(n_splits=5, random_state=seed, shuffle=True),
                             scoring = 'roc_auc', verbose=1, n_jobs=-1)
        model.fit(X=df_train[train_variables],
                  y=df_train['correct_pred'])
        
        print('Best GridSearchCV Score: {}'.format(model.best_score_))
        print(model.best_params_)
        
        
        model_full = LogisticRegression(**model.best_params_)
        model_full.fit(df[train_variables], df['correct_pred'])
    
    
    
    
    # Test
    print(model)
    y_probas = model.predict_proba(df_test[train_variables])[:,1]
    y_preds = np.where(y_probas >= thresh, 1, 0)


    cm = confusion_matrix(df_test['correct_pred'], y_preds)
    cr = classification_report(df_test['correct_pred'], y_preds)
    accuracy = accuracy_score(df_test['correct_pred'], y_preds)
    rocauc = roc_auc_score(df_test['correct_pred'], y_probas)

    print('Accuracy: {}'.format(accuracy))
    print('ROC AUC: {}'.format(rocauc))
    print(cr)
    print('Classes: {}'.format(model.classes_))    
    print('Confusion Matrix: \n{}'.format(cm))
#    print('Model Feature Importances: \n{} \n{}'.format(train_variables, model.feature_importances_))

    
    df_test['pred_score'] = y_probas
    df_test_incorrects = df_test[df_test['correct_pred'] != y_preds]
    
    # plot hist
    hist_compare(df_test, 'pred_score')

    
    # Plot ROC AUC
    fpr, tpr, _ = roc_curve(df_test['correct_pred'], y_probas)    
    
    plt.figure()
    plt.style.use('ggplot')
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='AUC = {0:.4f})'.format(rocauc, 3))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right", prop={'size': 12})
    plt.show()
    
    
    # save full trained model
    print('-'*30)
    print('Save model:')
    print(model_full)
    dump(model_full, save_fp)


if __name__ == '__main__':
    time_total0 = time.time()
    
    main()
    
    print('-'*30)
    print('Total Time: {} min'.format(round((time.time()-time_total0)/60, 2)))
    
    