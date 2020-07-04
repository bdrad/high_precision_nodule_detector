# High Precision Localization of Pulmonary Nodules on Chest CT utilizing Axial Slice Number Labels
This is a repository accompanying the manuscript of the same name.

### Note:
This model utilizes the pytorch Retinanet implementation available at https://github.com/yhenon/pytorch-retinanet/. The relevant code has been included in this repo, but see their documentation for further information.

# Data:
To train and test model, users will need to obtain access to LUng Nodule Analysis (LUNA) data and National Lung Cancer Screening Trial (NLST) data.

# Installation Instructions:
1. Clone this repository

2. Install pytorch and torchvision: https://pytorch.org/

3. Install https://github.com/yhenon/pytorch-retinanet/ dependencies:
```
pip install cffi
pip install pandas
pip install cython
pip install opencv-python
pip install requests
```

4. Install additional python dependencies
- numpy
- matplotlib
- argparse
- pydicom
- skimage
- SimpleITK
- scipy
- XGBoost
- sklearn
- imgaug
- joblib
- csv


# Pipeline:
1. Generate Maximum Intensity Projection (MIP) LUNA data:

 ```
 generate_data_LUNA_coronalaxial_combined.py \
  --LUNA_DATA_DIR "directory to LUNA .mhd files" \
  --LUNA_ANNS "directory to LUNA annotations" \
  --no_masking \
  --mip 25 \
  --positives_only \
  --SAVE_DIR "directory where to save formatted LUNA data" \
 ```

2. Train retinanet model:
```
train.py \
  --csv_train "path to annotations_train.csv in the formatted LUNA data directory" \
  --csv_classes "path to classes.csv in the formatted LUNA data directory" \
  --csv_val "path to annotations_val.csv in the formatted LUNA data directory" \
  --csv_test "path to annotations_test.csv in the formatted LUNA data directory" \
  --depth 101 \
  --model_output_dir "directory to save model" \
  --lr 1e-5 \
  --epochs 30 \
  --train_all_labeled_data
```
Note: csv_train, csv_val, and csv_test will be grouped together for training if `--train_all_labeled_data` is specified. Otherwise, train and val will be grouped to become the train data, and test set will become the val data. No test data will be used.

3. Inference raw NLST data:
```
analysis_nlst.py \
  --datadir "path to NLST CT data" \
  --anns "path to NLST nodule annotations 'sctabn.csv'" \
  --model "path to saved from from train.py" \
  --results_dir "directory where to save raw inference results" \
  --mip 25 \
  --no_mask \
  --pids "(Optional) specify specific PIDs to inference on"
```

4. Perform DBSCAN unsupervised clustering and apply False Positive Reducer:
```
clustering.py \
  --nlst_dir "path to NLST CT data" \
  --nlst_anns "path to NLST nodule annotations 'sctabn.csv'" \
  --inference_dir "directory where raw inference results were saved" \
  --save_dir "directory where to save clustered & fp_reduced results" \
  --conf_thresh 0.10 \
  --fp_reducer './fp_reducer_final.joblib' # path to FP reducer (XGBoost Classifier)
```

5. Apply axial slice localizer:
```
localize_with_slicenum.py \
  --inferences_dir "directory where raw inference results were saved" \
  --nodules_filepath "where clustered nodule results \ (nodules.csv) from clustering.py are located" \
  --nlst_dir "path to NLST CT data" \
  --nlst_anns "path to NLST nodule annotations 'sctabn.csv'" \
```
