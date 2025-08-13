# Multi-Scale-Dual-Cross-Attention-Network-for-Pulmonary-Nodule-Segmentation
Title:
MSDC-Net: Multi-Scale Dual Cross Attention Network for Pulmonary Nodule Segmentation.

Description:
Overview of the codebase for implementing MSDC-Net, a hybrid CNN–Transformer model with Multi-Scale Shared Attention (MSSA) and Cross Fusion Dual Attention (CFDA) modules for pulmonary nodule segmentation using the LIDC-IDRI dataset.

Dataset Information:
Dataset: LIDC-IDRI (Lung Image Database Consortium and Image Database Resource Initiative).
Format: DICOM images with corresponding binary masks annotated by four radiologists.
Source: The Cancer Imaging Archive (TCIA).
Link: https://www.kaggle.com/datasets/zhangweiled/lidcidri
Total Images: 13491 annotated slices derived from 1018 CT scans.

Code Information:
Implemented in Python 3.10 using PyTorch and relevant deep learning libraries.
Includes modules for data loading, preprocessing, model definition, training, evaluation, and visualization.

Usage Instructions:
Clone the repository and install dependencies listed in requirements.txt.
Place the dataset in the /data directory.
Run train.py to train the model.
Use test.py to evaluate on the test set and generate segmentation results.

Requirements:
Python 3.10
PyTorch ≥ 1.13
CUDA-enabled GPU (recommended)
Dependencies: NumPy, OpenCV, scikit-image, scikit-learn, matplotlib, pydicom, albumentations

Methodology:
Preprocessing includes resizing to 128×128, intensity normalization, noise reduction, CLAHE contrast enhancement, and data augmentation (rotation, flipping, intensity shift).
Training with Adam optimizer (LR = 0.001) using hybrid Dice + BCE loss.
Evaluation metrics: Dice Similarity Coefficient (DSC), Hausdorff Distance (HD), and Intersection over Union (IoU).

Citations:
The LIDC-IDRI dataset must be cited as:
Armato III, S.G., et al. “The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI): A Completed Reference Database of Lung Nodules on CT Scans.” Medical Physics, 38(2): 915–931, 2011.
