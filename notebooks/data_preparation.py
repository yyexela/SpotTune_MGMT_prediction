# # Data Preparation
# The purpose of this notebook is to create the npy files that will be used for training purposes
import os
from datetime import datetime
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from collections import OrderedDict
import SimpleITK as sitk
#import logging
#logging.getLogger("tensorflow").setLevel(logging.ERROR)
from collections import OrderedDict
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets

import pickle, subprocess
import scipy
import sklearn
import csv

import torchmetrics

from pathlib import Path
import sys
pkg_path = str(Path(os.path.abspath('')).parent.absolute())
print(pkg_path)
sys.path.insert(0, pkg_path)

#import initial_ml as iml
import data_prep as dp
from pytorch.run_model_torch import RunModel
from pytorch import resnet_spottune as rs
from MedicalNet.models import resnet

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)
print(f"using {device} device")
#torch.backends.cudnn.benchmark = False
#torch.use_deterministic_algorithms(True)

csv_dir = '/home/ee577/project/Datasets/UPENN_GBM/radiomic_features_CaPTk/'
image_dir = '/home/ee577/project/Datasets/UPENN_GBM/PKG-UPENN-GBM-NIfTI/UPENN-GBM/NIfTI-files'

# The modality to create npy files out of. The numpy file will have a different derivative at each index
#modality = 'struct'
#modality = 'DSC'
modality = 'DTI'

# The output directory of the npy files
out_dir = f'../../data/upenn_GBM/numpy_conversion_{modality}_channels/'

# Specify the derivatives to put into the npy file, this corresponds to the order of the derivatives when using 'channel_idx' as a reference in the training notebook
derivatives = {
    'struct': ['T2', 'FLAIR', 'T1', 'T1GD'],
    'DTI':  ['DTI_AD', 'DTI_FA', 'DTI_RD', 'DTI_TR'],
    'DSC':['DSC_ap-rCBV', 'DSC_PH', 'DSC_PSR'],
}
#scale_file = 'image_scaling_'+modality+'.json'

#auto_df, man_df, comb_df = dp.retrieve_data(csv_dir, modality=modality)
#patients = pd.DataFrame(auto_df.iloc[:, 0:2])
#classifier = 'Survival_from_surgery_days'
classifier = 'MGMT'
patients = dp.retrieve_patients(csv_dir, image_dir, modality='DTI', classifier=classifier)

#image_df = dp.retrieve_image_data(patients, modality=modality, image_dir_=image_dir)
path_df = dp.convert_image_data_mod(patients, modality=derivatives[modality], 
                                image_dir_=image_dir, out_dir=out_dir,
                                image_type='autosegm',
                                #scale_file=scale_file,
                                window=(140, 172, 164),
                                pad_window=(70, 86, 86),
                                #window=(64, 80, 60),
                                base_dim=(155, 240, 240), downsample=True,
                                window_idx = ((0, 140), (39, 211), (44,208)), down_factor=0.5,
                                augments = ['base'])


