import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nilearn import image
from nilearn import plotting
from nilearn import datasets 
from nilearn.connectome import ConnectivityMeasure
import nibabel as nib
from nilearn import datasets

path = "/Users/leoschild/Desktop/LEO/RSDS.nosync/100307_FIX/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR_hp2000_clean.nii"

def funconn(path):
    Nifti_img = nib.nifti1.load(path)
