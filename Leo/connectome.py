import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nilearn import image
from nilearn import plotting
from nilearn import datasets
import nilearn.maskers as maskers
from nilearn.connectome import ConnectivityMeasure
import nibabel as nib

path = "/Users/leoschild/Desktop/LEO/RSDS.nosync/100307_FIX/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR_hp2000_clean.nii"


def connectome(path=""):
    Nifti_img = nib.nifti1.load(path)
    msdl_atlas = datasets.fetch_atlas_msdl(data_dir= None)
    masker = maskers.NiftiMapsMasker(msdl_atlas.maps, resampling_target="data").fit()
    roi_time_series = masker.transform(Nifti_img)
    correlation_measure = ConnectivityMeasure(kind='correlation')
    correlation_matrix = correlation_measure.fit_transform([roi_time_series])[0]
    np.fill_diagonal(correlation_matrix, 0)
    plotting.plot_matrix(correlation_matrix, labels=msdl_atlas.labels,vmax=0.8, vmin=-0.8, colorbar=True)



