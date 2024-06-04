import nibabel as nib
from nilearn import datasets
import numpy as np
import nilearn.maskers as maskers
from nilearn import plotting
from nilearn.connectome import ConnectivityMeasure

# Function to load the nifti image
def load_nifti_image(path):
    try:
        Nifti_image = nib.load(path)
        return Nifti_image
    except Exception as error:
        print("Problem while loading Nifti =>" + str(error) + "\n")
        return False

# Function to load the MSDL atlas file
def load_msdl_atlas_file(directory_path):
    try:
        msdl_atlas = datasets.fetch_atlas_msdl(data_dir=directory_path)
        n_regions = len(msdl_atlas.region_coords)
        print(f'The Atlas has {n_regions} ROIs, part of the following networks:\n{np.unique(msdl_atlas.networks)}.')
        return msdl_atlas
    except Exception as error:
        print("Problem while loading MSDL Atlas =>" + str(error) + "\n")
        return False 

# Function to create a time series from the nifti image and atlas
def create_time_series(nifti_image_path, atlas_directory_path):
    Nifti_image = load_nifti_image(nifti_image_path)
    msdl_atlas = load_msdl_atlas_file(atlas_directory_path)
        
    masker = maskers.NiftiMapsMasker(msdl_atlas.maps, resampling_target="data", detrend=True).fit()
    roi_time_series = masker.transform(Nifti_image)
    print("The shape of the Time Series: ", roi_time_series.shape,"\n")
    return roi_time_series

# Function to calculate the correlation matrix from the time series
def calculate_correlation(nifti_image_path, atlas_directory_path):
    Nifti_image = load_nifti_image(nifti_image_path)
    msdl_atlas = load_msdl_atlas_file(atlas_directory_path)
    
    masker = maskers.NiftiMapsMasker(msdl_atlas.maps, resampling_target="data", detrend=True).fit()
    roi_time_series = masker.transform(Nifti_image)
    
    correlation_matrix = ConnectivityMeasure(kind='correlation').fit_transform([roi_time_series])[0]
    np.fill_diagonal(correlation_matrix, 0)
    
    plotting.plot_matrix(correlation_matrix, labels=msdl_atlas.labels, vmax=1.0, vmin=0, colorbar=True, title='Correlation matrix of the MSDL atlas')
    
    return correlation_matrix

# Fuction to extract Datapoints from the correlation matrix.
def extract_datapoints(corr_matrix):
    # Get the indices of the upper triangle of the matrix
    indices = np.triu_indices(corr_matrix.shape[0], k=1)
    # Extract the values of the top diagonal without zeros and Flatten the array
    Datapoints = corr_matrix[indices].flatten()
    return Datapoints

# Function to extract features from the nifti image
def extract_features(nifti_image_path, atlas_directory_path):
    correlation_matrix = calculate_correlation(nifti_image_path, atlas_directory_path)
    return extract_datapoints(correlation_matrix)