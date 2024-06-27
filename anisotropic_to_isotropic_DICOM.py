from tkinter import filedialog
from os import listdir
import pydicom
import SimpleITK as sitk
import numpy as np
import os

def read_dicom_series():
    # Read all DICOM files in the directory and sort by instance number
    path_to_dicom = filedialog.askdirectory(title='Select folder with DICOM data')

    files_path = [f'{path_to_dicom}/{file}' for file in listdir(path_to_dicom)]
    dicom_volume = []
    files_path.sort(key=lambda x: pydicom.dcmread(x).InstanceNumber)
    for file in files_path:
        # Read the first file to get metadata
        ref_ds = pydicom.dcmread(file)

        #slice_thickness = ref_ds.SliceThickness
        #pixel_spacing = ref_ds.PixelSpacing

        # Read the pixel data from all DICOM files and stack them
        pixel_array = ref_ds.pixel_array
        dicom_volume.append(pixel_array)

        if len(dicom_volume) == 1:
            slice_thickness = ref_ds.SliceThickness
            pixel_spacing = ref_ds.PixelSpacing
            series_description = ref_ds.SeriesDescription

    return dicom_volume, slice_thickness, pixel_spacing, series_description

def dicom_to_sitk(dicom_volume, slice_thickness, pixel_spacing):
    image = sitk.GetImageFromArray(dicom_volume)
    spacing = list(pixel_spacing)
    spacing.append(slice_thickness)
    image.SetSpacing(spacing)
    return image

def resample_image(image):
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_spacing = [np.min(original_spacing)] * 3

    new_size = [
        int(np.round(original_size[0] * (original_spacing[0] /  np.min(image.GetSpacing())))),
        int(np.round(original_size[1] * (original_spacing[1] /  np.min(image.GetSpacing())))),
        int(np.round(original_size[2] * (original_spacing[2] / np.min(image.GetSpacing()))))
    ]
    
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetOutputDirection(image.GetDirection())
    resample.SetDefaultPixelValue(image.GetPixelIDValue())

    return resample.Execute(image)

#path_to_dicom = filedialog.askdirectory(title='Select folder with DICOM data')
dicom_volume, slice_thickness, pixel_spacing, series_description = read_dicom_series()
sitk_image = dicom_to_sitk(dicom_volume, slice_thickness, pixel_spacing)
resampled_image = resample_image(sitk_image)
#files_path = [f'{path_to_dicom}/{file}' for file in listdir(path_to_dicom)]

new_spacing = resampled_image.GetSpacing()

resampled_slices = [resampled_image[:, :, i] for i in range(resampled_image.GetDepth())]

# Write the sorted slices to the output directory
output_directory = 'Resampled_DICOM'
writer = sitk.ImageFileWriter()

for i, image_slice in enumerate(resampled_slices):
    metadata_value = [
        ("0028|0030", f'{new_spacing[0]},{new_spacing[1]}'),
        ("0018|0050", str(new_spacing[2])),
        ("0020|0013", str(i+1)),
        ("0008|103E", series_description)
    ]

    for tag, value in metadata_value:
        image_slice.SetMetaData(tag, value)

    writer.SetFileName(os.path.join(output_directory, f'resampled_{i}.dcm'))
    writer.Execute(image_slice)
    