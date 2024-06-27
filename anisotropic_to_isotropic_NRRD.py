import SimpleITK as sitk
from tkinter import filedialog
import nrrd 
import numpy as np

# Read the .nrrd file
path_to_nrrd = filedialog.askopenfilename(title='Select .NRRD file', filetypes=[("NRRD files", "*.nrrd")])
original_data, header = nrrd.read(path_to_nrrd)

img = sitk.ReadImage(path_to_nrrd)
rs2 = sitk.ResampleImageFilter()

new_x = round(img.GetSize()[0] * img.GetSpacing()[0] / np.min(img.GetSpacing()))
new_y = round(img.GetSize()[1] * img.GetSpacing()[1] / np.min(img.GetSpacing()))
new_z = round(img.GetSize()[2] * img.GetSpacing()[2] / np.min(img.GetSpacing()))

rs2.SetOutputSpacing([np.min(img.GetSpacing()), np.min(img.GetSpacing()), np.min(img.GetSpacing())])
rs2.SetSize([new_x, new_y, new_z])
rs2.SetOutputDirection(img.GetDirection())
rs2.SetOutputOrigin(img.GetOrigin())
rs2.SetTransform(sitk.Transform())
rs2.SetInterpolator(sitk.sitkLinear)

img3 = rs2.Execute(img)

resampled_path = filedialog.asksaveasfilename(title='Save resampled .NRRD file', defaultextension=".nrrd", filetypes=[("NRRD files", "*.nrrd")])
sitk.WriteImage(img3, resampled_path)
