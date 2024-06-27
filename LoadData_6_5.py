import pydicom
import tkinter as tk
from tkinter import filedialog
from screeninfo import get_monitors
import os
from os import listdir
import trimesh as tm
import matplotlib
import matplotlib.pylab as plt
from matplotlib.widgets import Slider, Button, RangeSlider, Cursor
from matplotlib import cm
from matplotlib.backend_tools import Cursors
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy import ndimage as ndi
import skimage.morphology
from skimage.morphology import closing
from scipy import ndimage
from skimage.morphology import skeletonize 
import pymeshfix
import sys
import atexit

#from NRRD_implementation import showNRRD
from NRRD_implementation import executeNRRD
import keyboard
import skimage
import pyvista as pv
import numpy as np
#from mayavi import mlab
import vtk
import SimpleITK as sitk
import nrrd
import vtkmodules.numpy_interface.dataset_adapter as dsa

######### READ AND DISPLAY DICOM DATA #############
######### Class copied from https://stackoverflow.com/questions/48185544/read-and-open-dicom-images-using-python ##########
class IndexTracker(object):
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('Scroll to Navigate through the DICOM Image Slices')
        self.list_with_highlighted_points = []
        self.highlighted_point = None
        self.clicked_points_array = []
        self.clicked = False
        self.clicked_point = None
        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = self.slices//2
        self.cmap = plt.cm.gray
        self.im = ax.imshow(self.X[:,:,self.ind],cmap='gray')
        self.update()

    def onscroll(self, event):
        global thresholding_state

        if region_growing_state == True:
            fig.canvas.set_cursor(Cursors.SELECT_REGION)

        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
            vertical_slider.set_val(self.ind)
        else:
            self.ind = (self.ind - 1) % self.slices
            vertical_slider.set_val(self.ind)

        if horizontal_slider is not None:
            highlightPixels(horizontal_slider.val)
 #           ax.set_ylabel('Slice Number: %s' % self.ind)
        else:
            self.update()
        #self.update()
        

    def onmove(self, event):
        if self.clicked == True:
            self.clicked_point = (event.xdata,event.ydata)
    
    def onclick(self,event):
        
        if event.button == 1:
            self.clicked = True
            try:
                self.clicked_point = (int(event.xdata),int(event.ydata))
            except TypeError:
                print('This is not a valid point to be clicked')

            if event.inaxes == vertical_slider.ax:
                None
            else:
                try:
                    if self.clicked_point[0] != None and self.clicked_point[1] != None and region_growing_state == True:
                        if (event.inaxes == region_growing_button.ax or event.inaxes == threshold_button.ax or event.inaxes == OK_button_axis or event.inaxes == apply_thresholding_button.ax or event.inaxes == horizontal_slider.ax):
                            None
                        else:
                            self.highlighted_point = ax.plot(event.xdata, event.ydata, 'ro',markersize=1)
                            self.list_with_highlighted_points.append(self.highlighted_point)
                            fig.canvas.draw()
                            fig.canvas.set_cursor(Cursors.SELECT_REGION)
                            self.clicked_points_array.append((self.clicked_point,self.ind))
                except TypeError:
                    print('This is not a valid point to be clicked')

        elif event.button == 3:
            self.clicked = False

    def enter_axis(self,event):
        if region_growing_state == True:
            if (event.inaxes != region_growing_button.ax and event.inaxes != threshold_button.ax and event.inaxes != vertical_slider.ax and event.inaxes != horizontal_slider.ax and event.inaxes != OK_button_axis and event.inaxes != apply_thresholding_button.ax):
                fig.canvas.set_cursor(Cursors.SELECT_REGION)
        
    def leave_axis(self,event):
        fig.canvas.set_cursor(Cursors.POINTER)
        
    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        ax.set_ylabel('Slice Number: %s' % self.ind)
        self.im.axes.figure.canvas.draw()

#### START SCREEN #####

files_path = None
sorted_files_path = None
DICOM_mode = False
NRRD_mode = False
screen_info = get_monitors()

def on_window_close():
    print('Window is being closed.')

def load_NRRD(event):
    global files_path 
    global NRRD_mode
    root = tk.Tk()
    root.geometry(f'{screen_info[0].width}x{screen_info[0].height}+{int(screen_info[0].width/2)-int(screen_info[0].width/4)}+{int(screen_info[0].height/4)}')
    root.withdraw()
    executeNRRD()
    NRRD_mode = True
    plt.close(start_screen)
    

def load_DICOM(event):
    global DICOM_mode
    global files_path
    global sorted_files_path
    root = tk.Tk()
    root.geometry(f'{screen_info[0].width}x{screen_info[0].height}+{int(screen_info[0].width/2)-int(screen_info[0].width/4)}+{int(screen_info[0].height/4)}')
    root.withdraw()

    path_to_dicom = filedialog.askdirectory(title='Select folder with DICOM data')
    if path_to_dicom:
        print('Loading DICOM data')
        files_path = [f'{path_to_dicom}/{file}' for file in listdir(path_to_dicom)]

        files_with_instance = []
        for file in files_path:
            ds = pydicom.dcmread(file)
            instance_number = int(ds.InstanceNumber)  # Convert to int for proper sorting
            files_with_instance.append((file, instance_number))
        
        # Sort the files by instance number
        files_with_instance.sort(key=lambda x: x[1])
        
        # Extract the sorted file paths
        sorted_files_path = [f[0] for f in files_with_instance]
    else: 
        print('No DICOM folder selected')
        sys.exit()

    plt.close(start_screen)

    DICOM_mode = True
    return sorted_files_path

def convert_to_float(input_list):
    joined_string = ''.join(input_list)
    split_string = joined_string.split(',')
    output_list = [float(num) for num in split_string]

    return output_list


start_screen = plt.figure()
start_screen.suptitle('Pick the data type to load in')
nrrd_load_axes = start_screen.add_axes([0.3,0.4,0.1,0.05])
nrrd_load_button = Button(nrrd_load_axes,'NRRD')
nrrd_load_button.on_clicked(load_NRRD)

DICOM_load_axes = start_screen.add_axes([0.7,0.4,0.1,0.05])
DICOM_load_button = Button(DICOM_load_axes,'DICOM')
DICOM_load_button.on_clicked(load_DICOM)

plt.show()

if NRRD_mode == True:
    sys.exit()


fig, ax = plt.subplots(1,1)
anisotropic_count = 0
plots = []
for file in sorted_files_path:
    spacings = []
    
    ds = pydicom.dcmread(file)
    if isinstance(ds.PixelSpacing, str):
        ds.PixelSpacing = convert_to_float(ds.PixelSpacing)
        ds.SliceThickness = float(ds.SliceThickness)
    
    pixel_spacings = ds.PixelSpacing
    for pixel_spacing in pixel_spacings:
        spacings.append(pixel_spacing)
    spacings.append(ds.SliceThickness)
    if (np.all(ds.PixelSpacing != np.min(spacings))):
        anisotropic_count +=1
    
    pix = ds.pixel_array
    
    #pix = pix*1+(-1024)
    plots.append(pix)

if (anisotropic_count == 0):
    print('The dataset is isotropic')
else:
    warning_box = tk.messagebox.askokcancel(title='Dataset anisotropic',message='This dataset is anisotropic! Resample the data first',icon='error')
    if warning_box:
        print("Visualizing anisotropic CTA data")
    else:
        sys.exit()

y = np.dstack(plots)

tracker = IndexTracker(ax, y)

fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
fig.canvas.mpl_connect('motion_notify_event', tracker.onmove)
fig.canvas.mpl_connect('button_press_event', tracker.onclick)
fig.canvas.mpl_connect('axes_enter_event', tracker.enter_axis)
fig.canvas.mpl_connect('axes_leave_event', tracker.leave_axis)

ax.set_title(ds.get('SeriesDescription', "N/A"))
vertical_axis_left = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
vertical_slider = Slider(vertical_axis_left, "Slice", 0, y.shape[2]-1,orientation='vertical',valinit=tracker.ind)

horizontal_slider = None
region_growing_state = False
thresholding_state = False

def update_slice(val):
    if len(tracker.list_with_highlighted_points) !=0:
        for point in tracker.list_with_highlighted_points:
            for item in point:
                item.set_alpha(0)

    tracker.ind = int(vertical_slider.val)
    if horizontal_slider is not None:
        highlightPixels(horizontal_slider.val)
        ax.set_ylabel('Slice Number: %s' % tracker.ind)
    else:
        tracker.update()



def threshold_callback(event):
    global horizontal_slider
    global thresholding_state
    
    
    if thresholding_state == True:
        thresholding_state = False
    else:
        thresholding_state = True


    if thresholding_state == True:
        if horizontal_slider is None:
            horizontal_axis = fig.add_axes([0.20, 0.05, 0.60, 0.03])
            horizontal_slider = RangeSlider(horizontal_axis, "Threshold",y.min(), y.max())

            highlightPixels(horizontal_slider.val)
            horizontal_slider.on_changed(highlightPixels)
            apply_thresholding_axis.set_visible(True)
        else:
            apply_thresholding_axis.set_visible(False)
            horizontal_slider.ax.remove()
            horizontal_slider = None
            tracker.update()

    
    if thresholding_state == True:
        thresholding_state = False
    else:
        thresholding_state = True
    
    """
    if horizontal_slider != None:
        apply_thresholding_axis.set_visible(True)
    else:
        apply_thresholding_axis.remove()
    """
    fig.canvas.draw()

    plt.show()


def highlightPixels(threshold):
    bottom_value = threshold[0]
    top_value = threshold[1]

    slice_2d = tracker.X[:,:,tracker.ind].copy()
    normalized_slice = ((slice_2d-slice_2d.min())/(slice_2d.max()-slice_2d.min()))*255
    normalized_slice = normalized_slice.astype(int)
    mask = (slice_2d >=bottom_value) & (slice_2d <=top_value)
    rgb_slice = np.stack([normalized_slice]*3,axis=-1)

    rgb_slice[mask] = [0,0,255]

    tracker.im.set_data(rgb_slice)
    fig.canvas.draw_idle()

    return (np.where(mask))

def generateMesh(extractedVolume,useFilter):
    def testFunction(selected_grid):
    
        if not selected_grid.n_cells:
            return
        ghost_cells = np.zeros(grid.n_cells, np.uint8)
        ghost_cells[selected_grid['orig_extract_id']] = 1
        grid.cell_data[vtk.vtkDataSetAttributes.GhostArrayName()] = ghost_cells
        grid.RemoveGhostCells()

    ###### FOR GETTING CONTOUR DON"T USE FILTER
    ###### FOR GETTING AORTA USE FILTER
    if useFilter == True:
        struct = ndimage.generate_binary_structure(3, 3)
        closed = skimage.morphology.binary_closing(image=extractedVolume,footprint=struct)
        opened = skimage.morphology.binary_opening(closed,struct)
        median_filter = ndi.median_filter(opened,(5,5,5))
        #median_filter2 = ndi.median_filter(median_filter, (5,5,5))

    
        mesh_verts,mesh_faces,mesh_normals,mesh_values = skimage.measure.marching_cubes(median_filter, level=0,step_size=1)

        faces_with_counts = np.hstack([np.full((mesh_faces.shape[0],1),3),mesh_faces])

        filtered_mesh = pv.PolyData(mesh_verts,faces_with_counts)

        largest = filtered_mesh.connectivity('largest')
    else:
        struct = ndimage.generate_binary_structure(3,3)
        closed = skimage.morphology.binary_closing(image=extractedVolume,footprint=struct) #still use binary closing to not get small holes in the contour
        mesh_verts,mesh_faces,mesh_normals,mesh_values = skimage.measure.marching_cubes(closed, level=0,step_size=1)

        faces_with_counts = np.hstack([np.full((mesh_faces.shape[0],1),3),mesh_faces])

        unfiltered_mesh = pv.PolyData(mesh_verts,faces_with_counts)

        largest = unfiltered_mesh.connectivity('largest')

    grid = largest.cast_to_unstructured_grid()

    plotter = pv.Plotter()
    plotter.add_mesh(grid,color='blue')
    plotter.enable_cell_picking(callback=testFunction,show=False)
    plotter.show()

    saved_file = tk.simpledialog.askstring(title='Save file',prompt='Enter file name (without .STL):')
    final_mesh = grid.extract_surface()
    """
    fixer = pymeshfix.MeshFix(final_mesh)
    fixer.repair(joincomp=True, remove_smallest_components=False)
    if saved_file:
        pv.save_meshio(f'{saved_file}.stl',fixer.mesh)
        print('Saved file')
    """
    
    if useFilter == True:
        fixer = pymeshfix.MeshFix(final_mesh)
        fixer.repair(joincomp=True, remove_smallest_components=False)
        if saved_file:
            pv.save_meshio(f'{saved_file}.stl',fixer.mesh)
            print('Saved file')
        else:
            print('File not saved, framework can be closed')
    else:
        if saved_file:
            pv.save_meshio(f'{saved_file}.stl',final_mesh)
            print('Saved file')
        else:
            print('File not saved, framework can be closed')
    
def extractVolume(event):
    
    global horizontal_slider

    if (region_growing_state != True):
        print("Extracting volume")
        volume = tracker.X[:,:,:]
        mask = (volume>=horizontal_slider.val[0]) & (volume<=horizontal_slider.val[1])
        
        filterMessageBox = tk.messagebox.askquestion(title='Apply filter',message='Do you want to apply the filtering steps?')
        if filterMessageBox == 'yes':
            useFilter = True
        else:
            useFilter = False
        
        generateMesh(mask,useFilter)
    else:
        tk.messagebox.showwarning(title='Extract Volume',message="Press the 'Apply Region' button in order to extract the volume")


def toggle_region_growing(event):
    global region_growing_state 
    if region_growing_state == True:
        region_growing_state = False
    else:
        if (horizontal_slider != None):
            region_growing_state = True
            horizontal_slider.set_active(False)
        else: 
            tk.messagebox.showwarning(title='Region Growing method',message="Can't go into region growing mode before setting threshold boundaries")


    OK_button_axis.set_visible(region_growing_state)
    fig.canvas.draw()
    


################### REGION GROWING ###############
def apply_region_growing(event):
    
    print('Applying region growing segmentation...')
    
    imageSITK = sitk.GetImageFromArray(tracker.X[:,:,:])
    test_image = tracker.X[:,:,tracker.ind]
    
    seed_points = [(point[1], point[0][0], point[0][1]) for point in tracker.clicked_points_array]

    testImage3D = sitk.ConnectedThreshold(imageSITK,seed_points,
                                            lower=horizontal_slider.val[0], 
                                            upper=horizontal_slider.val[1],
                                            replaceValue=1, 
                                            connectivity=0)
    
    result_array = sitk.GetArrayFromImage(testImage3D)
    
    filterMessageBox = tk.messagebox.askquestion(title='Apply filter',message='Do you want to apply the filtering steps?')
    if filterMessageBox == 'yes':
        generateMesh(result_array,True)
        #generateMesh(median_filter_result,True)
        #median_filter_result_2 = ndi.median_filter(median_filter_result, (5,5,5))
        #dilated_result = skimage.morphology.binary_dilation(result_array,struct)

    else:
        generateMesh(result_array,False)
        #generateMesh(closed_result,False)
        

    """
    region_growing_verts_initial, region_growing_faces_initial, region_growing_normals_initial, _ = skimage.measure.marching_cubes(result_array, level=0,step_size=1)
    faces_with_counts_region_growing_initial = np.hstack([np.full((region_growing_faces_initial.shape[0],1),3),region_growing_faces_initial])
    mesh_initial = pv.PolyData(region_growing_verts_initial,faces_with_counts_region_growing_initial)

    region_growing_verts_closing, region_growing_faces_closing, region_growing_normals_closing, _ = skimage.measure.marching_cubes(closed_result, level=0,step_size=1)
    faces_with_counts_region_growing_closing = np.hstack([np.full((region_growing_faces_closing.shape[0],1),3),region_growing_faces_closing])
    mesh_closing = pv.PolyData(region_growing_verts_closing,faces_with_counts_region_growing_closing)

    region_growing_verts, region_growing_faces, region_growing_normals, _ = skimage.measure.marching_cubes(opened_result, level=0,step_size=1)
    faces_with_counts_region_growing = np.hstack([np.full((region_growing_faces.shape[0],1),3),region_growing_faces])
    mesh_opening = pv.PolyData(region_growing_verts,faces_with_counts_region_growing)

    region_growing_verts_dilated, region_growing_faces_dilated, region_growing_normals_dilated, _ = skimage.measure.marching_cubes(dilated_result, level=0,step_size=1)
    faces_with_counts_region_growing_dilated = np.hstack([np.full((region_growing_faces_dilated.shape[0],1),3),region_growing_faces_dilated])
    mesh_dilated = pv.PolyData(region_growing_verts_dilated,faces_with_counts_region_growing_dilated)
    """
    
    
    
    



        
vertical_slider.on_changed(update_slice)

threshold_axis = fig.add_axes([0.8,0.8,0.1,0.05])
region_growing_axis = fig.add_axes([0.8,0.6,0.1,0.05])
OK_button_axis = fig.add_axes([0.8,0.15,0.1,0.05])
OK_button_axis.set_visible(False)
apply_thresholding_axis = fig.add_axes([0.8,0.3,0.1,0.05])
apply_thresholding_axis.set_visible(False)
threshold_button = Button(threshold_axis,'Threshold')
region_growing_button = Button(region_growing_axis,'Region Growing')
OK_button = Button(OK_button_axis,'Apply Region Growing')
apply_thresholding_button = Button(apply_thresholding_axis,'Extract Volume')
threshold_button.on_clicked(threshold_callback)
region_growing_button.on_clicked(toggle_region_growing)
OK_button.on_clicked(apply_region_growing)
apply_thresholding_button.on_clicked(extractVolume)

plt.show()



