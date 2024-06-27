
def executeNRRD():
    print('Loading NRRD data')
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button, RangeSlider, Cursor
    import numpy as np
    import skimage.morphology
    from tkinter import filedialog
    from screeninfo import get_monitors
    import nrrd
    import pyvista as pv
    import vtk
    from scipy import ndimage
    from scipy import ndimage as ndi
    import tkinter as tk
    import pymeshfix
    import SimpleITK as sitk
    from matplotlib.backend_tools import Cursors
    import sys

    ######### Class copied from https://stackoverflow.com/questions/48185544/read-and-open-dicom-images-using-python ##########
    class IndexTracker(object):
        def __init__(self, ax, X):
            self.ax = ax
            ax.set_title('Scroll to Navigate through the NRRD Image Slices')
            self.list_with_highlighted_points = []
            self.highlighted_point = None
            self.clicked_points_array = []
            self.clicked = False
            self.clicked_point = None
            self.X = X
            rows, self.slices, cols = X.shape
            self.ind = int(self.slices//2)
            self.cmap = plt.cm.gray
            self.im = ax.imshow(self.X[:,self.ind,:],cmap='gray')
            #self.im = ax.imshow(self.X[:, :, self.ind], cmap='gray', vmin=0, vmax=255)
            self.update()

        def onscroll(self, event):

            if region_growing_state == True:
                figure_2.canvas.set_cursor(Cursors.SELECT_REGION)
            if event.button == 'up':
                self.ind = (self.ind + 1) % self.slices
                vertical_slider.set_val(int(self.ind))
            else:
                self.ind = (self.ind - 1) % self.slices
                vertical_slider.set_val(int(self.ind))

            if horizontal_slider is not None:
                highlightPixels(horizontal_slider.val)
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
                            if (event.inaxes == region_growing_button.ax or event.inaxes == threshold_button.ax or event.inaxes == region_growing_axis or event.inaxes == apply_thresholding_button.ax or event.inaxes == horizontal_slider.ax):
                                None
                            else:
                                self.highlighted_point = axis_2.plot(event.xdata, event.ydata, 'ro',markersize=1)
                                self.list_with_highlighted_points.append(self.highlighted_point)
                                figure_2.canvas.draw()
                                figure_2.canvas.set_cursor(Cursors.SELECT_REGION)
                                self.clicked_points_array.append((self.clicked_point,self.ind))
                    except TypeError:
                        print('This is not a valid point to be clicked')


            elif event.button == 3:
                self.clicked = False

        def enter_axis(self,event):
            
            if region_growing_state == True:
                if (event.inaxes != region_growing_button.ax and event.inaxes != threshold_button.ax and event.inaxes != vertical_slider.ax and event.inaxes != horizontal_slider.ax and event.inaxes != apply_region_growing_axis and event.inaxes != apply_thresholding_button.ax):
                    figure_2.canvas.set_cursor(Cursors.SELECT_REGION)
            
        def leave_axis(self,event):
            figure_2.canvas.set_cursor(Cursors.POINTER)

        def update(self):
            if horizontal_slider == None:
                self.im.set_data(self.X[:, self.ind, :])
            self.ax.set_ylabel('Slice Number: %s' % self.ind)
            self.im.axes.figure.canvas.draw()

    screen_info = get_monitors()
    horizontal_slider = None
    thresholding_state = False
    region_growing_state = False

    def load_NRRD():
        
        path_to_nrrd = filedialog.askopenfilename(title='Select .NRRD file',filetypes=[("NRRD files", "*.nrrd")])
        if path_to_nrrd:
            readdata, header = nrrd.read(path_to_nrrd)
            diagonal_values = np.diagonal(header['space directions'])
            if(np.all(diagonal_values == np.min(diagonal_values))):
                print(f'Dataset is isotropic')
            else:
                warning_box = tk.messagebox.askokcancel(title='Dataset anisotropic',message='This dataset is anisotropic! Resample the data first',icon='error')
                if warning_box:
                    print("Visualizing anisotropic CTA data")
                else:
                    sys.exit()
            return readdata
        else:
            print('No .NRRD file selected')
            sys.exit()

    
    data = load_NRRD()
    figure_2, axis_2 = plt.subplots(1,1) 
    
    y = np.dstack(data)
    tracker = IndexTracker(axis_2, y)

    figure_2.canvas.mpl_connect('scroll_event', tracker.onscroll)
    figure_2.canvas.mpl_connect('motion_notify_event', tracker.onmove)
    figure_2.canvas.mpl_connect('button_press_event', tracker.onclick)
    figure_2.canvas.mpl_connect('axes_enter_event', tracker.enter_axis)
    figure_2.canvas.mpl_connect('axes_leave_event', tracker.leave_axis)

    vertical_axis_left = figure_2.add_axes([0.1, 0.25, 0.0225, 0.63])
    vertical_slider = Slider(vertical_axis_left, "Slice", 0, y.shape[1]-1,orientation='vertical',valinit=tracker.ind,valfmt="%i")

    
    def update_slice(val):
        if len(tracker.list_with_highlighted_points) !=0:
            for point in tracker.list_with_highlighted_points:
                for item in point:
                    item.set_alpha(0)
        tracker.ind = int(vertical_slider.val)
        #tracker.update()
        if horizontal_slider is not None:
            tracker.update()
            highlightPixels(horizontal_slider.val)
            #ax.set_ylabel('Slice Number: %s' % tracker.ind)
        else:
            tracker.update()
        

    def threshold_callback(event):
        
        nonlocal horizontal_slider
        nonlocal thresholding_state

        if thresholding_state == True:
            thresholding_state = False
        else:
            thresholding_state = True

        if thresholding_state == True:
            if horizontal_slider is None:
                horizontal_axis = figure_2.add_axes([0.20, 0.05, 0.60, 0.03])
                horizontal_slider = RangeSlider(horizontal_axis, "Threshold", y.min(), y.max())

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

        #apply_thresholding_axis.set_visible(thresholding_state)

    def highlightPixels(threshold):
        
        bottom_value = threshold[0]
        top_value = threshold[1]

        slice_2d = tracker.X[:,tracker.ind,:].copy()
        normalized_slice = ((slice_2d-slice_2d.min())/(slice_2d.max()-slice_2d.min()))*255
        normalized_slice = normalized_slice.astype(int)
        mask = (slice_2d >=bottom_value) & (slice_2d <=top_value)
        rgb_slice = np.stack([normalized_slice]*3,axis=-1)

        rgb_slice[mask] = [0,0,255]
        tracker.im.set_data(rgb_slice)
        
        figure_2.canvas.draw_idle()
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


        #root.withdraw()
        saved_file = tk.simpledialog.askstring(title='Save file',prompt='Enter file name (without .STL):')
        final_mesh = grid.extract_surface()
        
        if useFilter == True:
            fixer = pymeshfix.MeshFix(final_mesh)
            fixer.repair(joincomp=True, remove_smallest_components=False)
            if saved_file:
                pv.save_meshio(f'{saved_file}.stl',fixer.mesh)
                print('Saved mesh')
            else:
                print('File not saved, framework can be closed')
        else:
            if saved_file:
                pv.save_meshio(f'{saved_file}.stl',final_mesh)
                print('Saved mesh')
            else:
                print('File not saved, framework can be closed')

    
    def extractVolume(event):
        nonlocal horizontal_slider
        
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
        #return mask

    def toggle_region_growing(event):
        nonlocal region_growing_state 
        #if thresholding_state == True:
        if region_growing_state == True:
            region_growing_state = False
        else:
#            if (thresholding_state == True):
            if (horizontal_slider != None):
                region_growing_state = True
                horizontal_slider.set_active(False)
            else: 
                tk.messagebox.showwarning(title='Region Growing method',message="Can't go into region growing mode before setting threshold boundaries")

        apply_region_growing_axis.set_visible(region_growing_state)
        figure_2.canvas.draw()
    
    def apply_region_growing(event):
    
        print('Apply region growing segmentation...')
        
        """
        def editPlotter(mesh_grid):
            if not mesh_grid.n_cells:
                return
            ghost_cells = np.zeros(region_growing_grid.n_cells, np.uint8)
            ghost_cells[mesh_grid['orig_extract_id']] = 1
            region_growing_grid.cell_data[vtk.vtkDataSetAttributes.GhostArrayName()] = ghost_cells
            region_growing_grid.RemoveGhostCells()
        """
        imageSITK = sitk.GetImageFromArray(tracker.X[:,:,:])
        #test_image = tracker.X[:,:,tracker.ind]
        
        #seed_points = [(point[1], point[0][0], point[0][1]) for point in tracker.clicked_points_array]
        seed_points = [(point[0][0], point[1], point[0][1]) for point in tracker.clicked_points_array]
        
        testImage3D = sitk.ConnectedThreshold(imageSITK,seed_points,
                                                lower=horizontal_slider.val[0], 
                                                upper=horizontal_slider.val[1],
                                                replaceValue=1, 
                                                connectivity=0)
        
        result_array = sitk.GetArrayFromImage(testImage3D)
        
        filterMessageBox = tk.messagebox.askquestion(title='Apply filter',message='Do you want to apply the filtering steps?')
        if filterMessageBox == 'yes':
            """
            struct = ndimage.generate_binary_structure(3, 3)
            closed_result = skimage.morphology.binary_closing(result_array,struct)
            opened_result = skimage.morphology.binary_opening(closed_result,struct)
            median_filter_result = ndi.median_filter(opened_result,(5,5,5))
            """
            #generateMesh(median_filter_result,True)
            generateMesh(result_array,True)
            #median_filter_result_2 = ndi.median_filter(median_filter_result, (5,5,5))
            #dilated_result = skimage.morphology.binary_dilation(result_array,struct)

        else:
            """
            struct = ndimage.generate_binary_structure(3, 3)
            closed_result = skimage.morphology.binary_closing(result_array,struct)
            """
            #generateMesh(closed_result,False)
            generateMesh(result_array,False)

        np.save('segmentation_result', result_array)

    vertical_slider.on_changed(update_slice)

    threshold_axis = figure_2.add_axes([0.8,0.8,0.1,0.05])
    region_growing_axis = figure_2.add_axes([0.8,0.6,0.1,0.05])
    apply_thresholding_axis = figure_2.add_axes([0.8,0.3,0.1,0.05])
    apply_thresholding_axis.set_visible(False)
    apply_region_growing_axis = figure_2.add_axes([0.8,0.15,0.1,0.05])
    apply_region_growing_axis.set_visible(False)
    
    threshold_button = Button(threshold_axis,'Threshold')
    region_growing_button = Button(region_growing_axis,'Region Growing')
    apply_thresholding_button = Button(apply_thresholding_axis,'Extract Volume')
    apply_region_growing_button = Button(apply_region_growing_axis,'Apply Region Growing')

    threshold_button.on_clicked(threshold_callback)
    region_growing_button.on_clicked(toggle_region_growing)
    apply_thresholding_button.on_clicked(extractVolume)
    apply_region_growing_button.on_clicked(apply_region_growing)


    threshold_axis.button = threshold_button
    vertical_axis_left.slider = vertical_slider
    region_growing_axis.button = region_growing_button
    apply_thresholding_axis.button = apply_thresholding_button
    apply_region_growing_axis.button = apply_region_growing_button

    plt.show()