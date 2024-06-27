import os 
import vtk
import tkinter as tk 
from tkinter import filedialog
from tkinter import *
from tkinter import simpledialog
import pyvista as pv
import numpy as np 
from skimage.morphology import skeletonize 
import trimesh as tm
from screeninfo import get_monitors
from scipy import ndimage as ndi
import xml.etree.ElementTree as ET

from HelperFunctions import listWithPoints,direction_vectors,inverted_direction_vectors,startPoints,arrayWithArteries,end_point_list,final_points,labels_normal_points, final_direction_vectors
from HelperFunctions import increaseLength_direction_vector

class CustomDialog(simpledialog.Dialog):
    def __init__(self, parent, title=None, width=800,height=200):
        self.width = width
        self.height = height
        self.OK_clicked = False
        self.result = None
        super().__init__(parent, title)

        

    def body(self, master):
        self.geometry(f'{self.width}x{self.height}')  # Set the size of the dialog window
        self.title("Selected artery")
        Label(master, text="Enter artery name:").grid(row=0)
        self.entry = Entry(master)
        self.entry.grid(row=0, column=1)
        return self.entry  # Return the entry widget

    def apply(self):
        self.result = self.entry.get()

    def get_user_input(self):
        return self.result
    
    def buttonbox(self):
        box = tk.Frame(self)

        ok_button = tk.Button(box, text="OK", width=10, command=self.ok_button_clicked, default=tk.ACTIVE)
        ok_button.pack(side="left", padx=5)

        cancel_button = tk.Button(box, text="Cancel", width=10, command=self.cancel_button_clicked)
        cancel_button.pack(side="left", padx=5)

        self.bind("<Return>", self.ok_button_clicked)
        self.bind("<Escape>", self.cancel_button_clicked)

        box.pack()

    def ok_button_clicked(self, event=None):
        self.OK_clicked = True
        #self.result = self.entry.get()
        self.ok()

    def cancel_button_clicked(self, event=None):
        self.cancel_clicked = True
        self.cancel()
        


"""
    def buttonbox(self):
        super().buttonbox()
        self.ok_button = self.buttonbox.children["!button"]
        self.cancel_button = self.buttonbox.children["!button2"]
        
        # Add a click listener to the Cancel button
        self.cancel_button.bind("<Button-1>", self.cancel_clicked)

    def cancel_clicked(self, event):
        print("Cancel button clicked")
"""
screen_info = get_monitors()

root = tk.Tk()

root.geometry(f'{screen_info[0].width}x{screen_info[0].height}+{int(screen_info[0].width/2)-int(screen_info[0].width/4)}+{int(screen_info[0].height/4)}')
root.withdraw()

file_path_vascular_structure = filedialog.askopenfilename(title='Pick 3D model of vascular structure')

chosenVascularStructure = tm.load_mesh(f'{file_path_vascular_structure}')



meshPlotter = pv.Plotter(window_size=[screen_info[0].width, screen_info[0].height])
meshPlotter.add_mesh(chosenVascularStructure)
meshPlotter.add_title('Chosen mesh')
meshPlotter.show()

file_path_contour = filedialog.askopenfilename(title='Pick contour of patient')
chosenContour = pv.read(f'{file_path_contour}')
chosenContourMesh = pv.PolyData(chosenContour)

contourPlotter = pv.Plotter(window_size=[screen_info[0].width, screen_info[0].height])
contourPlotter.add_mesh(chosenContourMesh)
contourPlotter.add_title('Chosen contour')
contourPlotter.show()


#### FUNCTIONS #####
f_pressed = False
def on_f_key_pressed():
    global f_pressed
    f_pressed = True
    return f_pressed


def get_user_input():
    centerlinePlotter.disable()
    customDialog = CustomDialog(root)
    user_input = customDialog.get_user_input()

    #cancelButtonIsClicked = customDialog.cancel_button_clicked()
    
    
#    user_input = simpledialog.askstring(title='Selected artery', prompt="Enter artery name:")
    if user_input is not None:
        print('Voeg naam toe aan arrayWithArteries')
        arrayWithArteries.append(user_input)
        #centerlinePlotter.enable()
    return customDialog.cancel_clicked

def take_screenshot():
    planningPlotter.screenshot('PreoperativePlan.png')

red_point_list = []
def pickPoint(point):
    global f_pressed

    if (f_pressed == True): 
        None
    else:
        listWithPoints.append(point)
    
    #centerlinePlotter.add_points(point,color='red')
        red_point_list.append(centerlinePlotter.add_points(point,color='red'))
    
        if (len(listWithPoints) ==2):
            centerlinePlotter.disable()

            customDialog = CustomDialog(root)
            user_input = customDialog.get_user_input()

            if (customDialog.OK_clicked == True):
                startPoints.append(listWithPoints[0])
                direction_vector = listWithPoints[1]-listWithPoints[0]
                direction_vector /= np.linalg.norm(direction_vector)
        
                direction_vectors.append(direction_vector)
                inverted_direction_vectors.append(direction_vector*-1)

                line = pv.Line(listWithPoints[0],listWithPoints[1])
                #centerlinePlotter.add_mesh(line,color='red')
                centerlinePlotter.add_mesh(line,color='red',line_width=5)

                centerlinePlotter.disable()
                arrayWithArteries.append(user_input)
            else:
                for point in red_point_list:
                    centerlinePlotter.remove_actor(point)
            listWithPoints.clear()    
    f_pressed = False
        

#### CENTERLINE EXTRACTION #####
print('Extracting centerline of vascular structure...')
volume = tm.voxel.creation.voxelize(mesh=chosenVascularStructure, pitch=chosenVascularStructure.extents.max() / 800)
array = volume.matrix
filled = ndi.binary_fill_holes(array)

skeleton = skeletonize(filled)

x_skel, y_skel, z_skel = np.where(skeleton)
min_bound = chosenVascularStructure.bounds[0]
max_bound = chosenVascularStructure.bounds[1]

x_scaled = (x_skel / skeleton.shape[0]) * (max_bound[0] - min_bound[0]) + min_bound[0]
y_scaled = (y_skel / skeleton.shape[1]) * (max_bound[1] - min_bound[1]) + min_bound[1]
z_scaled = (z_skel / skeleton.shape[2]) * (max_bound[2] - min_bound[2]) + min_bound[2]

points = np.column_stack((x_scaled, y_scaled, z_scaled))


centerlinePlotter = pv.Plotter(shape=(1,2),window_size=[screen_info[0].width, screen_info[0].height])

centerlinePlotter.subplot(0,0)
centerlinePlotter.add_points(points,color='black')


centerlinePlotter.subplot(0,1)
centerlinePlotter.add_mesh(chosenVascularStructure,pickable=False)

centerlinePlotter.enable_point_picking(callback=pickPoint, left_clicking=True,tolerance=0.001)
centerlinePlotter.add_key_event('f', on_f_key_pressed)
#centerlinePlotter.enable_point_picking(callback=pickPoint, left_clicking=True,tolerance=0.001)

centerlinePlotter.link_views()

#interactor = pv.RenderWindowInteractor(centerlinePlotter)
#interactor.track_mouse_position(get_mouse_position)
centerlinePlotter.show()


"""
def show_left_plot(left_plot):
    left_plot.add_points(points, color='black')
    left_plot.enable_point_picking(callback=pickPoint, left_clicking=True, tolerance=0.001)
    left_plot.show()

def show_right_plot(right_plot):
    right_plot.add_mesh(chosenVascularStructure)
    right_plot.show()

left_plot = pv.Plotter(window_size=[int(screen_info[0].width/2),screen_info[0].height])
right_plot = pv.Plotter(window_size=[int(screen_info[0].width/2),screen_info[0].height])
"""

##### CALCULATE POINTS #####

increaseLength_direction_vector(chosenContourMesh)


##### PLOT POINTS #####
print('Creating preoperative plan...')
planningPlotter = pv.Plotter(window_size=[screen_info[0].width, screen_info[0].height])
planningPlotter.add_mesh(chosenContourMesh,opacity=0.2)
planningPlotter.add_points(points,color='black')
planningPlotter.add_points(np.array((0,0,0)),color='red')

poly = pv.PolyData(np.array(end_point_list).flatten())
poly["My labels"] = [f"{artery}" for artery in arrayWithArteries]
planningPlotter.add_point_labels(poly, "My labels", point_color='#03fc45',shape=None,font_size=30,point_size=20)

"""
for index,point in enumerate(end_point_list):
    if (len(point) !=0):
        labels_normal_points.append(index)
        planningPlotter.add_points(point,color='green',point_size=10)

        
        planningPlotter.add_point_labels(point, [str(index)], point_color='#03fc45',shape=None,font_size=20)
        final_points.append((index,point))
    else:
        print(f"Can't find intersection for branch: {index}")
"""


planningPlotter.set_background('white')
planningPlotter.add_axes()
planningPlotter.add_text("Press 'S' to take a screenshot", position='upper_right',font_size=10, color='black')
planningPlotter.add_key_event('s', take_screenshot)
planningPlotter.show()

"""
count = 0
for direction in direction_vectors:
    for point in final_points:
        if point[0] == count:
            final_direction_vectors.append((count,direction))
    count+=1
"""

combined_position = list(zip(arrayWithArteries, end_point_list))
for string, array in combined_position:
    final_points.append((string, array))

combined_direction = list(zip(arrayWithArteries, direction_vectors))
for string, array in combined_direction:
    final_direction_vectors.append((string,array))


######### WRITE STL FILE ##########
print('Writing .STL file...')
decimatedMesh = chosenContourMesh.decimate(target_reduction=0.9)
pv.save_meshio('PatientContour.stl', decimatedMesh)


######### WRITE XML FILE ##########
print('Writing .XML file...')
root = ET.Element("PositionsAndDirections")

for artery,point in final_points:
    point_element = ET.SubElement(root,"Artery")
    point_element.set("id",artery)
    coords_element = ET.SubElement(point_element,"Position")
    coords_element.text = " ".join(str(coord) for coord in point[0])

for artery,directions in final_direction_vectors:
    point_element = root.find(f"./Artery[@id='{artery}']")
    direction_element = ET.SubElement(point_element,"Direction")
    direction_element.text = " ".join(str(direction) for direction in directions)

tree = ET.ElementTree(root)
tree.write("PositionsAndDirections.xml")
