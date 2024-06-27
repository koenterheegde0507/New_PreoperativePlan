import numpy as np
import pyvista as pv
import math
import json
import os
import xml.etree.ElementTree as ET
import tkinter as tk 
from tkinter import filedialog

import matplotlib.pyplot as plt
import pyvista as pv
import pymesh

root = tk.Tk()
root.withdraw()
root.title('Pick patient data')
file_path = filedialog.askdirectory()

files = os.listdir(file_path)
filename_vtk = [file_name for file_name in files if file_name.endswith('.vtk')][0]

#Read body contour 
body_contour = pv.read(f'{file_path}\{filename_vtk}')
contour_mesh = pv.PolyData(body_contour)

#Load aorta mesh
aorta_contour = pv.read(f'{file_path}/aorta_pig_model.stl')
aorta_mesh = pv.PolyData(aorta_contour)

folder_path = f"{file_path}/Network curve"
file_list = os.listdir(folder_path)

listWithPoints = []

direction_vectors = []
inverted_direction_vectors = []
startPoints = []

def take_screenshot():
    new_plotter.screenshot('PreoperativePlan.png')

def callback(point):
    listWithPoints.append(point)
    
    big_plotter.add_points(point,color='red')
    if (len(listWithPoints) ==2):
        startPoints.append(listWithPoints[0])
        direction_vector = listWithPoints[1]-listWithPoints[0]
        direction_vector /= np.linalg.norm(direction_vector)
        
        direction_vectors.append(direction_vector)
        inverted_direction_vectors.append(direction_vector*-1)

        line = pv.Line(listWithPoints[0],listWithPoints[1])
        big_plotter.add_mesh(line,color='red')
        listWithPoints.clear()
    

centerline_coordinates = []
for file_name in file_list:
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, 'r') as file:
        data = json.load(file)
        for point in data["markups"][0]["controlPoints"]:
            coordinates = point['position']
            centerline_coordinates.append(coordinates)

big_plotter = pv.Plotter(shape=(1,2))
big_plotter.subplot(0,1)
big_plotter.add_mesh(aorta_mesh)
big_plotter.subplot(0,0)

for coordinate in centerline_coordinates:
    big_plotter.add_points(np.array(coordinate),color='black')

big_plotter.enable_point_picking(callback=callback, left_clicking=True,tolerance=0.001)
big_plotter.add_axes()
big_plotter.set_background('white')
big_plotter.show()



mesh_boundaries = list(contour_mesh.bounds)
end_point_list = []

#def increaseLength_direction_vector():
#    for index, directions in enumerate(direction_vectors):
#            i = 0
#            while True:
#                end_point = startPoints[index] + i*directions

#                points, ind = contour_mesh.ray_trace(startPoints[index],end_point)
                
#                if len(points) != 0 or end_point[0]<mesh_boundaries[0] or end_point[0]>mesh_boundaries[1] or end_point[1]<mesh_boundaries[2] or end_point[1]>mesh_boundaries[3] or end_point[2]>mesh_boundaries[5] or end_point[2]<mesh_boundaries[4]:
#                    print('Reached boundary without finding intersection point')
                    
#                    end_point_list.append(end_point)
#                    print(f'added point {end_point}')
#                    break  
#                else:
#                    i += 1 
#            end_point_list.append(points)
            
def increaseLength_direction_vector():
    for index,directions in enumerate(direction_vectors):
        i=0
        while True:
            end_point = startPoints[index]+i*directions

            points,ind = contour_mesh.ray_trace(startPoints[index],end_point)
        
            if (len(points) !=0):
                end_point_list.append(points)
                break
            elif(end_point[0]<mesh_boundaries[0] or end_point[0]>mesh_boundaries[1] or end_point[1]<mesh_boundaries[2] or end_point[1]>mesh_boundaries[3] or end_point[2]>mesh_boundaries[5] or end_point[2]<mesh_boundaries[4]):
                end_point_list.append(end_point.reshape(1, 3).astype(np.float32))
                break
            else:
                i+=1


inverted_end_points_list = []
#def increaseLength_inverted_direction_vector():
#    for index, inverted_directions in enumerate(direction_vectors):
#            i = 0
#           while True:
#                inverted_end_point = startPoints[index] - i*inverted_directions
#                inverted_points, _ = contour_mesh.ray_trace(inverted_end_point, startPoints[index])
#                if len(inverted_points) != 0 or inverted_end_point[2]>mesh_boundaries[5] or inverted_end_point[2]<mesh_boundaries[4]:
#                    print('Reached boundary without finding intersection point')
#                    break  
#                else:
#                    i += 1 
#            inverted_end_points_list.append(inverted_points)

def increaseLength_inverted_direction_vector():
    for index,inverted_directions in enumerate(direction_vectors):
        i=0
        while True:
            inverted_end_point = startPoints[index] - i*inverted_directions
            inverted_points, _ = contour_mesh.ray_trace(inverted_end_point, startPoints[index])
            if(len(inverted_points)!=0):
                inverted_end_points_list.append(inverted_points)
                break
            elif(inverted_end_point[0]<mesh_boundaries[0] or inverted_end_point[0]>mesh_boundaries[1] or inverted_end_point[1]<mesh_boundaries[2] or inverted_end_point[1]>mesh_boundaries[3] or inverted_end_point[2]>mesh_boundaries[5] or inverted_end_point[2]<mesh_boundaries[4]):
                inverted_end_points_list.append(inverted_end_point.reshape(1, 3).astype(np.float32))
                break
            else:
                i+=1

increaseLength_direction_vector()
increaseLength_inverted_direction_vector()

new_plotter = pv.Plotter()
new_plotter.add_mesh(contour_mesh,opacity=0.2)

for coordinate in centerline_coordinates:
    new_plotter.add_points(np.array(coordinate),color='black')

final_points = []
final_inverted_points = []
labels_normal_points = []


for index,point in enumerate(end_point_list):
    if (len(point) !=0):
        labels_normal_points.append(index)
        new_plotter.add_points(point,color='green',point_size=10)

        
        new_plotter.add_point_labels(point, [str(index)], point_color='#03fc45',shape=None,font_size=20)
        final_points.append((index,point))
    else:
        print(f"Can't find intersection for branch: {index}")

for index,inverted_point in enumerate(inverted_end_points_list):
    if(len(inverted_point) !=0):
        #new_plotter.add_points(inverted_point,color='yellow')

        #labels = [f"{index}: inverted" for _ in range(len(inverted_point))]
        #new_plotter.add_point_labels(inverted_point, labels, point_color='yellow', shape=None)
        final_inverted_points.append((index,inverted_point))
    else:
        print(f"Can't find inverted intersection for branch: {index}")


final_direction_vectors = []
final_inverted_direction_vectors = []

count = 0
for direction in direction_vectors:
    for point in final_points:
        if point[0] == count:
            final_direction_vectors.append((count,direction))
    count+=1


count_inverted = 0
for inverted_direction in inverted_direction_vectors:
    for point in final_inverted_points:
        if point[0] == count_inverted:
            final_inverted_direction_vectors.append((count_inverted,inverted_direction))
    count_inverted+=1

new_plotter.set_background('white')
new_plotter.add_axes()
new_plotter.add_text("Press 'S' to take a screenshot", position='upper_right',font_size=10, color='black')
new_plotter.add_key_event('s', take_screenshot)
new_plotter.show()


decimated_mesh = contour_mesh.decimate(target_reduction=0.9)
patientContour = pv.save_meshio('PatientContour.stl', decimated_mesh)

#patientContourLowQuality, ___ = pymesh.collapse_short_edges(patientContour,1e-6)
## Write XML File 
root = ET.Element("PositionsAndDirections")

for index,point in final_points:
    point_element = ET.SubElement(root,"Point")
    point_element.set("id",str(index))
    coords_element = ET.SubElement(point_element,"Position")
    if (point.size>1):
        coords_element.text = " ".join(str(coord) for coord in point[0])
    else:
        coords_element.text = " ".join(str(coord) for coord in point) 

for index,inverted_point in final_inverted_points:
    if (root.find(f"./Point[@id='{index}']") == None):
        point_element = ET.SubElement(root,"Point")
        point_element.set("id",str(index))
    
    point_element = root.find(f"./Point[@id='{index}']")
    inverted_point_element = ET.SubElement(point_element,"InvertedPosition")
    inverted_point_element.text = " ".join(str(inverted_coord) for inverted_coord in inverted_point[0])

for index,directions in final_direction_vectors:
    point_element = root.find(f"./Point[@id='{index}']")
    direction_element = ET.SubElement(point_element,"Direction")
    direction_element.text = " ".join(str(direction) for direction in directions)

for index,inverted_directions in final_inverted_direction_vectors:
    point_element = root.find(f"./Point[@id='{index}']")
    inverted_direction_element = ET.SubElement(point_element,"InvertedDirection")
    inverted_direction_element.text = " ".join(str(inverted_direction) for inverted_direction in inverted_directions)

tree = ET.ElementTree(root)
tree.write("PositionsAndDirections.xml")
