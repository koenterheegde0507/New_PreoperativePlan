import numpy as np


listWithPoints = []
direction_vectors = []
inverted_direction_vectors = []
startPoints = []
arrayWithArteries = []

end_point_list = []
final_points = []
labels_normal_points = []
final_direction_vectors = []

def increaseLength_direction_vector(contour_mesh):
    mesh_boundaries = list(contour_mesh.bounds)
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


