
import os
import trimesh
import numpy as np
import pyvista as pv



def load_mesh(path_to_meshes, shape_index):

    # compute path
    mesh_path = os.path.join(path_to_meshes, f'mesh_{shape_index}.stl')

    # load mesh (Trimesh mesh from '.stl' file)
    mesh = trimesh.load_mesh(mesh_path)

    return mesh

def compute_bounding_box(mesh, extrapol_ratio=1):

    mins = np.min(mesh.vertices, axis=0).reshape(1,-1)
    maxs = np.max(mesh.vertices, axis=0).reshape(1,-1)
    new_mins = mins - ( extrapol_ratio-1 ) / 2 * (maxs-mins)
    new_maxs = maxs + ( extrapol_ratio-1 ) / 2 * (maxs-mins)
    bbox = np.concatenate((new_mins, new_maxs), axis=0)

    return bbox

def trimesh_to_pyvista(mesh):
    
    vertices = mesh.vertices
    faces = np.hstack((3*np.ones((mesh.faces.shape[0],1)), mesh.faces)).flatten().astype(int)
    
    pv_mesh = pv.PolyData(vertices, faces)
    
    return pv_mesh


class Udf3d_ShapeNet():

    """
    Generic class representing an Unsigned Distance Function from ShapeNet dataset.
    Must be intialized with a trimesh mesh.
    """

    def __init__(
        self,
        mesh,
        extrapol_ratio=1.25
    ):
        
        self.mesh = mesh
        self.extrapol_ratio = extrapol_ratio
        
        # Compute bounding box
        self.bbox = compute_bounding_box(mesh, extrapol_ratio=extrapol_ratio)
        
        # Convert mesh to PyVista PolyData
        self.pv_mesh = trimesh_to_pyvista(self.mesh)

    def __call__(
        self,
        query_points
    ):

        assert query_points.shape[1] == 3
        
        n = query_points.shape[0]

        # Compute distances using PyVista
        _, closest_points_on_mesh = self.pv_mesh.find_closest_cell(query_points, return_closest_point=True)
        distances = np.linalg.norm(query_points-closest_points_on_mesh, axis=1)

        return distances.flatten()