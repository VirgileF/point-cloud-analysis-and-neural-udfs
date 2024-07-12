
import numpy as np

import trimesh
from trimesh.sample import sample_surface

import os

from src.point_cloud_analysis import compute_indicator_on_surface
from src.utils.shape_encoding import Udf3d_ShapeNet



def sample_uniform_3d_ball(n_samples, radius):

    # sample from 5d Gaussian
    points_5d = np.random.normal(size=(n_samples, 5))

    # normalize to project on the 5d hypersphere
    norm = np.linalg.norm(points_5d, axis=1)
    points_5d_normalized = points_5d / norm[:, np.newaxis]

    # extract 3 first coordinates
    samples = points_5d_normalized[:, :3]

    # normalize to ball of radius r
    samples *= radius

    return samples

def sample_uniform_box(n_samples, bounding_box):
    
    # Extract the minimum and maximum values from the bounding box
    (xmin, ymin, zmin), (xmax, ymax, zmax) = bounding_box

    # Sample points uniformly within the bounding box
    x = np.random.uniform(xmin, xmax, n_samples)
    y = np.random.uniform(ymin, ymax, n_samples)
    z = np.random.uniform(zmin, zmax, n_samples)

    # Combine the coordinates into a single array
    samples = np.vstack((x, y, z)).T

    return samples

def sample_training_points_from_surface(surface_points, is_warm, n_points, oversampling_strength):
    
    p_W = np.mean(is_warm)
    new_p_W = (1-oversampling_strength)*p_W + oversampling_strength
    n_W = int(new_p_W*n_points)
    
    warm_points = surface_points[is_warm==1]
    random_indices = np.random.choice(warm_points.shape[0], size=n_W, replace=True)
    new_warm_points = warm_points[random_indices]
    
    cool_points = surface_points[is_warm==0]
    random_indices = np.random.choice(cool_points.shape[0], size=n_points-n_W, replace=True)
    new_cool_points = cool_points[random_indices]
    
    surface_training_points = np.vstack((new_warm_points, new_cool_points))
    np.random.shuffle(surface_training_points)

    return surface_training_points

def sample_training_points(mesh,
                           n_training_points,
                           proportion_training_points_from_surface,
                           oversampling_strength,
                           n_surface_points_for_indicator,
                           indicator,
                           k_neighbors,
                           decision_threshold
):
    
    n_surface = int(proportion_training_points_from_surface * n_training_points)
    n_ambient_space = n_training_points - n_surface

    udf = Udf3d_ShapeNet(mesh)
    # ambient_space_training_points = sample_uniform_box(n_ambient_space, udf.bbox)
    ambient_space_training_points = sample_uniform_3d_ball(n_ambient_space, 0.55)
    
    surface_points = np.array(sample_surface(mesh, n_surface_points_for_indicator)[0])
    
    indicator_values = compute_indicator_on_surface(surface_points, indicator, k_neighbors)
    is_warm = indicator_values > decision_threshold
    
    surface_training_points = sample_training_points_from_surface(surface_points, is_warm, n_surface//2, oversampling_strength)
    
    perturb_points_1 = surface_training_points + np.random.normal(0, np.sqrt(.0025), size=(n_surface//2,3))
    perturb_points_2 = surface_training_points + np.random.normal(0, np.sqrt(.00025), size=(n_surface//2,3))
    
    surface_training_points = np.vstack([perturb_points_1, perturb_points_2])
    
    X = np.vstack((ambient_space_training_points, surface_training_points))
    
    y = udf(X).reshape(-1,1)
    
    return X, y

