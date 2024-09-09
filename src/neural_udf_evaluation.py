import os

from src.utils.system import load_json, load_pickle
from src.utils.shape_encoding import load_mesh, Udf3d_ShapeNet
from src.point_cloud_analysis import compute_indicator_on_surface

import numpy as np
import torch

from copy import deepcopy

import trimesh
from trimesh.sample import sample_surface

from sklearn.neighbors import KDTree
import ot
from scipy.spatial import distance_matrix


def sample_uniform_3d_sphere(n_samples, radius):

    # sample from 5d Gaussian
    gaussian_samples = np.random.normal(size=(n_samples, 3))

    # normalize to project on the 3d sphere
    norm = np.linalg.norm(gaussian_samples, axis=1)
    points_on_sphere = radius * gaussian_samples / norm[:, np.newaxis]

    return points_on_sphere

def load_results_from_single_run(path_to_results, neural_udf_index):
    
    # load experiment params
    experiment_parameters = load_json(os.path.join(path_to_results, 'experiment_parameters.json'))

    # load training parameters
    path_to_neural_udf = os.path.join(path_to_results, f'neural_udfs/neural_udf_{neural_udf_index}')
    training_parameters = load_json(os.path.join(path_to_neural_udf, 'training_parameters.json'))
        
    # load results
    X = load_pickle(os.path.join(path_to_neural_udf, 'X.pickle'))
    neural_udf = load_pickle(os.path.join(path_to_neural_udf, 'neural_udf.pickle'))
        
    # load mesh and convert to udf
    mesh = load_mesh(experiment_parameters['path_to_meshes'], training_parameters['shape_index'])
    udf = Udf3d_ShapeNet(mesh)
    
    return X, udf, neural_udf, training_parameters

def descend_udf(
    neural_udf_torch, 
    points,
    n_steps=20,
    dt=1e-1,
    return_L_list=False
):
    
    if isinstance(points, np.ndarray):
        q = torch.tensor(points).float()
        
    N = q.shape[0]
        
    losses = None
    step_damping = torch.ones([N])
    L_list = []
    for step in range(n_steps):
        # Autograd for Loss
        q.requires_grad = True
        losses_old = losses
        losses = torch.norm(neural_udf_torch(q, with_grad=True), dim=1)**2
        if losses_old is not None:
            mask = 0.0+(losses > losses_old)
            step_damping -= 0.5*step_damping*mask
            q = (mask[:, None]*q_old + (1-mask[:,None])*q).clone().detach().requires_grad_(True)
            losses = torch.norm(neural_udf_torch(q, with_grad=True), dim=1)**2
        L = torch.sum(losses)
        L_list.append(L.item())
        L.backward()
        dL = q.grad
        # Making step with line search
        q_old = q
        q = (q-dt*step_damping[:,None]*dL).clone().detach().requires_grad_(True)
    
    new_points = q.detach().numpy()
    
    if return_L_list:
        return new_points, L_list
    else:
        return new_points

def compute_hausdorff_distance(A1, A2):

    search_index1 = KDTree(A1)
    search_index2 = KDTree(A2)

    distances2, _ = search_index1.query(A2)
    distances1, _ = search_index2.query(A1)

    hausdorff_distance =  max(np.max(distances1), np.max(distances2))

    return hausdorff_distance

def compute_chamfer_distance(A1, A2):

    search_index1 = KDTree(A1)
    search_index2 = KDTree(A2)

    distances2, _ = search_index1.query(A2)
    distances1, _ = search_index2.query(A1)

    N1, N2 = A1.shape[0], A2.shape[0]
    chamfer_dist = 1/N1*np.sum(distances1) + 1/N2*np.sum(distances2)

    return chamfer_dist

def compute_W1_distance(A1, A2):

    M = distance_matrix(A1, A2)
    
    w1 = 1/A1.shape[0]*np.ones(A1.shape[0],)
    w2 = 1/A2.shape[0]*np.ones(A2.shape[0],)
    
    return ot.emd2(w1, w2, M)

def compute_metrics(
    udf, 
    neural_udf, 
    n_surface_points_for_metrics,
    use_true_surface_points_as_initialization=False,
    compute_correlations_with_indicator=True,
    indicator=None,
    k_neighbors=None,
    decision_threshold=None,
):

    true_surface_points = np.array(sample_surface(udf.mesh, n_surface_points_for_metrics)[0])

    if not use_true_surface_points_as_initialization:
        initial_points = sample_uniform_3d_sphere(n_surface_points_for_metrics, radius=1)
    else:
        initial_points = deepcopy(true_surface_points)
        
    new_points = descend_udf(neural_udf, initial_points)
    
    metrics = {}

    metrics['hausdorff'] = compute_hausdorff_distance(true_surface_points, new_points)
    metrics['chamfer'] = compute_chamfer_distance(true_surface_points, new_points)
    metrics['W1'] = compute_W1_distance(true_surface_points, new_points)
    metrics['abs_neural_udf_on_true_surface_points'] = np.abs(neural_udf(true_surface_points)).mean()
    metrics['abs_true_udf_on_reconstructed_pc'] = udf(new_points).mean()
    
    if compute_correlations_with_indicator:
        
        indicator_values = compute_indicator_on_surface(true_surface_points, indicator, k_neighbors)
        warm_points = true_surface_points[indicator_values > decision_threshold]
        cool_points = true_surface_points[indicator_values <= decision_threshold]
    
        metrics['abs_neural_udf_on_warm_points'] = np.abs(neural_udf(warm_points)).mean()
        metrics['abs_neural_udf_on_cool_points'] = np.abs(neural_udf(cool_points)).mean()

        metrics['correlation_loss_est'] = np.corrcoef(np.abs(neural_udf(true_surface_points)).flatten(), indicator_values)[0,1]
        is_warm = np.int32(indicator_values > decision_threshold)
        metrics['correlation_loss_is_warm'] = np.corrcoef(np.abs(neural_udf(true_surface_points)).flatten(), is_warm)[0,1]
    
    return metrics
