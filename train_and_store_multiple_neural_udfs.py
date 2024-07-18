
from src.utils.system import create_directory, dump_pickle, dump_json, load_json
from src.utils.shape_encoding import load_mesh, Udf3d_ShapeNet
from src.training_points_sampling import sample_training_points
from src.neural_udf_training import train_mlp_network, NeuralUdf

import time as T
import os
import numpy as np
from itertools import product
from copy import deepcopy
from multiprocessing import Pool
from pprint import pprint


def train_and_store_single_neural_udf(training_parameters):

    t0=T.time()
    
    # Load mesh
    mesh = load_mesh(training_parameters['path_to_meshes'], training_parameters['shape_index'])
    udf = Udf3d_ShapeNet(mesh)

    # Sample training points
    t=T.time()
    X, y = sample_training_points(mesh,
                                  training_parameters['n_training_points'],
                                  training_parameters['proportion_training_points_from_surface'],
                                  training_parameters['oversampling_strength'],
                                  training_parameters['n_surface_points_for_indicator'],
                                  training_parameters['indicator'],
                                  training_parameters['k_neighbors'],
                                  training_parameters['decision_threshold']
                                  )
    print(f'Training points sampled in {T.time()-t} s.')
    
    # Train neural UDF
    t=T.time()
    model, input_scaler, output_scaler = train_mlp_network(X,
                                                           y,
                                                           training_parameters['n_blocks'],
                                                           training_parameters['n_units_per_layer'],
                                                           training_parameters['learning_rate'],
                                                           training_parameters['batch_size'],
                                                           training_parameters['n_epochs'],
                                                           training_parameters['random_seed'],
                                                           plot_loss=False,
                                                           print_loss=True
                                                           )
    neural_udf = NeuralUdf(model, input_scaler, output_scaler)
    print(f'Neural UDF trained in {T.time()-t} s.')
    
    # Store results
    t=T.time()
    create_directory(training_parameters['neural_udf_subfolder'])
    dump_pickle(X, os.path.join(training_parameters['neural_udf_subfolder'], 'X.pickle'))
    dump_pickle(neural_udf, os.path.join(training_parameters['neural_udf_subfolder'], 'neural_udf.pickle'))
    print(f'Results stored in {T.time()-t} s.')

    # Compute runtime
    runtime = T.time()-t0
    print(f'Function train_and_store_single_neural_udf runtime: {np.round(runtime, 1)} s.')
    training_parameters['runtime'] = runtime

    # Store training_parameters
    dump_json(training_parameters, os.path.join(training_parameters['neural_udf_subfolder'], 'training_parameters.json'))
    
    return X, udf, neural_udf


def train_and_store_multiple_neural_udfs(experiment_parameters):

    t0 = T.time()

    # Create directory 
    create_directory(experiment_parameters['path_to_results'])
    
    # Store xp parameters
    dump_json(experiment_parameters, os.path.join(experiment_parameters['path_to_results'], 'experiment_parameters.json'))
    
    # Training parameters' values as a list of list
    list_of_list_parameters = []
    for parameter_name in experiment_parameters['training_parameters_values']:
        list_of_list_parameters.append(experiment_parameters['training_parameters_values'][parameter_name])
    
    # List of parameter combinations
    combinations_list = list(product(*list_of_list_parameters))
    
    # Create list of training parameters dictionaries
    training_parameters_list_of_dicts = []
    for i, combination in enumerate(combinations_list):
        training_parameters = deepcopy(experiment_parameters['training_parameters_values'])
        training_parameters['path_to_meshes'] = experiment_parameters['path_to_meshes']
        training_parameters['neural_udf_subfolder'] = os.path.join(experiment_parameters['path_to_results'], 'neural_udfs', f'neural_udf_{i}')
        for j, parameter_name in enumerate(experiment_parameters['training_parameters_values']):
            training_parameters[parameter_name] = combination[j]
        training_parameters_list_of_dicts.append(training_parameters)
            
    # Call function
    if experiment_parameters['n_cpus'] == 1:
        for i, training_parameters in enumerate(training_parameters_list_of_dicts):
            print('==============================')
            print('Training single neural UDF with these parameters: ')
            pprint(training_parameters, sort_dicts=False)
            train_and_store_single_neural_udf(training_parameters)

    else:
        p = Pool(experiment_parameters['n_cpus'])
        p.map(train_and_store_single_neural_udf, training_parameters_list_of_dicts)
        p.close()

    # Compute runtime
    total_runtime = T.time()-t0
    experiment_parameters['total_runtime'] = total_runtime
    print(f'Total runtime for all the neural UDFs: {np.round(total_runtime,1)} s.')
    
    # Store xp parameters (with total runtime)
    dump_json(experiment_parameters, os.path.join(experiment_parameters['path_to_results'], 'experiment_parameters.json'))


if __name__ == '__main__':

    import sys

    assert len(sys.argv) == 2

    experiment_json = str(sys.argv[1])

    experiment_parameters = load_json(experiment_json)

    print('Training and storing neural UDFs with these experiment parameters:')
    pprint(experiment_parameters, sort_dicts=False)

    train_and_store_multiple_neural_udfs(experiment_parameters)