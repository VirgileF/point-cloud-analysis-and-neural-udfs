from src.utils.system import load_json
from src.neural_udf_evaluation import load_results_from_single_run, compute_metrics

import os
from itertools import product

import time as T

import numpy as np
import pandas as pd

def compute_and_store_evaluation_metrics(path_to_results, csv_file_name):

    # Runtime
    t0 = T.time()

    # Load experiment parameters
    experiment_parameters = load_json(os.path.join(path_to_results, 'experiment_parameters.json'))

    # Sanity check
    assert os.path.normpath(path_to_results) == os.path.normpath(experiment_parameters['path_to_results'])
    
    # Training parameters' values as a list of list
    list_of_list_parameters = []
    for parameter_name in experiment_parameters['training_parameters_values']:
        list_of_list_parameters.append(experiment_parameters['training_parameters_values'][parameter_name])
    
    # List of parameter combinations
    combinations_list = list(product(*list_of_list_parameters))
    n_combinations = len(combinations_list)
    
    metrics_list = []
    
    print(f'Computing metrics for {n_combinations} parameter combinations.')
    for i in range(n_combinations):
        
        print(f'Step {i+1}/{n_combinations}...', end='\r')
        
        neural_udf_path = os.path.join(path_to_results, 'neural_udfs', f'neural_udf_{i}')
        
        if os.path.exists(neural_udf_path):
            X, udf, neural_udf, training_parameters = load_results_from_single_run(path_to_results, i)
            
            metrics_dict = compute_metrics(udf,
                                           neural_udf,
                                           n_surface_points_for_metrics=training_parameters['n_surface_points_for_indicator'],
                                           use_true_surface_points_as_initialization=True,
                                           compute_correlations_with_indicator=True,
                                           indicator=training_parameters['indicator'],
                                           k_neighbors=training_parameters['k_neighbors'],
                                           decision_threshold=training_parameters['decision_threshold']
            )
            metrics = [metrics_dict[key] for key in metrics_dict]
            metrics_list.append(metrics)
        else:
            print(f'Not loading any result for path={neural_udf_path}, parameters={combinations_list[i]}')
            
    columns = list(experiment_parameters['training_parameters_values'].keys()) + list(metrics_dict.keys())
    data = [list(combination)+metrics for (combination, metrics) in zip(combinations_list, metrics_list)]
        
    df = pd.DataFrame(data, columns=columns)
    
    path_to_metrics_df = os.path.join(experiment_parameters['path_to_results'], csv_file_name)
    df.to_csv(path_to_metrics_df)
    
    print(f'Saved metrics to {path_to_metrics_df} in {np.round(T.time()-t0,1)} s.')

if __name__ == '__main__':

    import sys

    assert len(sys.argv) == 3

    path_to_results = str(sys.argv[1])
    csv_file_name = str(sys.argv[2])

    compute_and_store_evaluation_metrics(path_to_results, csv_file_name)

    


