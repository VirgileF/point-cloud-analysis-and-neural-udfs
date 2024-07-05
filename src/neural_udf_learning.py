
def load_mesh(category, shape_index):

    # compute path

    # load mesh (Trimesh mesh from '.stl' file)

    return B

def sample_uniform_unit_ball(n_samples, random_seed):

    # sample from 4d Gaussian

    # extract 3 first coordinates


    return samples

def sample_training_points(B, training_params, random_seed):

    # rescale B to fit the unit ball

    # sample points on the surface B
    
    # compute descriptor for edge detection

    # identify the points located on surface edges

    # identify the points not located on surface edges

    # compute the weights of the three sub-distributions

    # sample from the uniform distribution on the unit ball

    # sample from the uniform distribution on the surface edges

    # sample from the uniform distribution on the quasi-planar areas

    # concatenate the three sub-datasets


    return X

def compute_output_targets(X, B):

    # convert mesh to pyvista format

    # compute the true UDF


    return y

def train_neural_udf(B, training_params):


    return neural_udf

def store_neural_udf(neural_udf):

    return None