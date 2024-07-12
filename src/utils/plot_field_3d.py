

import torch
import numpy as np

class GridSampler():
    
    def __init__(self, dim, bounding_box):
        bounding_box = np.array(bounding_box)
        assert bounding_box.shape[0] == 2 and bounding_box.shape[1] == dim
        for i in range(dim):
            assert bounding_box[0,i] < bounding_box[1,i]
        self.dim = dim
        self.bounding_box = bounding_box
        
    def sample(self, grid_size):
        assert len(grid_size) == self.dim
        linspace_vectors = []
        for j in range(self.dim):
            linspace_vectors.append(np.linspace(self.bounding_box[0,j], self.bounding_box[1,j], grid_size[j]).astype(np.float32))
        meshgrids = np.meshgrid(*linspace_vectors)
        meshgrids = [meshgrid.reshape(-1,1) for meshgrid in meshgrids]
        samples = np.concatenate(meshgrids, axis=1)
        return torch.from_numpy(samples)

def plot_field_3d(
    function,
    bbox,
    n_values_per_axis=100, 
    color_range=None,
    ):
    
    """
    Generic function that displays interactive slices of a scalar function defined in R^3.
    Can only be used in a Python Notebook
    """

    from bokeh.palettes import diverging_palette, gray, viridis
    from bokeh.plotting import figure
    from bokeh.models import LinearColorMapper, ColorBar, ColumnDataSource
    from bokeh.io import show, output_notebook, push_notebook
    from bokeh.io.state import curstate
    from bokeh.layouts import row

    from ipywidgets import interact, widgets, Layout

    if not curstate()._notebook:
        output_notebook()
        
    assert '__call__' in function.__dir__()
    
    # Sample points on a 3D grid
    sampler = GridSampler(dim=3, bounding_box=bbox)
    voxel_grid_points = sampler.sample(grid_size=[n_values_per_axis, n_values_per_axis, n_values_per_axis]).numpy()
    
#     # Apply rotations
#     voxel_grid_points = R.from_euler('x', phi).apply(voxel_grid_points_)
#     voxel_grid_points = R.from_euler('z', theta).apply(voxel_grid_points)
    
    values = function(voxel_grid_points)
    grid = np.reshape(values, (n_values_per_axis, n_values_per_axis, n_values_per_axis), order='C')

    # Retrieve axis-values from grid
    x_values = np.unique(voxel_grid_points[:,0])
    y_values = np.unique(voxel_grid_points[:,1])
    z_values = np.unique(voxel_grid_points[:,2])

    # Slice indices
    x = x_values[n_values_per_axis//2]
    y = y_values[n_values_per_axis//2]
    z = z_values[n_values_per_axis//2]

    # Define diverging color mapper (white in zero and black on the edges)

    palette = diverging_palette(gray(256)[128:], viridis(128), n=256)
    if color_range is None:
        color_range = (-np.max(np.abs(grid)), np.max(np.abs(grid)))
    color_mapper = LinearColorMapper(
        palette=palette, 
        low=color_range[0], 
        high=color_range[1]
    )

    # X-slice

    i = np.where(x_values==x)[0][0]
    slice2d = grid[:,i,:].T
    source = ColumnDataSource({'value': [slice2d]})
    fig_x = figure(
        title='X-slice', 
        x_range=(bbox[0,1],bbox[1,1]), 
        y_range=(bbox[0,2],bbox[1,2]), 
        width=365, 
        height=250,
        x_axis_label='Y axis',
        y_axis_label='Z axis'
    )
    img_x = fig_x.image(
        'value',
        source=source,
        #[field],
        x=bbox[0,1],
        y=bbox[0,2],
        dw=bbox[1,1]-bbox[0,1],
        dh=bbox[1,2]-bbox[0,2],
        color_mapper=color_mapper
    )

    fig_x.add_layout(ColorBar(color_mapper=color_mapper), 'left')

    # Y-slice

    j = np.where(y_values==y)[0][0]
    slice2d = grid[j,:,:].T
    source = ColumnDataSource({'value': [slice2d]})
    fig_y = figure(
        title='Y-slice', 
        x_range=(bbox[0,0],bbox[1,0]), 
        y_range=(bbox[0,2],bbox[1,2]), 
        width=300, 
        height=250,
        x_axis_label='X axis',
        y_axis_label='Z axis'
    )
    img_y = fig_y.image(
        'value',
        source=source,
        x=bbox[0,0],
        y=bbox[0,2],
        dw=bbox[1,0]-bbox[0,0],
        dh=bbox[1,2]-bbox[0,2],
        color_mapper=color_mapper
    )

    # Z-slice

    k = np.where(z_values==z)[0][0]
    slice2d = grid[:,:,k]
    source = ColumnDataSource({'value': [slice2d]})
    fig_z = figure(
        title='Z-slice', 
        x_range=(bbox[0,0],bbox[1,0]), 
        y_range=(bbox[0,1],bbox[1,1]), 
        width=300, 
        height=250,
        x_axis_label='X axis',
        y_axis_label='Y axis'
    )
    img_z = fig_z.image(
        'value',
        source=source,
        x=bbox[0,0],
        y=bbox[0,1],
        dw=bbox[1,0]-bbox[0,0],
        dh=bbox[1,1]-bbox[0,1],
        color_mapper=color_mapper
    )

    show(row(fig_x, fig_y, fig_z), notebook_handle=True)

    def update(
        x=x_values[n_values_per_axis//2],
        y=y_values[n_values_per_axis//2],
        z=z_values[n_values_per_axis//2]
    ):

        i = np.where(x_values==x)[0][0]
        slice2d = grid[:,i,:].T
        img_x.data_source.data['value'] = [slice2d]

        j = np.where(y_values==y)[0][0]
        slice2d = grid[j,:,:].T
        img_y.data_source.data['value'] = [slice2d]

        k = np.where(z_values==z)[0][0]
        slice2d = grid[:,:,k]
        img_z.data_source.data['value'] = [slice2d]

        push_notebook()

    interact(
        update, 
        x=widgets.SelectionSlider(options=x_values, value=x_values[n_values_per_axis//2], layout=Layout(width='500px')),
        y=widgets.SelectionSlider(options=y_values, value=y_values[n_values_per_axis//2], layout=Layout(width='500px')),
        z=widgets.SelectionSlider(options=z_values, value=z_values[n_values_per_axis//2], layout=Layout(width='500px'))
    )