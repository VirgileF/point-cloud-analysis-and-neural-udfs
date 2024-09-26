# Point cloud analysis and distance function learning

Code repository supporting preprint https://hal.science/IMT/hal-04568278v1

## Data

Link towards preprocessed meshes in .stl format
https://www.dropbox.com/scl/fo/85niz2p9qis9subfbxijz/AKWMU92jwh0c4PXCWIhUNfg?rlkey=rfgfs3zozsdbmpd4nu6zga42z&e=3&st=7s7a3zgz&dl=0

## Requirements :

- Python 3.8.19

Python packages needed are listed in the file ```requirements.txt```.

### Main packages:

- numpy 1.24.4
- trimesh 4.4.1
- vtk 9.3.1
- matplotlib 3.7.5
- scipy 1.10.1
- scikit-learn 1.3.2
- torch 2.2.2
- pandas 2.0.3
- POT 0.9.4
- pyvista 0.43.10

## Point cloud analysis

To plot High Frequency Area and Pauly indicators on surfaces from ShapeNet dataset (Figures 4.1, 4.2 and 4.3 in the aforementioned paper), one can use the notebook ```notebooks/plot_indicators_on_surfaces.ipynb```. It mainly uses functions from python script ```src/point_cloud_analysis.py```.

## Neural UDFs training

In order to train Neural UDFs of surfaces from ShapeNet dataset, one must run the script ```train_and_store_multiple_neural_udfs.py```. The parameters of the training must be specified in a json file. Some examples of experiments are given in the folder ```experiments```. The Neural UDFs thus generated are stored as pickle files in the folder ```path_to_results``` (see templates of json files).

For example, one can run:

```bash
python train_and_store_multiple_neural_udfs.py experiments/xp_chairs_3.json
```

This will train the Neural UDFs of three chair surfaces. The indices of chairs, the training parameters and the path in which the Neural UDFs are stored are specified in the json file.

## Neural UDFs evaluation

In order to evaluate the precision of Neural UDFs from ShapeNet dataset, one must run the script ```compute_and_store_evaluation_metrics.py```. The only inline parameter that needs to be specified is the path to the folder containing the Neural UDFs to be evaluated. It corresponds to the parameter ```path_to_results``` specified during training. 

To follow the same example, one can run:

```bash
python compute_and_store_evaluation_metrics.py "<path_to_results>" "<csv_file>"
```
where ```"<path_to_results>"``` is the path specified in the json file ```experiments/xp_chairs_3.json```.

This will store all the evaluation metrics in the csv file ```"<path_to_results>/<csv_file>"```. 

## Interpret results

The notebook ```notebooks/plot_histograms_of_improvements.ipynb``` allows to plot histograms of metrics as in the Figure 4.6 of the article. It uses data aggregation on the csv of metrics built during evaluation step. 

All the plots are stored in the subfolder ```notebooks/figures.```.

## Main references

DeepSDF: https://arxiv.org/abs/1901.05103 (git: https://github.com/facebookresearch/DeepSDF)

ShapeNet: https://shapenet.org/







