import os
import shutil
import pickle
import json

def create_directory(directory_path):
    # Check if the directory exists
    if os.path.exists(directory_path):
        # If it exists, delete it
        shutil.rmtree(directory_path)
        print(f"Deleted existing directory: {directory_path}")

    # Create the directory
    os.makedirs(directory_path)
    print(f"Created directory: {directory_path}")

def dump_pickle(python_object, path):
    
    with open(path, 'wb') as f:
        pickle.dump(python_object, f)

def dump_json(python_dict, path):
    
    # Write the dictionary to a JSON file
    with open(path, 'w') as json_file:
        json.dump(python_dict, json_file, indent=4)

def load_pickle(path):

    with open(path, 'rb') as pickle_file:
        python_object = pickle.load(pickle_file)

    return python_object

def load_json(path):

    # Read the JSON file and load it into a dictionary
    with open(path, 'r') as json_file:
        python_dict = json.load(json_file)

    return python_dict
