import itertools
import pickle
import sys

sys.path.append("graph_evolution")

from graph_evolution.organism import Organism
# from organism import Organism
from random import random,uniform
import numpy as np
from PIL import Image
from utils_gan_flow import get_exp_measures
import random
import os

def array_to_greyscale_image(array: np.ndarray, output_path: str):
    """
    Convert a 2D numpy array to a greyscale image and save it to the specified path.

    Parameters:
    array (np.ndarray): 2D numpy array to be converted to a greyscale image.
    output_path (str): Path where the greyscale image will be saved.
    """

    # Transform a list of lists into a numpy array
    if isinstance(array, list):
        array = np.array(array)
    if array.ndim != 2:
        raise ValueError("Input array must be 2D")

    # Normalize the array to be in the range 0-255
    normalized_array = ((array - array.min()) / (array.max() - array.min()) * 255).astype(np.uint8)

    # Create an image from the normalized array
    image = Image.fromarray(normalized_array, mode='L')

    # Save the image to the specified path
    image.save(output_path)

def compute_metrics(metric_names, population, dataset, run_name):


    fake_set = [organism.adjacencyMatrix for organism in population]
    test_set_path = f"GAN-flow/{dataset}/v_test.txt"
    with open(test_set_path, 'rb') as file:
        test_set = pickle.load(file)

    number_of_items = len(fake_set)
    uno = random.sample(test_set, number_of_items)
    due = random.sample(fake_set, number_of_items)
    mixed_set_pairs = [pair for pair in itertools.product(uno , due)]
    len(fake_set), len(test_set), len(mixed_set_pairs), number_of_items

    for metric_name in metric_names:

        # Load the already compute metrics
        test_metric_path = f"GAN-flow/{dataset}/experiments/{metric_name}/MoGAN/1.txt"
        with open(test_metric_path, 'rb') as file:
            test_metric = pickle.load(file)

        fake_metric = get_exp_measures(fake_set, method=metric_name)
        fake_metric_path = f"{run_name}/experiments/{metric_name}_fake.pkl"
        with open(fake_metric_path, 'wb') as file:
            pickle.dump(fake_metric, file)

        mixed_metric = get_exp_measures(mixed_set_pairs, paired = True,  method=metric_name)
        mixed_metric_path = f"{run_name}/experiments/{metric_name}_mixed.pkl"
        with open(mixed_metric_path, 'wb') as file:
            pickle.dump(mixed_metric, file)

        mogan_metric_path = f"GAN-flow/{dataset}/experiments/{metric_name}/MoGAN/2.txt"
        with open(mogan_metric_path, 'rb') as file:
            mogan_metric = pickle.load(file)

        plot_metrics(test_metric,fake_metric,mixed_metric, mogan_metric, metric_name, dataset)


def plot_metrics(test_metric,fake_metric,mixed_metric, mogan_metric, metric_name, dataset):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,6))
    plt.hist(test_metric, bins = 50, alpha = 0.5, label = 'Real')
    plt.hist(fake_metric, bins = 50, alpha = 0.5, label = 'Fake')
    plt.hist(mixed_metric, bins = 50, alpha = 0.5, label = 'Mixed')
    plt.hist(mogan_metric, bins = 50, alpha = 0.5, label = 'MoGAN')
    plt.title(f'{metric_name} - {dataset}')
    plt.legend()
    plt.savefig(f"results/{metric_name}_{dataset}.png")

def get_all_npy_files(path):
    """
    Get a list of all .npy files in the specified directory.

    Parameters:
    path (str): Directory path to search for .npy files.

    Returns:
    list: List of paths to .npy files.
    """
    npy_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".npy"):
                npy_files.append(os.path.join(root, file))
    return npy_files


if __name__ == "__main__":
    
    run_name = 'data/sample'
    with open(f'data/sample/0/final_pop.pkl', 'rb') as file:
        obj_list = pickle.load(file)
    
    metrics_name= ['cutnorm']#['indegree', 'degree', 'topo', 'weight', 'cpc']
    
    compute_metrics(metrics_name, obj_list, 'BikeCHI', run_name)


    # for i,organism in enumerate(obj_list[:5]):
    #     # organism.saveGraphFigure(f'data/sample/0/organismo_{i}.png')
    #     array_to_greyscale_image(organism.adjacencyMatrix, f'data/sample/0/organismo_{i}.png')


    # print(obj_list)


    # Example usage
    npy_files = get_all_npy_files("GAN-flow/adj/BikeCHI")

    for i, npy_file in enumerate(npy_files[:10]):
        npy_obj = np.load(npy_file)
        npy_obj = npy_obj.astype(np.float64)
        print("npy sum", np.sum(npy_obj))
        npy_obj /= 327.0

        array_to_greyscale_image(npy_obj, f"test/original{i}.png")
        org = Organism(64, uniform(0, np.max(npy_obj)**3), [0, 327], npy_obj)
        print("org sum", np.sum(org.adjacencyMatrix))
        array_to_greyscale_image(org.adjacencyMatrix, f"test/rescaled{i}.png")

        diff = np.abs(npy_obj - np.array(org.adjacencyMatrix)/327)
        print("diff sum", np.mean(diff)*327)
        array_to_greyscale_image(diff/327, "test/diff.png")