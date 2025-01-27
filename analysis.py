import itertools
import pickle
import sys
import json
import networkx as nx
import numpy as np
import random
import os

sys.path.append("graph_evolution")
from graph_evolution.organism import Organism
from PIL import Image
from utils_gan_flow import calculate_kl_divergence, get_exp_measures, plot_distributions, plot_normalized_distributions
from reference_metrics import (
    get_degree_metric,
    get_indegree_metric,
    get_outdegree_metric,
    get_flux_metric
)
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

    number_of_items = min(len(fake_set), len(test_set))
    uno = random.sample(test_set, number_of_items)
    due = random.sample(fake_set, number_of_items)
    mixed_set_pairs = [pair for pair in itertools.product(uno , due)]
    len(fake_set), len(test_set), len(mixed_set_pairs), number_of_items

    if len(fake_set) > number_of_items:
        fake_set = random.sample(fake_set, number_of_items)
    if len(test_set) > number_of_items:
        test_set = random.sample(test_set, number_of_items)

    
    os.makedirs(os.path.join(run_name, 'evaluations'), exist_ok=True)
    for metric_name in metric_names:
        # Load the already compute metrics
        test_metric_path = f"GAN-flow/{dataset}/experiments/{metric_name}/MoGAN/1.txt"
        with open(test_metric_path, 'rb') as file:
            test_metric = pickle.load(file)

        fake_metric_path = f"{run_name}/evaluations/{metric_name}_fake.pkl"
        if not os.path.exists(fake_metric_path):
            fake_metric = get_exp_measures(fake_set, method=metric_name)
            with open(fake_metric_path, 'wb') as file:
                pickle.dump(fake_metric, file)
        else:
            with open(fake_metric_path, 'rb') as file:
                fake_metric = pickle.load(file)

        mixed_metric_path = f"{run_name}/evaluations/{metric_name}_mixed.pkl"
        if not os.path.exists(mixed_metric_path):
            mixed_metric = get_exp_measures(mixed_set_pairs, paired = True,  method=metric_name)
            with open(mixed_metric_path, 'wb') as file:
                pickle.dump(mixed_metric, file)
        else:
            with open(mixed_metric_path, 'rb') as file:
                mixed_metric = pickle.load(file)

        mogan_metric_path = f"GAN-flow/{dataset}/experiments/{metric_name}/MoGAN/2.txt"
        with open(mogan_metric_path, 'rb') as file:
            mogan_metric = pickle.load(file)

        plot_metrics(test_metric,fake_metric,mixed_metric, mogan_metric, metric_name, dataset, run_name)

def get_distributions(distribution_names, population, dataset, run_name):
    fake_set = [organism.adjacencyMatrix for organism in population]
    test_set_path = f"GAN-flow/{dataset}/v_test.txt"
    with open(test_set_path, 'rb') as file:
        test_set = pickle.load(file)

    mogan_set_path = f"GAN-flow/{dataset}/fake_set.txt"
    with open(mogan_set_path, 'rb') as file:
        mogan_set = pickle.load(file)

    if 'degree' in distribution_names:
        test_dist, _ = get_degree_metric(test_set)
        fake_dist, _ = get_degree_metric(fake_set)
        mogan_dist, _ = get_degree_metric(mogan_set)

        plot_distributions([test_dist, fake_dist,mogan_dist],
                           ['Real', 'Fake', 'MoGAN'], 
                           'Degree Distribution', 'Degree', 'Frequency', 
                           f"{run_name}/evaluations/degree_distribution.png")
        
        plot_normalized_distributions([test_dist, fake_dist, mogan_dist],
                           ['Real', 'Fake', 'MoGAN'], 
                           'Normalized Degree Distribution', 'Degree', 'Frequency', 
                           f"{run_name}/evaluations/normalized_degree_distribution.png")
        
        kl_mogan = calculate_kl_divergence(test_dist, mogan_dist)
        kl_ours = calculate_kl_divergence(test_dist, fake_dist)
        print(f"KL divergence between real and MoGAN: {kl_mogan}")
        print(f"KL divergence between real and ours: {kl_ours}")
        
    if 'indegree' in distribution_names:
        test_dist, _ = get_indegree_metric(test_set)
        fake_dist, _ = get_indegree_metric(fake_set)
        mogan_dist, _ = get_indegree_metric(mogan_set)

        plot_distributions([test_dist, fake_dist, mogan_dist],
                           ['Real', 'Fake', 'MoGAN'], 
                           'In-Degree Distribution', 'In-Degree', 'Frequency', 
                           f"{run_name}/evaluations/indegree_distribution.png")
        plot_normalized_distributions([test_dist, fake_dist, mogan_dist],
                            ['Real', 'Fake', 'MoGAN'], 
                            'Normalized In-Degree Distribution', 'In-Degree', 'Frequency', 
                            f"{run_name}/evaluations/normalized_indegree_distribution.png")
        
        kl_mogan = calculate_kl_divergence(test_dist, mogan_dist)
        kl_ours = calculate_kl_divergence(test_dist, fake_dist)
        print(f"KL divergence between real and MoGAN: {kl_mogan}")
        print(f"KL divergence between real and ours: {kl_ours}")


    if 'outdegree' in distribution_names:
        test_dist, _ = get_outdegree_metric(test_set)
        fake_dist, _ = get_outdegree_metric(fake_set)
        mogan_dist, _ = get_outdegree_metric(mogan_set)

        plot_distributions([test_dist, fake_dist, mogan_dist],
                           ['Real', 'Fake', 'MoGAN'], 
                           'Out-Degree Distribution', 'Out-Degree', 'Frequency', 
                           f"{run_name}/evaluations/outdegree_distribution.png")
        plot_normalized_distributions([test_dist, fake_dist, mogan_dist],
                            ['Real', 'Fake', 'MoGAN'], 
                            'Normalized Out-Degree Distribution', 'Out-Degree', 'Frequency', 
                            f"{run_name}/evaluations/normalized_outdegree_distribution.png")

        kl_mogan = calculate_kl_divergence(test_dist, mogan_dist)
        kl_ours = calculate_kl_divergence(test_dist, fake_dist)
        print(f"KL divergence between real and MoGAN: {kl_mogan}")
        print(f"KL divergence between real and ours: {kl_ours}")
        
    if 'flux' in distribution_names:
        test_dist, _ = get_flux_metric(test_set)
        fake_dist, _ = get_flux_metric(fake_set)
        mogan_dist, _ = get_flux_metric(mogan_set)

        plot_distributions([test_dist, fake_dist, mogan_dist],
                            ['Real', 'Fake', 'MoGAN'], 
                            'Flux Distribution', 'Normalised Flux', 'Frequency', 
                            f"{run_name}/evaluations/flux_distribution.png")


def plot_metrics(test_metric,fake_metric,mixed_metric, mogan_metric, metric_name, dataset, run_name):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,6))
    plt.hist(test_metric, bins = 50, alpha = 0.5, label = 'Real')
    plt.hist(fake_metric, bins = 50, alpha = 0.5, label = 'Fake')
    plt.hist(mixed_metric, bins = 50, alpha = 0.5, label = 'Mixed')
    plt.hist(mogan_metric, bins = 50, alpha = 0.5, label = 'MoGAN')
    plt.title(f'{metric_name} - {dataset}')
    plt.legend()
    plt.savefig(f"{run_name}/evaluations/{metric_name}_{dataset}.png")

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
    if len(sys.argv) < 2:
        print("Usage: python analysis.py <run_name>")
        sys.exit(1)
    run_name = sys.argv[1]    
    
    if run_name[-1] == '/':
        run_name = run_name[:-1]
    
    with open(os.path.join(run_name,'config.json'), 'rb') as file:
        config = json.load(file)

    runs = config['reps']

    for run in range(runs):
        with open(os.path.join(run_name, str(run), 'final_pop.pkl'), 'rb') as file:
            obj_list = pickle.load(file)
        
        metrics_name= [
            'indegree',
            'degree',
            'outdegree',
            # 'weight',
            # 'cpc', 
            # 'cutnorm',
            # 'topo',
        ]
        distribution_names = [
            'degree',
            'indegree',
            'outdegree',
            'flux',
        ]
        compute_metrics(metrics_name, obj_list, 'BikeCHI', os.path.join(run_name, str(run)))
        get_distributions(distribution_names, obj_list, 'BikeCHI', os.path.join(run_name, str(run)))


    # for i,organism in enumerate(obj_list[:5]):
    #     # organism.saveGraphFigure(f'data/sample/0/organismo_{i}.png')
    #     array_to_greyscale_image(organism.adjacencyMatrix, f'data/sample/0/organismo_{i}.png')


    # print(obj_list)


    # Example usage
    # npy_files = get_all_npy_files("GAN-flow/adj/BikeCHI")

    # for i, npy_file in enumerate(npy_files[:10]):
    #     npy_obj = np.load(npy_file)
    #     npy_obj = npy_obj.astype(np.float64)
    #     print("npy sum", np.sum(npy_obj))
    #     npy_obj /= 327.0

    #     array_to_greyscale_image(npy_obj, f"test/original{i}.png")
    #     org = Organism(64, uniform(0, np.max(npy_obj)**3), [0, 327], npy_obj)
    #     print("org sum", np.sum(org.adjacencyMatrix))
    #     array_to_greyscale_image(org.adjacencyMatrix, f"test/rescaled{i}.png")

    #     diff = np.abs(npy_obj - np.array(org.adjacencyMatrix)/327)
    #     print("diff sum", np.mean(diff)*327)
    #     array_to_greyscale_image(diff/327, "test/diff.png")