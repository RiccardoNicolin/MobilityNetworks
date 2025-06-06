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
from scipy.sparse.linalg import eigsh
import glob
from PIL import Image
import matplotlib.pyplot as plt


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
        print(f"KL divergence [degree] between real and MoGAN: {kl_mogan}")
        print(f"KL divergence [degree] between real and ours: {kl_ours}")
        
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
        print(f"KL divergence [indegree] between real and MoGAN: {kl_mogan}")
        print(f"KL divergence [indegree] between real and ours: {kl_ours}")


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
        print(f"KL divergence [outdegree] between real and MoGAN: {kl_mogan}")
        print(f"KL divergence [outdegree] between real and ours: {kl_ours}")
        
    if 'flux' in distribution_names:
        test_dist, _ = get_flux_metric(test_set)
        fake_dist, _ = get_flux_metric(fake_set)
        mogan_dist, _ = get_flux_metric(mogan_set)

        plot_distributions([test_dist, fake_dist, mogan_dist],
                            ['Real', 'Fake', 'MoGAN'], 
                            'Flux Distribution', 'Normalised Flux', 'Frequency', 
                            f"{run_name}/evaluations/flux_distribution.png")
        
        kl_mogan = calculate_kl_divergence(test_dist, mogan_dist)
        kl_ours = calculate_kl_divergence(test_dist, fake_dist)
        print(f"KL divergence [flux] between real and MoGAN: {kl_mogan}")
        print(f"KL divergence [flux] between real and ours: {kl_ours}")
        
    if "embedding" in distribution_names:
        compute_embedding_metrics(test_set, fake_set, 'Real', 'Fake', run_name)
        compute_embedding_metrics(test_set, mogan_set, 'Real', 'MoGAN', run_name)
        compute_embedding_metrics(fake_set, mogan_set, 'Fake', 'MoGAN', run_name)



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

def compute_embedding(adjs):
    embedding_set = []
    for adj in adjs:
        A = np.array(adj)
        # Compute degree matrix (weighted degrees)
        D = np.diag(A.sum(axis=1))

        # Weighted unnormalized Laplacian
        L = D - A

        # Number of dimensions for the embedding
        k = 2

        # Compute the smallest k eigenvectors of the Laplacian
        try:
            eigenvalues, eigenvectors = eigsh(L, k=k, which='SM', maxiter=1000, tol=0)  # 'SM' for smallest magnitude
            embedding = eigenvectors
            embedding_set.append(embedding)
        except Exception as e:
            print(f"Error computing embedding: {e}")
    return embedding_set

def compute_inital_pop_vs_sparse_pop(population):

    random.seed(None)

    sparse_set = []
    for individual in population:

        genome = individual / 327.0
        sparse_set.append(Organism(64, 
                                    random.uniform(0, np.mean(genome)**5),
                                    [0,327], 
                                    genome=genome).adjacencyMatrix)
    
    compute_embedding_metrics(population, sparse_set, 'Initial Population', 'Sparse Set', 'data/topo_flux_degree/0')


def compute_embedding_metrics(set1, set2, name_set1, name_set2, run_name):
    embedding_set1 = compute_embedding(set1)
    embedding_set2 = compute_embedding(set2)
    
    import matplotlib.pyplot as plt

    # Compute the mean of the embeddings for each set
    mean_embedding_set1 = np.mean(embedding_set1, axis=0)
    mean_embedding_set2 = np.mean(embedding_set2, axis=0)

    centroid_set1 = np.mean(mean_embedding_set1, axis=0)
    centroid_set2 = np.mean(mean_embedding_set2, axis=0)

    print(f"Centroid of {name_set1}: {centroid_set1}")
    print(f"Centroid of {name_set2}: {centroid_set2}")

    # Plot the embeddings
    plt.figure(figsize=(8, 6))
    plt.scatter(mean_embedding_set1[:, 0], mean_embedding_set1[:, 1], label=name_set1, alpha=0.5)
    plt.scatter(mean_embedding_set2[:, 0], mean_embedding_set2[:, 1], label=name_set2, alpha=0.5)
    plt.scatter(centroid_set1[0], centroid_set1[1], label=f'Centroid {name_set1}', color='blue', marker='x', s=100)
    plt.scatter(centroid_set2[0], centroid_set2[1], label=f'Centroid {name_set2}', color='red', marker='x', s=100)
    plt.title('2D Embeddings')
    plt.xlabel('Embedding Dimension 1')
    plt.ylabel('Embedding Dimension 2')
    plt.legend()
    plt.savefig(f"{run_name}/evaluations/{name_set1}_vs_{name_set2}_embedding.png")

def plot_results_summary(path):
    # Read all images in the specified path
    image_files = glob.glob(os.path.join(path, "*.png"))
    images = [Image.open(image_file) for image_file in image_files]

    distribution_images = []
    embedding_images = []
    other_images = []

    for img in images:
        if 'distribution' in os.path.basename(img.filename):
            distribution_images.append(img)
        elif 'embedding' in os.path.basename(img.filename):
            embedding_images.append(img)
        else:
            other_images.append(img)

    # Plot the distribution images
    plot_set_images(distribution_images, os.path.join(path, 'distribution_summary.png'))
    plot_set_images(embedding_images, os.path.join(path, 'embedding_summary.png'))
    plot_set_images(other_images, os.path.join(path, 'other_summary.png'), images_per_row=3)
   
def plot_set_images(images, path, images_per_row=2):
    """
    Plot a set of images in a grid with a maximum number of images per row.
    
    Args:
        images (list): List of PIL Image objects.
        path (str): Path to save the plotted grid as an image.
        images_per_row (int): Maximum number of images in a single row (default: 4).
    """
    # Determine grid dimensions
    num_images = len(images)
    if num_images == 0:
        return
    rows = (num_images + images_per_row - 1) // images_per_row  # Ceiling division
    cols = min(images_per_row, num_images)

    # Set up the figure
    fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), dpi=300)
    axs = axs.ravel() if num_images > 1 else [axs]  # Flatten axis array for easier indexing

    for i in range(len(axs)):
        if i < num_images:
            axs[i].imshow(images[i])
            axs[i].axis('off')  # Hide axes
        else:
            axs[i].axis('off')  # Turn off unused subplots

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

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
            'cpc', 
            'cutnorm',
            'topo',
        ]
        distribution_names = [
            'degree',
            'indegree',
            'outdegree',
            'flux',
            'embedding',
            # 'initial_vs_sparse'
        ]
        compute_metrics(metrics_name, obj_list, 'BikeCHI', os.path.join(run_name, str(run)))
        get_distributions(distribution_names, obj_list, 'BikeCHI', os.path.join(run_name, str(run)))
        plot_results_summary(os.path.join(run_name, str(run), 'evaluations'))

    # path_inital_population = "GAN-flow/BikeCHI/v_train.txt"
    # with open(path_inital_population, 'rb') as file:
    #     initial_population = pickle.load(file)
    # compute_inital_pop_vs_sparse_pop(initial_population)


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