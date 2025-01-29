import numpy as np
import pandas as pd
import itertools
from scipy.spatial import distance
# from cutnorm import compute_cutnorm
import evaluation
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from cutnorm import compute_cutnorm
from scipy.stats import entropy

#import warnings
#warnings.filterwarnings("ignore")

import random
random.seed(3110)

def plot_distributions(distributions: list,
                       labels: list,
                       title: str,
                       x_label: str,
                       y_label: str,
                       output_path: str):
    """
    Plot the given distributions with the specified labels and title.
    The plot is saved to the specified output path.

    :param distributions: The distributions to plot
    :param labels: The labels for the distributions
    :param title: The title of the plot
    :param x_label: The label for the x-axis
    :param y_label: The label for the y-axis
    :param output_path: The path to save the plot
    """
    # Create a new figure
    plt.figure(figsize=(10, 6))

    bins = 50 if 'flux' in title.lower() else 30
    
    # Plot the distributions as histograms
    
    shift = np.repeat(-0.25, bins)
    for distribution, label in zip(distributions, labels):
        if 'flux' in title.lower():
            x = np.arange(len(distribution))
            plt.bar(x + shift, distribution, width=0.25, alpha=0.5, align='center', label=label)
            plt.yscale('log')
            shift += np.repeat(0.25, len(distribution))
        else:
            distr, bins = np.histogram(distribution, bins=bins, density=True)
            x = np.arange(len(distr))
            plt.bar(x + shift, distr, width=0.25, alpha=0.5, align='center', label=label)
            plt.yscale('log')
            shift += np.repeat(0.25, len(distr))

    # Add a legend
    plt.legend()

    # Add a title and labels
    plt.title(title)
    plt.xlabel(x_label + ' bins')
    plt.ylabel(y_label + ' (log scale)')

    # Save the plot to the specified output path
    plt.savefig(output_path)

def plot_normalized_distributions(distributions: list,
                                    labels: list,
                                    title: str,
                                    x_label: str,
                                    y_label: str,
                                    output_path: str):
        """
        Plot the given distributions with the specified labels and title.
        The plot is saved to the specified output path.
    
        :param distributions: The distributions to plot
        :param labels: The labels for the distributions
        :param title: The title of the plot
        :param x_label: The label for the x-axis
        :param y_label: The label for the y-axis
        :param output_path: The path to save the plot
        """
        # Create a new figure
        plt.figure(figsize=(10, 6))
    
        # Plot the distributions as histograms
        for distribution, label in zip(distributions, labels):
            plt.hist(distribution, bins=30, alpha=0.5, label=label, density=True)
    
        # Add a legend
        plt.legend()
    
        # Add a title and labels
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
    
        # Save the plot to the specified output path
        plt.savefig(output_path)

def calculate_kl_divergence(dist1, dist2, bins=30, epsilon=1e-10):
    """
    Calculate the KL divergence between two distributions.

    :param dist1: First distribution (list or numpy array)
    :param dist2: Second distribution (list or numpy array)
    :param bins: Number of bins to use for the histograms
    :param epsilon: Smoothing value to avoid log(0)
    :return: KL divergence value
    """
    # Create histograms (normalized to represent probabilities)
    hist1, bin_edges = np.histogram(dist1, bins=bins, density=True)
    hist2, _ = np.histogram(dist2, bins=bin_edges, density=True)
    
    # Add smoothing to avoid zeros
    hist1 = hist1 + epsilon
    hist2 = hist2 + epsilon

    # Normalize the histograms
    hist1 /= np.sum(hist1)
    hist2 /= np.sum(hist2)
    
    # Compute KL divergence
    kl_div = entropy(hist1, hist2)
    
    return kl_div

def get_rmse(x, y):
    return np.sqrt(np.mean(np.subtract(x,y) ** 2))

def get_exp_dist(lista, paired = False, method = "weight-edge", distanze = None):

    exp = []
    js = 0

    if paired:
        insieme = lista
    else:
        insieme = itertools.combinations(lista, r =2)

    for pair in insieme:


        if method == "weight-edge":
            weights_1 = pair[0].flatten()
            weights_2 = pair[1].flatten()
        elif method == "weight-dist":
            weights_1 = (pair[0]/distanze).flatten()
            weights_2 = (pair[1]/distanze).flatten()


        massim = max(max(weights_1), max(weights_2))
        bins = np.arange(0,np.ceil(massim)  )

        values_1, base_1 = np.histogram(weights_1, bins=bins, density=1)
        values_2, base_2 = np.histogram(weights_2, bins=bins, density=1)

        js = distance.jensenshannon(np.asarray(values_1), np.asarray(values_2), np.e)

        exp.append(js)

    return exp

def get_exp_measures(lista, paired = False, method = "cutnorm"):
    exp = []

    if paired:
        insieme = lista
    else:
        insieme = itertools.combinations(lista, r=2)

    if method == "topo":
            exp = []

            for pair in tqdm(insieme):

                G1 = nx.from_numpy_array(np.matrix(pair[0]), create_using=nx.DiGraph)
                G2 = nx.from_numpy_array(np.matrix(pair[1]), create_using=nx.DiGraph)
                cl1 = list(nx.clustering(G1,weight='weight').values())
                cl2 = list(nx.clustering(G2,weight='weight').values())
                rmse = get_rmse(cl1,cl2)
                nrmse = rmse/(max(np.max(cl1),np.max(cl2)) - min(np.min(cl1), np.min(cl2)))
                exp.append(nrmse)
            return exp
    elif method == "degree":
            exp = []

            for pair in tqdm(insieme):
                G1 = nx.from_numpy_array(np.matrix(pair[0]), create_using=nx.DiGraph)
                G2 = nx.from_numpy_array(np.matrix(pair[1]), create_using=nx.DiGraph)
                deg1 = [val for (node, val) in G1.degree(weight = "weight")]
                deg2 = [val for (node, val) in G2.degree(weight = "weight")]
                rmse = get_rmse(deg1,deg2)
                nrmse = rmse/(max(np.max(deg1),np.max(deg2)) - min(np.min(deg1), np.min(deg2)))
                exp.append(nrmse)
            return exp
    elif method == "indegree":
            exp = []

            for pair in tqdm(insieme):
                G1 = nx.from_numpy_array(np.matrix(pair[0]), create_using=nx.DiGraph)
                G2 = nx.from_numpy_array(np.matrix(pair[1]), create_using=nx.DiGraph)
                deg1 = [val for (node, val) in G1.in_degree(weight = "weight")]
                deg2 = [val for (node, val) in G2.in_degree(weight = "weight")]
                rmse = get_rmse(np.array(deg1),np.array(deg2))
                nrmse = rmse/(max(np.max(deg1),np.max(deg2)) - min(np.min(deg1), np.min(deg2)))
                exp.append(nrmse)
            return exp
    elif method == "outdegree":
            exp = []

            for pair in tqdm(insieme):
                G1 = nx.from_numpy_array(np.matrix(pair[0]), create_using=nx.DiGraph)
                G2 = nx.from_numpy_array(np.matrix(pair[1]), create_using=nx.DiGraph)
                deg1 = [val for (node, val) in G1.out_degree(weight = "weight")]
                deg2 = [val for (node, val) in G2.out_degree(weight = "weight")]
                rmse = get_rmse(np.array(deg1),np.array(deg2))
                nrmse = rmse/(max(np.max(deg1),np.max(deg2)) - min(np.min(deg1), np.min(deg2)))
                exp.append(nrmse)
            return exp
    elif method == "cpc":
        misura =  evaluation.common_part_of_commuters
        exp=[]

        for pair in tqdm(insieme):
            weights_1 = np.array(pair[0]).flatten()
            weights_2 = np.array(pair[1]).flatten()
            m = misura(weights_1, weights_2)
            exp.append(m)
        return exp

    elif method == "rmse":
        misura = get_rmse
        exp=[]
        for pair in tqdm(insieme):
            weights_1 = (pair[0]).flatten()
            weights_2 = (pair[1]).flatten()
            rmse = misura(weights_1, weights_2)
            nrmse = rmse/(max(np.max(weights_1),np.max(weights_2)) - min(np.min(weights_1), np.min(weights_2)))
            exp.append(nrmse)
        return exp
    elif method == "cutnorm":
        k = 0
        for pair in tqdm(insieme):
                _, cutn_sdp, _ = compute_cutnorm(pair[0], pair[1])
                exp.append(cutn_sdp)
                k+=1
        return exp
    else:
        raise ValueError("Invalid method '{}'".format(method))


if __name__ == "__main__":
    pass
