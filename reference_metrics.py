import pickle
import networkx as nx
import numpy as np
import json
from scipy.stats import norm


def get_indegree_metric(obj_list):
    distrubutions = []
    max_degree = 0
    for i,adj in enumerate(obj_list):

        graph = nx.DiGraph(np.array(adj))
        num_nodes = graph.number_of_nodes()
        degree_sequence = list(d for _, d in graph.in_degree())
        max_degree = max(max_degree, max(degree_sequence))
        freq = [0]*(num_nodes+1)
        for d in degree_sequence:
            freq[d] += 1
        distrubutions.append([x/num_nodes for x in freq])
        
    distrubutions = np.array(distrubutions)
    mean_distrubution = np.mean(distrubutions, axis=0).tolist()
    std = np.std(distrubutions, axis=0).tolist()

    return mean_distrubution, std

def get_outdegree_metric(obj_list):
    distrubutions = []
    max_degree = 0
    for i,adj in enumerate(obj_list):

        graph = nx.DiGraph(np.array(adj))
        num_nodes = graph.number_of_nodes()
        degree_sequence = list(d for _, d in graph.out_degree())
        max_degree = max(max_degree, max(degree_sequence))
        freq = [0]*(num_nodes+1)
        for d in degree_sequence:
            freq[d] += 1
        distrubutions.append([x/num_nodes for x in freq])
        
    distrubutions = np.array(distrubutions)
    mean_distrubution = np.mean(distrubutions, axis=0).tolist()
    std = np.std(distrubutions, axis=0).tolist()

    return mean_distrubution, std

def get_degree_metric(obj_list):
    distrubutions = []
    max_degree = 0
    for i,adj in enumerate(obj_list):
        graph = nx.DiGraph(np.array(adj))
        num_nodes = graph.number_of_nodes()
        degree_sequence = list(d for _, d in graph.degree())
        max_degree = max(max_degree, max(degree_sequence))
        freq = [0]*(2*num_nodes+1)
        for d in degree_sequence:
            freq[d] += 1
        distrubutions.append([x/num_nodes for x in freq])
        
    distrubutions = np.array(distrubutions)
    mean_distrubution = np.mean(distrubutions, axis=0).tolist()
    std = np.std(distrubutions, axis=0).tolist()

    return mean_distrubution, std


def get_topo_metric(obj_list):
    means = []
    stds = []
    for i,adj in enumerate(obj_list):

        graph = nx.DiGraph(np.array(adj))
        cl = list(nx.clustering(graph,weight='weight').values())
        mean = np.mean(cl)
        std = np.std(cl)
        means.append(mean)
        stds.append(std)

    mean = np.mean(means)
    std = np.mean(stds)
    return mean, std


def get_weights_metric(obj_list):
    all_weights = []
    for i,adj in enumerate(obj_list):

        weights = np.array(adj).flatten()
        all_weights.extend(weights)

    mean = np.mean(all_weights)
    std = np.std(all_weights)
    return mean, std


if __name__ == "__main__":

    with open('data/preprocessed/normalized_population_BikeCHI.pkl', 'rb') as file:
        obj_list = pickle.load(file)
    
    metrics = {}

    mean_distrubution, std = get_indegree_metric(obj_list)    
    metrics["indegree"] = {"mean": mean_distrubution, "std": std}

    mean_distrubution, std = get_outdegree_metric(obj_list)
    metrics["outdegree"] = {"mean": mean_distrubution, "std": std}

    mean_distrubution, std = get_degree_metric(obj_list)
    metrics["degree"] = {"mean": mean_distrubution, "std": std}

    mean, std = get_topo_metric(obj_list)
    metrics["topo"] = {"mean": mean, "std": std}

    mean, std = get_weights_metric(obj_list)
    metrics["weights"] = {"mean": mean, "std": std}

    with open('data/metrics.json', 'w') as json_file:
        json.dump(metrics, json_file, indent=4)

    get_weights_metric(obj_list)
    pass

