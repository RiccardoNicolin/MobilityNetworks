import pickle
import networkx as nx
import numpy as np

if __name__ == "__main__":


    with open('data/preprocessed/normalized_population_BikeCHI.pkl', 'rb') as file:
        obj_list = pickle.load(file)
    
    means = []
    stds = []
    for i,adj in enumerate(obj_list):

        graph = nx.from_numpy_array(adj, create_using=nx.DiGraph)
        indegree = [val for (node, val) in graph.in_degree(weight = "weight")]
        means.append(np.mean(indegree))
        stds.append(np.std(indegree))
    pass

