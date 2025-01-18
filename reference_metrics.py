import pickle
import networkx as nx
import numpy as np

if __name__ == "__main__":

    with open('data/preprocessed/normalized_population_BikeCHI.pkl', 'rb') as file:
        obj_list = pickle.load(file)
    
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
    print(mean_distrubution)
    print(std)
    pass

