import statistics

import networkx as nx
import numpy as np


class Evaluation:
    #node-level topological properties
    def in_degree_distribution(self, network) -> list[float]:
        networkx_obj = network.getNetworkxObject()
        num_nodes = network.numNodes
        degree_sequence = list(d for _, d in networkx_obj.in_degree())
        freq = [0]*(num_nodes+1)
        for d in degree_sequence:
            freq[d] += 1
        return [x/num_nodes for x in freq]    

    def out_degree_distribution(self, network) -> list[float]:
        networkx_obj = network.getNetworkxObject()
        num_nodes = network.numNodes
        degree_sequence = list(d for _, d in networkx_obj.out_degree())
        freq = [0]*(num_nodes+1)
        for d in degree_sequence:
            freq[d] += 1
        return [x/num_nodes for x in freq]
    
    def degree_distribution(self, network) -> list[float]:
        networkx_obj = network.getNetworkxObject()
        num_nodes = network.numNodes
        degree_sequence = list(d for _, d in networkx_obj.degree())
        freq = [0]*(2*num_nodes+1)
        for d in degree_sequence:
            freq[d] += 1
        return [x/num_nodes for x in freq]
    
    def topology(self, network) -> tuple[float, float]:

        networkx_obj = network.getNetworkxObject()
        cl = list(nx.clustering(networkx_obj, weight='weight').values())
        mean = np.mean(cl)
        std = np.std(cl)

        return mean, std
    
    def weights(self, network):

        weights = np.array(network.adjacencyMatrix).flatten()
        mean = np.mean(weights)
        std = np.std(weights)

        return mean, std

    def flux_distribution(self, network) -> list[float]:
        """
        This function evaluates the flux distribution for a given network and 
        is invariant wrt to the time.
        e.g. the flux distribution of Chicago at two given times should be similar.
        """
        networkx_obj = network.getNetworkxObject()
        weights = [networkx_obj[i][j]['weight'] for i, j in networkx_obj.edges()]
        normalised_weights = [w/np.max(weights) for w in weights]
        
        n_bins = 10
        hist, bins = np.histogram(normalised_weights, bins=n_bins, density=True)
        
        return hist.tolist()
        

    def avg_shortest_path_length_distribution(self, network) -> list[float]:
        weight = 1/network.numNodes**2
        shortest_path = dict(nx.shortest_path_length(network.getNetworkxObject()))
        avg_shortest = sorted([weight*np.mean(list(shortest_path[i].values())) for i in range(len(shortest_path))], reverse=True)
        
        return avg_shortest
    
    def betweenness_distribution(self, network) -> list[float]:
        networkx_obj = network.getNetworkxObject()
        return sorted(list(nx.betweenness_centrality(networkx_obj).values()), reverse=True)  
    
    #topological properties
    def connectance(self, network) -> float:
        return network.numInteractions / network.numNodes**2

    def clustering_coefficient(self, network) -> float:
        return round(nx.average_clustering(network.getNetworkxObject()), 3)    

    def transitivity(self, network) -> float:
        return round(nx.transitivity(network.getNetworkxObject()), 3)

    def proportion_of_self_loops(self, network) -> float:
        return sum([1 for i in range(network.numNodes) if network.adjacencyMatrix[i][i] != 0]) / network.numNodes    

    def diameter(self, network) -> int:
        shortest_path = dict(nx.shortest_path_length(network.getNetworkxObject()))
        return max([max(shortest_path[i].values()) for i in range(len(shortest_path))])    

    def number_of_modules(self, network) -> int:
        if network.numInteractions > 0:
            return len(nx.community.greedy_modularity_communities(network.getNetworkxObject()))
        else:
            return 0

    #interaction strength properties
    def positive_interactions_proportion(self, network) -> float:
        num_interactions = network.numInteractions
        return network.numPositive / num_interactions if num_interactions > 0 else 0

    def average_positive_interactions_strength(self, network) -> float:
        sum_pos = sum([sum([y for y in x if y > 0]) for x in network.adjacencyMatrix])
        num_pos = network.numPositive
        return round(sum_pos/num_pos, 3) if num_pos > 0 else 0
   

    def average_negative_interactions_strength(self, network) -> float:
        sum_neg = sum([sum([y for y in x if y < 0]) for x in network.adjacencyMatrix])
        num_neg = network.numNegative
        return round(sum_neg/num_neg, 3) if num_neg > 0 else 0    

    def variance_positive_interactions_strength(self, network) -> float:
        flat = [x for y in network.adjacencyMatrix for x in y if x > 0]
        if len(flat) < 2:
            return 0
        var_pos = statistics.variance(flat)
        return round(var_pos, 3)    
    
    def variance_negative_interactions_strength(self, network) -> float:
        flat = [x for y in network.adjacencyMatrix for x in y if x < 0]
        if len(flat) < 2:
            return 0
        var_neg = statistics.variance(flat)
        return round(var_neg, 3)

    def proportion_of_self_loops_positive(self, network) -> float:
        return sum([1 for i in range(network.numNodes) if network.adjacencyMatrix[i][i] > 0]) / network.numNodes    

    #motifs
    def proportion_of_mutualistic_pairs(self, network) -> float:
        adj = network.adjacencyMatrix
        nn = network.numNodes
        count_pairs = sum([sum([1 for j in range(i+1, nn) if adj[i][j] > 0 and adj[j][i] > 0]) for i in range(nn)])
        possible_pairs = ((nn)*(nn-1))/2
        return count_pairs / possible_pairs

    def proportion_of_competition_pairs(self, network) -> float:
        adj = network.adjacencyMatrix
        nn = network.numNodes
        count_pairs = sum([sum([1 for j in range(i+1, nn) if adj[i][j] < 0 and adj[j][i] < 0]) for i in range(nn)])
        possible_pairs = ((nn)*(nn-1))/2
        return count_pairs / possible_pairs

    def proportion_of_parasitism_pairs(self, network) -> float:
        adj = network.adjacencyMatrix
        nn = network.numNodes
        count_pairs = sum([sum([1 for j in range(i+1, nn) if (adj[i][j] < 0 and adj[j][i] > 0) or (adj[i][j] > 0 and adj[j][i] < 0)]) for i in range(nn)])
        possible_pairs = ((nn)*(nn-1))/2
        return count_pairs / possible_pairs    

#all callable evaluation functions
functions = {funcName:getattr(Evaluation, funcName) for funcName in dir(Evaluation) 
                if callable(getattr(Evaluation, funcName)) and not funcName.startswith("__")}

#non-distribution callables
properties = {funcName:getattr(Evaluation, funcName) for funcName in dir(Evaluation) 
                if callable(getattr(Evaluation, funcName)) and not funcName.startswith("__") and
                not funcName.endswith("_distribution")}

#distribution callables
distributions = {funcName:getattr(Evaluation, funcName) for funcName in dir(Evaluation) 
                if callable(getattr(Evaluation, funcName)) and not funcName.startswith("__") and
                funcName.endswith("_distribution")}


if __name__ == "__main__":
    e = Evaluation()
    print(e.functions)



#ADDED BY NICOLIN RICCARDO

    def CPC_evaluation(self, network) -> tuple[float, float]:
        """
        This function evaluates the CPC (Common part of commuter), also known as Sørensen-Dice index,
        that range between 0 and 1 (1 being a perfect match between a generated network and the ground truth - in our case a random matrix in the original dataset).
        It is calculate from a 
        """
        networkx_obj = network.getNetworkxObject()
        cpc = 0


        return cpc, cmc