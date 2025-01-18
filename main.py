import pickle
import json
import os, sys

from collections import Counter
from random import random, seed, uniform
from numpy import mean 
import numpy as np
sys.path.append("graph_evolution")

from nsga import nsga_tournament
from organism import Organism
from nsga import fast_non_dominated_sort, nsga_distance_assignment
from plot_utils import get_perfect_pop


def run(config):
    num_generations = config["num_generations"]
    popsize = config["popsize"]
    objectives = config["eval_funcs"]
    track_diversity_over = config["track_diversity_over"]
    tracking_frequency = config["tracking_frequency"]
    network_size = config["network_size"]
    weight_range = config["weight_range"]
    crossover_rate = config["crossover_rate"]
    crossover_odds = config["crossover_odds"]
    mutation_rate = config["mutation_rate"]
    mutation_odds = config["mutation_odds"]
    tournament_probability = config["tournament_probability"]
    population_file = config["population_file"]

    with open(population_file, 'rb') as f:
        initial_population = pickle.load(f)
    genomes_available = len(initial_population)

    fitnessLog = {funcName:[] for funcName in objectives}
    diversityLog = {o:[] for o in track_diversity_over}
    if tracking_frequency == 0:
        tracking_frequency = num_generations

    age_gap = config["age_gap"]
    age_progression = [age_gap*x**2 for x in range(1, 11)]
    age_layers = [[]]
    if num_generations > age_progression[-1]:
        age_progression.append(num_generations)

    for gen in range(num_generations+1):
        #add new age layer if it is time
        if (gen == age_progression[len(age_layers)-1]):
            parents = nsga_tournament(age_layers[-1], 2*popsize, tournament_probability)
            children = [parents[i].makeCrossedCopyWith(
                        parents[i+popsize], crossover_rate, crossover_odds, gen).makeMutatedCopy(
                        mutation_rate, mutation_odds) for i in range(popsize)]
            for name, target in objectives.items():
                _ = [org.getError(name, target) for org in children]
            age_layers.append(children)

        #initialize / reset layer 0
        if gen % age_gap == 0:
            #! Initialize the population with the initial population saved in the data folder
            population = []
            inserted = 0
            while genomes_available > 0 and inserted < popsize:

                population.append(Organism(network_size, uniform(0, np.max(initial_population[genomes_available-1])), weight_range, genome=initial_population[genomes_available-1]))
                inserted += 1
                genomes_available -= 1

            if inserted < popsize:
                raise ValueError("Not enough genomes available to fill population")

            for name, target in objectives.items():
                _ = [org.getError(name, target) for org in population]
            F = fast_non_dominated_sort(population)
            _ = [nsga_distance_assignment(F[f]) for f in F]
            age_layers[0] = population

        #iterate over layers from youngest to oldest
        for l in range(len(age_layers)):
            layer_l = age_layers[l]
            max_age = age_progression[l]

            #produce offspring from layer l and l-1 (if l>0)
            if l > 0:
                R = layer_l + age_layers[l-1]
                F = fast_non_dominated_sort(R)
                _ = [nsga_distance_assignment(F[f]) for f in F]
            else:
                R = layer_l
                age_migrants_in = []
            parents = nsga_tournament(R, 2*popsize, tournament_probability)
            children = [parents[i].makeCrossedCopyWith(
                        parents[i+popsize], crossover_rate, crossover_odds, gen).makeMutatedCopy(
                        mutation_rate, mutation_odds) for i in range(popsize)]
            for name, target in objectives.items():
                _ = [org.getError(name, target) for org in children]

            #move organisms that have aged out to next layer
            layer_candidates = layer_l + children
            if l < len(age_layers)-1: #corrected for index 0 index 1 mismatch
                age_migrants_out = [org for org in layer_candidates if org.age > max_age]
                layer_candidates = [org for org in layer_candidates if org.age <= max_age]
            else:
                age_migrants_out = []

            #selection
            R = layer_candidates + age_migrants_in
            if len(R) < popsize:
                padding = [Organism(network_size, random(), weight_range) for _ in range(popsize-len(R))]
                for name, target in objectives.items():
                    _ = [org.getError(name, target) for org in padding]
                R.extend(padding)
            F = fast_non_dominated_sort(R)
            if len(R) == popsize:
                _ = [nsga_distance_assignment(F[f]) for f in F]
                P = R
            else:
                P = []
                i = 1
                while len(P) + len(F[i]) <= popsize:
                    nsga_distance_assignment(F[i])
                    P.extend(F[i])
                    i += 1
                if len(P) < popsize:
                    nsga_distance_assignment(F[i])
                    F[i].sort(key=lambda org: org.nsga_distance, reverse=True)
                    P.extend(F[i][:popsize-len(P)])
            
            age_layers[l] = P
            age_migrants_in = age_migrants_out

        #evaluation
        if gen % tracking_frequency == 0:
            print("Generation", gen)
            oldest_layer = age_layers[-1]
            for name, target in objectives.items():
                popFitnesses = [org.getError(name, target) for org in oldest_layer]
                fitnessLog[name].append(mean(popFitnesses))
            for name in track_diversity_over:
                spread = len(Counter([org.getProperty(name) for org in oldest_layer]))
                diversityLog[name].append(spread)

    return oldest_layer, fitnessLog, diversityLog

def run_rep(i, save_loc, config):
    seed(i)
    save_loc_i = "{}/{}".format(save_loc, i)
    if not os.path.exists(save_loc_i):
        os.makedirs(save_loc_i)

    objectives = config["eval_funcs"]
    final_pop, fitness_log, diversity_log = run(config)
    perfect_pop = get_perfect_pop(final_pop, objectives)

    if config["save_data"] == 1:
        with open("{}/final_pop.pkl".format(save_loc_i), "wb") as f:
            pickle.dump(final_pop, f)

def main(config):
    save_loc = "{}/{}".format(config["data_dir"], config["name"])
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)

    config_path = "{}/config.json".format(save_loc)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    for i in range(config["reps"]):
        run_rep(i, save_loc, config)

if __name__ == "__main__":
    try:
        config_file = sys.argv[1]
        config = json.load(open(config_file))
    except:
        print("Please give a valid config json to read parameters from.")
        exit()
    
    if len(sys.argv) == 2:
        main(config)