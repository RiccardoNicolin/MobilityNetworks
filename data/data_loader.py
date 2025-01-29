import os
import numpy as np
import pickle
import sys

def load_npy_files_from_directory(directory_path):
    npy_files = []
    for file_name in os.listdir(directory_path):
        if file_name.endswith('.npy'):
            file_path = os.path.join(directory_path, file_name)
            npy_obj = np.load(file_path)
            npy_obj = npy_obj.astype(np.float64)
            npy_files.append(npy_obj)
    return npy_files

def get_max_in_population(population):
    max_value = 0.0
    for npy_array in population:
        max_value = max(max_value, np.max(npy_array))
    return max_value

def normalize_population(population):
    max_value = np.float64(get_max_in_population(population))

    for npy_array in population:
        npy_array /= (max_value)
    
    return population


if __name__ == "__main__":
    # preprocessing for the data
    if len(sys.argv) != 2:
        print("Usage: python data_loader.py <dataset_name>")
        sys.exit(1)
    else:
        dataset = sys.argv[1]
        assert dataset in ['BikeCHI', 'BikeNYC', 'TaxiNYC', 'TaxiCHI'], "Invalid dataset name"
    path = f'GAN-flow/adj/{dataset}'
    population = load_npy_files_from_directory(path)
    normalized_population = normalize_population(population)
    with open(f'data/preprocessed/normalized_population_{dataset}.pkl', 'wb') as f:
        pickle.dump(normalized_population, f)