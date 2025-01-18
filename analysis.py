import pickle
import sys

sys.path.append("graph_evolution")
from graph_evolution.organism import Organism
# from organism import Organism
from random import random,uniform
import numpy as np
from PIL import Image

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

if __name__ == "__main__":
    
    with open('data/sample/0/final_pop.pkl', 'rb') as file:
        obj_list = pickle.load(file)

    for i,oranism in enumerate(obj_list[:5]):
        # oranism.saveGraphFigure(f'data/sample/0/organismo_{i}.png')
        array_to_greyscale_image(oranism.adjacencyMatrix, f'data/sample/0/organismo_{i}.png')
    # print(obj_list)

    # npy_obj = np.load("GAN-flow/adj/BikeCHI/2018-01-01.npy")
    # npy_obj = npy_obj.astype(np.float64)
    # npy_obj /= 327.0

    # array_to_greyscale_image(npy_obj, "test/original.png")

    # org = Organism(64, uniform(0, np.max(npy_obj)), [0, 327], npy_obj)
    
    # array_to_greyscale_image(org.adjacencyMatrix, "test/rescaled.png")