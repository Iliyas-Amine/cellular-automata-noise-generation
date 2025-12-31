import numpy as np
from scipy.signal import convolve2d
from utils.config import (GRID_SIZE, INITIAL_SEEDS, UPDATE_ITERATIONS, NEIGHBOR_ACTIVATION_FACTOR, NEIGHBOR_KERNEL)

def _update_matrix_vectorized(matrix):
    # 1. Calculate sum of neighbors for EVERY cell at once using convolution
    # mode='same' keeps the grid size 64x64
    neighbor_counts = convolve2d(matrix, NEIGHBOR_KERNEL, mode='same', boundary='fill', fillvalue=0)
    
    # 2. Generate random values for the whole grid at once
    random_grid = np.random.uniform(0, 1, (GRID_SIZE, GRID_SIZE))
    
    # 3. Create a mask for where growth should happen
    # Logic: If cell is empty (0) AND random check passes
    growth_mask = (matrix == 0) & (random_grid < (neighbor_counts * NEIGHBOR_ACTIVATION_FACTOR))
    
    # 4. Apply the growth
    # We use bitwise OR to turn the 0s into 1s where the mask is True
    new_matrix = matrix | growth_mask.astype(np.int8)
    
    return new_matrix

def _gen_pop():
    # Initialize grid
    env = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
    
    # Place seeds (Vectorized random choice)
    # We pick flat indices and unravel them to x,y coordinates
    flat_indices = np.random.choice(GRID_SIZE * GRID_SIZE, INITIAL_SEEDS, replace=False)
    x_coords, y_coords = np.unravel_index(flat_indices, (GRID_SIZE, GRID_SIZE))
    env[x_coords, y_coords] = 1
    
    # Run iterations
    for _ in range(UPDATE_ITERATIONS):
        env = _update_matrix_vectorized(env)
        
    return env