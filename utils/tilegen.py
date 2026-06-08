import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import correlate

from utils.noiseutils import batch_tilemap

def _update_batch(matrices: NDArray[np.int8], config: dict[str, Any]) -> NDArray[np.int8]:
    """
    Performs a single simulation step on a batch of Cellular Automata grids.

    This function calculates neighbor counts for every cell in the batch using 
    convolution, generates a random probability field, and applies a stochastic 
    growth rule. Cells become active if they have sufficient neighbors and 
    satisfy the probability condition defined by NEIGHBOR_ACTIVATION_FACTOR.

    Args:
        matrices (NDArray[np.int8]): A 3D array (Batch, Row, Col) representing 
            the current binary state of the cellular grids (0 for empty, 1 for active).

    Returns:
        NDArray[np.int8]: The updated 3D array after applying the growth rules.
    """
    # Correlate calculates the sum of neighbors for each cell
    # 'mode=constant' and 'cval=0' ensures edges behave as if surrounded by empty space
    neighbor_counts = correlate(
        matrices, 
        config["NEIGHBOR_KERNEL"], 
        mode='wrap'
    )

    # Generate a random probability grid for stochastic growth
    # This ensures that even with identical neighbors, growth patterns differ
    random_grid = config["PROB_RNG"].uniform(0, 1, matrices.shape)
    
    # Determine which cells grow based on neighbors and chance
    # Rule: An empty cell becomes active IF random_val < (neighbors * factor)
    growth_mask = (matrices == 0) & (random_grid < (neighbor_counts * config["NEIGHBOR_ACTIVATION_FACTOR"]))

    # Return the union of old cells and newly grown cells
    return matrices | growth_mask

def _gen_pop_batch(config: dict[str, Any]) -> NDArray[np.int8]:
    """
    Initializes and evolves a fresh batch of cellular automata environments.

    This function creates an empty 3D environment (representing a batch of 2D tiles),
    seeds it with a specific number of random 'alive' cells, and then iterates 
    the simulation for a fixed number of steps (UPDATE_ITERATIONS) to generate 
    organic-looking clusters.

    Returns:
        NDArray[np.int8]: A 3D array containing the final binary states of 
        all generated tiles.
    """
    # Initialize parameters
    TILES = config["TILES"]
    GRID_SIZE = config["GRID_SIZE"]
    INITIAL_SEEDS = config["INITIAL_SEEDS"]
    UPDATE_ITERATIONS = config["UPDATE_ITERATIONS"]
    # Initialize empty 3D environment (Batch of 2D grids)
    # Shape is (TILES, 64, 64) allowing us to process all tiles in parallel
    env = np.zeros((TILES, GRID_SIZE, GRID_SIZE), dtype=np.int8)

    # Select random indices to seed the population
    # We sample from the flattened total size to ensure unique seeds across the batch
    flat_indices = config["POS_RNG"].choice(
        TILES * GRID_SIZE * GRID_SIZE, 
        INITIAL_SEEDS, 
        replace=False
    )
    
    # Convert flat indices to 3D coordinates (Tile Index, Row, Col)
    x, y, z = np.unravel_index(flat_indices, (TILES, GRID_SIZE, GRID_SIZE))
    
    # Set initial seeds to 'Alive' (1)
    env[x, y, z] = 1

    # Run the simulation steps
    for _ in range(UPDATE_ITERATIONS):
        env = _update_batch(env, config)
        
    return env

def gen_tiles(config: dict[str, Any]) -> NDArray[np.floating]:
    """
    Orchestrates the generation of smooth terrain tiles.

    This is the main public entry point for tile generation. It generates the 
    raw binary cellular automata patterns via `_gen_pop_batch` and then processes 
    them into smooth, floating-point heightmaps using `batch_tilemap`.

    Returns:
        List[NDArray[np.floating]]: A list of generated 2D heightmap arrays, 
        where each array represents a seamless terrain tile ready for noise stitching.
    """
    logging.info(f"Generating {config['TILES']} tiles...")
    
    # Create the raw binary (0/1) cellular automata grids
    matrices = _gen_pop_batch(config)
    
    # batch_tilemap takes the int8 binary maps and returns blurred float maps
    # This converts the sharp "islands" into smooth heightmap gradients
    tiles = batch_tilemap(matrices, config)
    
    logging.info(f"All {config['TILES']} tiles generated")
    return tiles
