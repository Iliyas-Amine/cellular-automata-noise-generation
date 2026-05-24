import numpy as np
from numpy.typing import NDArray
from typing import Tuple, List

# ==========================================
# KERNEL DEFINITIONS
# ==========================================

# _update_batch - Neighbors Kernel:
# A 3x3 Moore neighborhood matrix used to count active neighbors in the Cellular Automata.
# Center is 0 because a cell doesn't count itself as a neighbor.
NEIGHBOR_KERNEL: NDArray[np.int8] = np.array([[
    [1, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
]] , dtype=np.int8)

# batch_tilemap - Gaussian Blur Kernel:
# Large kernel (31x31) to smooth the binary CA grids into soft heightmaps.
KERNEL_01: Tuple[int, int] = (31, 31)

# _enhance - Box Blur Kernel:
# Used for general noise smoothing before contrast adjustment.
KERNEL_02: Tuple[int, int] = (15, 15)

# _enhance - (Unsharp) Box Blur Kernel:
# Used to create a blurred mask for the Unsharp Masking technique (sharpening).
KERNEL_03: Tuple[int, int] = (11, 11)

# stacking - (Threshold) Box Blur Kernel:
# Applied only to areas that fall below the THRESHOLD value (deep valleys/oceans).
KERNEL_04: Tuple[int, int] = (31, 31)

# stacking - Box Blur Kernel:
# A final, subtle pass to smooth out any artifacts from the stacking process.
KERNEL_05: Tuple[int, int] = (3, 3)

# ==========================================
# RNG DEFINITIONS
# ==========================================

# Master seed for deterministic terrain generation
SEED: str = str(1234567890123456).zfill(16)

# _gen_pop_batch - Automaton Seed Positions:
POS_SEED: int = int(SEED[:4])
POS_RNG = np.random.default_rng(seed=POS_SEED)

# _update_batch - Automaton Probability Grid:
PROB_SEED: int = int(SEED[4:8])
PROB_RNG = np.random.default_rng(seed=PROB_SEED)

# batch_tilemap - Base White Noise Matrix:
SMOO_SEED: int = int(SEED[8:12])
SMOO_RNG = np.random.default_rng(seed=SMOO_SEED)

# _stitch - Order of Tile Stitching:
STIT_SEED: int = int(SEED[12:])
STIT_RNG = np.random.default_rng(seed=STIT_SEED)

# ==========================================
# GENERATION SETTINGS
# ==========================================

# Number of individual source tiles to generate via Cellular Automata
TILES: int = 20

# Dimensions of the individual simulation grids (e.g., 64x64)
GRID_SIZE: int = 64

# Initial number of 'active' cells (population seeds) scattered across all tiles
INITIAL_SEEDS: int = 5 * TILES

# Number of generations to run the cellular automaton rules
UPDATE_ITERATIONS: int = 25

# The probability threshold: higher values make it harder for empty cells to become active
NEIGHBOR_ACTIVATION_FACTOR: float = 0.11

# The overlap during tile stitching
OVERLAP = 8

# Used to calculate the overall size of a stitched matrix
STRIDE = GRID_SIZE - OVERLAP

# stacking - Threshold Value for Blur:
# Noise values below this cut-off are treated as "lowlands" or "water" and smoothed differently
THRESHOLD: float = -0.55

# Multipliers list for noise generation
# Defines the scale (frequency) of each noise layer. 
# [2, 4, 8...] acts like Octaves in Perlin noise.
MULTIPLIERS: List[int] = [2, 4, 8, 16, 64]

# Amplitude of the rendered noise
# Vertical scaler for the final 3D mesh points (Z-axis magnitude)
AMPLITUDE: int = 48

# Weights for noise Stacking
# Determines how much influence each layer (defined by MULTIPLIERS) has on the final map.
# High influence for low-frequency (base) layers, low influence for high-frequency (detail) layers.
WEIGHTS: NDArray[np.float64] = np.array([0.50, 0.25, 0.13, 0.09, 0.03])

# Unsharp Mask Percent
# Strength of the edge enhancement filter (100 = strong sharpening)
UNSHARP_PERCENT: int = 45

# Final noise output size
# The resolution of the final heightmap passed to the mesher
RESIZE: int = 512

# Enhancement contrast factor
# Stretches the noise values to utilize the full -1.0 to 1.0 range
CONTRAST_FACTOR: float = 3.0

# Enabling file saving
# If True, intermediate noise maps and the final VTK mesh are written to disk
SAVE: bool = True
