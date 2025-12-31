import numpy as np

NEIGHBOR_KERNEL = np.array([
    [1, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
], dtype=np.int8)

# Multipliers list for noise generation
MULTIPLIERS = [2, 4, 8, 8, 16]

# Number of tiles generated
TILES = 20

# Amplitude of the rendered noise
AMPLITUDE = 64

# Pattern for noise Stacking
PATTERN = [0.4, 0.3, 0.15, 0.1, 0.05]

# Dimensions of the simulation grid
GRID_SIZE = 64

# Initial number of 'active' cells (population seeds)
INITIAL_SEEDS = 5

# Number of iterations to run the cellular automaton update
UPDATE_ITERATIONS = 25

# The probability factor for a cell to become 'active' based on neighbors
NEIGHBOR_ACTIVATION_FACTOR = 0.11

# Unsharp Mask Percent
UNSHARP_PERCENT = 100

# Final noise output size
RESIZE = 512

# Enhancement contrast factor
CONTRAST_FACTOR = 3.0

# Enabling file saving
SAVE = False