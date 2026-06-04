import numpy as np
import cv2
import os
from typing import List

from numpy.typing import NDArray

from utils.config import (
    SAVE, TILES, GRID_SIZE, KERNEL_01, KERNEL_02, KERNEL_03, 
    KERNEL_04, KERNEL_05, CONTRAST_FACTOR, 
    RESIZE, MULTIPLIERS, THRESHOLD, WEIGHTS, AMPLITUDE,
    STIT_RNG, SMOO_RNG, SEED, BLEND_PERCENT
)

def save_noise_image(data: NDArray[np.floating], folder: str, filename: str) -> None:
    """
    Normalizes and saves a floating-point noise array as a PNG image.

    This utility function scales the input noise data (expected to be roughly in 
    the range -1.0 to 1.0) to a 0-255 uint8 range and saves it to the specified 
    folder. It is used primarily for visual debugging of the intermediate generation steps.

    Args:
        data (NDArray[np.floating]): The 2D noise array to save.
        folder (str): The target directory for the image file.
        filename (str): A filename for the file.
    """
    if SAVE:
        os.makedirs(folder, exist_ok=True)
        # Normalize the float data (-1.0 to 1.0) to 0.0 to 2.0 range
        clipped = np.clip(data, -1.0, 1.0) 
        shifted = clipped + 1.0 
        
        # Scale to 0-255 for standard 8-bit image format
        norm_img = (shifted * 127.5).astype(np.uint8) 
        
        filename = os.path.join(folder, f"{filename}.png") 
        cv2.imwrite(filename, norm_img)

def batch_tilemap(matrices: NDArray[np.int8]) -> NDArray[np.floating]:
    """
    Converts binary cellular automata grids into smooth, distinct terrain tiles.

    This function takes the raw binary output from the simulation and applies 
    random noise. Areas marked as 'active' (1) receive positive noise values, 
    while 'empty' areas (0) receive negative values. A strong Gaussian blur is 
    then applied to create smooth transitions between these regions.

    Args:
        matrices (NDArray[np.int8]): A 3D array (Batch, Row, Col) of binary 
            cellular automata masks.

    Returns:
        List[NDArray[np.floating]]: A list of 2D arrays, each representing a 
        smoothed heightmap tile.
    """
    # Convert input binary matrices to float for noise processing
    # Generate base white noise
    noise_batch = SMOO_RNG.uniform(0.01, 1, size=(TILES, GRID_SIZE, GRID_SIZE)).astype(np.float32)

    # In-place negation for empty cells avoids allocating a third 3D matrix
    mask_below = (matrices == 0)
    noise_batch[mask_below] *= -1.0

    for i in range(TILES):
        # Apply a strong Gaussian blur to smooth the sharp white noise into rolling terrain
        cv2.GaussianBlur(noise_batch[i], KERNEL_01, 0, dst=noise_batch[i])
        save_noise_image(noise_batch[i], "tiles", f"tile_{SEED}_{i}")
    
    return noise_batch

def _stitch(multiplier: int, tiles: NDArray[np.floating]) -> NDArray[np.float32]:
    """
    Creates a large noise grid by stitching together random base tiles.
    
    This function scales up the terrain by selecting random pre-generated tiles 
    and arranging them into a larger grid format. It is completely vectorized 
    with zero redundant memory allocations.

    Args:
        multiplier (int): The scale factor defining the grid size (e.g., a multiplier 
            of 2 creates a 2x2 grid of tiles).
        tiles (NDArray[np.floating]): A 3D array of the pre-generated smooth terrain tiles.

    Returns:
        NDArray[np.float32]: A 2D array representing the stitched terrain map.
    """
    total_tiles = multiplier * multiplier
    selected_tiles = STIT_RNG.choice(tiles, size=total_tiles, replace=True)

    tensor_4d = selected_tiles.reshape(multiplier, multiplier, GRID_SIZE, GRID_SIZE)
    transposed = tensor_4d.transpose(0, 2, 1, 3)

    return transposed.reshape(multiplier * GRID_SIZE, multiplier * GRID_SIZE)

def _enhance(noise_grid: NDArray[np.float32], scratch_grid: NDArray[np.float32]) -> None:
    """
    Applies in-place smoothing and algebraic soft-clipping to the noise grid.

    To eliminate memory allocation overhead during batch processing, this function 
    performs all matrix operations in-place. It applies a border-replicated box 
    blur to smooth high-frequency artifacts, then normalizes the terrain using 
    a sigmoid-like transfer curve. This boosts mid-tone terrain while softly 
    rolling off extreme peaks and valleys to prevent hard clipping.

    Args:
        noise_grid (NDArray[np.float32]): The primary noise array, mutated in-place.
        scratch_grid (NDArray[np.float32]): A pre-allocated workspace array of the 
            exact same shape, used to hold intermediate squares to avoid RAM churn.
    """

    cv2.blur(noise_grid, KERNEL_02, dst=noise_grid, borderType=cv2.BORDER_REPLICATE)

    np.multiply(noise_grid, noise_grid, out=scratch_grid)
    scratch_grid += CONTRAST_FACTOR/100 
    np.sqrt(scratch_grid, out=scratch_grid)
    noise_grid /= scratch_grid

def _resize(noise_grid: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    Resizes the noise grid to the final target resolution.

    Ensures that all multi-scale noise layers, regardless of their original 
    'multiplier' grid size, are normalized to the final dimensions required 
    for coherent stacking.

    Args:
        noise_grid (NDArray[np.float32]): The input noise grid of arbitrary size.

    Returns:
        NDArray[np.float32]: The grid resized to the global RESIZE dimension using 
        bilinear interpolation.
    """
    # Upscale or downscale the grid to the final desired resolution
    if noise_grid.shape[0] != RESIZE:
        noise_grid = cv2.resize(
            noise_grid, 
            (RESIZE, RESIZE), 
            interpolation=cv2.INTER_LINEAR
        )
    return noise_grid

def _join_tiles(multiplier: int, tiles: NDArray[np.floating], scratch_grid: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    Executes the split pipeline for a single frequency layer of noise.

    This acts as a coordinator for an individual octave layer. It stitches tiles at 
    their native scale, applies initial smoothing, resizes to the master canvas 
    resolution, and finally enhances the contrast.

    Args:
        multiplier (int): The scale factor (frequency) for this specific noise layer.
        tiles (NDArray[np.floating]): The base 3D array of smooth terrain tiles.
        scratch_grid (NDArray[np.float32]): A pre-allocated workspace array for in-place math.

    Returns:
        NDArray[np.float32]: A single, fully processed 2D octave layer ready for stacking.
    """
    noise_grid = _stitch(multiplier, tiles)

    cv2.blur(noise_grid, KERNEL_02, dst=noise_grid, borderType=cv2.BORDER_REPLICATE)

    noise_grid = _resize(noise_grid)

    _enhance(noise_grid, scratch_grid)

    save_noise_image(noise_grid, "noises", f"noise_g{multiplier}_{SEED}")

    return noise_grid

def gen_noises(tiles: NDArray[np.floating]) -> List[NDArray[np.float32]]:
    """
    Generates multiple noise layers at different frequencies (octaves).

    Orchestrates the creation of all individual octave layers defined in the 
    MULTIPLIERS configuration. It pre-allocates a single workspace to prevent 
    RAM churn under load, executing the `_join_tiles` pipeline for each scale.

    Args:
        tiles (List[NDArray[np.floating]]): The list of generated 2D heightmap base tiles.

    Returns:
        List[NDArray[np.float32]]: A list of processed 2D arrays, representing 
        each octave layer needed for the final stacking process.
    """ 
    # Pre-allocate a single master workspace
    scratch_grid = np.empty((RESIZE, RESIZE), dtype=np.float32)

    noises: List[NDArray[np.float32]] = []
    for multiplier in MULTIPLIERS:
        noise_grid = _join_tiles(multiplier, tiles, scratch_grid)
        noises.append(noise_grid)
    return noises

def stacking(noises: List[NDArray[np.float32]]) -> NDArray[np.float32]:
    """
    Combines multiple noise layers into a final heightmap.

    This function calculates the weighted sum of all input noise layers (similar 
    to fractal noise generation). It also performs post-processing on specific 
    regions, such as smoothing 'underwater' areas (values below THRESHOLD), 
    before scaling the result by the global AMPLITUDE.

    Args:
        noises (List[NDArray[np.float32]]): The list of multi-scale noise layers.

    Returns:
        NDArray[np.float32]: The final, single-layer 2D heightmap ready for meshing.
    """
    noise_stack = np.array(noises, dtype=np.float32)
    weights = np.asarray(WEIGHTS, dtype=np.float32)

    # Perform weighted sum (Dot product) to combine all noise layers
    # This creates the fractal "Perlin-like" effect where some layers provide large shapes and others provide detail
    noise_sum = np.tensordot(weights, noise_stack, axes=1)

    # Identify "water" or "lowland" areas and map them to 1.0 and 0.0
    mask = (noise_sum <= THRESHOLD).astype(np.float32)
    
    if mask.any():
        # Soften the hard binary edges by blurring the mask itself
        smooth_alpha = cv2.GaussianBlur(mask, KERNEL_03, 0)
        # Apply extra smoothing to low areas to simulate sediment or water
        blurred_section: NDArray[np.float32] = cv2.blur(noise_sum, KERNEL_04)
        # Linearly blend across the entire map using the alpha channel
        blend_factor = smooth_alpha * BLEND_PERCENT
        noise_sum = (blurred_section * blend_factor) + (noise_sum * (1.0 - blend_factor))
    # Final overall smooth to remove any remaining artifacts
    cv2.blur(noise_sum, KERNEL_05, dst=noise_sum)

    save_noise_image(noise_sum, "fnoises", f"noise_{SEED}")

    # Scale the normalized heightmap to the final physical height (Amplitude)
    noise_sum *= float(AMPLITUDE)
    return noise_sum
