import numpy as np
import random
import cv2
import os
from typing import List

from numpy.typing import NDArray

from utils.config import (
    SAVE, TILES, GRID_SIZE, KERNEL_01, KERNEL_02, KERNEL_03, 
    KERNEL_04, KERNEL_05, CONTRAST_FACTOR, UNSHARP_PERCENT, 
    RESIZE, MULTIPLIERS, THRESHOLD, WEIGHTS, AMPLITUDE
)

def save_noise_image(data: NDArray[np.floating], folder: str, prefix: str) -> None:
    """
    Normalizes and saves a floating-point noise array as a PNG image.

    This utility function scales the input noise data (expected to be roughly in 
    the range -1.0 to 1.0) to a 0-255 uint8 range and saves it to the specified 
    folder. It is used primarily for visual debugging of the intermediate generation steps.

    Args:
        data (NDArray[np.floating]): The 2D noise array to save.
        folder (str): The target directory for the image file.
        prefix (str): A prefix for the filename (e.g., 'tile', 'noise').
    """
    if SAVE:
        os.makedirs(folder, exist_ok=True)
        # Normalize the float data (-1.0 to 1.0) to 0.0 to 2.0 range
        clipped: NDArray[np.floating] = np.clip(data, -1.0, 1.0) 
        shifted: NDArray[np.floating] = clipped + 1.0 
        
        # Scale to 0-255 for standard 8-bit image format
        norm_img: NDArray[np.uint8] = (shifted * 127.5).astype(np.uint8) 
        
        filename: str = os.path.join(folder, f"{prefix}_{os.urandom(4).hex()}.png") 
        cv2.imwrite(filename, norm_img)

def batch_tilemap(matrices: NDArray[np.int8]) -> List[NDArray[np.floating]]:
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
    matrices_arr: NDArray[np.int8] = np.array(matrices)
    # Generate base white noise
    noise_batch: NDArray[np.floating] = np.random.uniform(0.01, 1, size=(TILES, GRID_SIZE, GRID_SIZE))

    mask_above: NDArray[np.bool_] = (matrices_arr == 1)

    # Apply noise selectively based on the cellular automata mask
    # Cells that are '1' get positive noise (mountains), '0' get negative noise (valleys)
    noise_batch = np.where(mask_above, noise_batch, -noise_batch)

    smoothed_batch: List[NDArray[np.floating]] = []
    for i in range(TILES):
        # Apply a strong Gaussian blur to smooth the sharp white noise into rolling terrain
        blurred: NDArray[np.floating] = cv2.GaussianBlur(noise_batch[i], KERNEL_01, 0)
        smoothed_batch.append(blurred)
        
        save_noise_image(blurred, "tiles", "tile")
    
    return smoothed_batch

def _stitch(multiplier: int, tiles: NDArray[np.floating]) -> NDArray[np.float32]:
    """
    Creates a large noise grid by stitching together random base tiles.

    This function expands the grid size by the given multiplier and fills it 
    by randomly sampling from the provided list of base tiles. This technique 
    allows for infinite texture generation from a finite set of source patterns.

    Args:
        multiplier (int): The factor by which to multiply the base grid size 
            (e.g., a multiplier of 2 creates a 2x2 grid of tiles).
        tiles (NDArray[np.floating]): The pool of base tiles to sample from.

    Returns:
        NDArray[np.float32]: A single large 2D grid composed of stitched tiles.
    """
    # Calculate the size of the new combined grid (e.g., if multiplier is 2, size is 2x64 = 128)
    current_size: int = multiplier * GRID_SIZE
    noise_grid: NDArray[np.float32] = np.zeros((current_size, current_size), dtype=np.float32)

    total_tiles: int = multiplier * multiplier
    
    # Randomly sample tiles to fill the grid
    # We sample with replacement to create a larger texture from our limited tile set
    selected_tiles: List[NDArray[np.floating]] = random.choices(list(tiles), k=total_tiles)

    # Paste the selected tiles into the large grid row by row
    for idx, tile in enumerate(selected_tiles):
        row: int = idx // multiplier
        col: int = idx % multiplier
        
        y_start: int = row * GRID_SIZE
        y_end: int = y_start + GRID_SIZE
        x_start: int = col * GRID_SIZE
        x_end: int = x_start + GRID_SIZE
        
        noise_grid[y_start:y_end, x_start:x_end] = tile
    
    return noise_grid

def _enhance(noise_grid: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    Refines a raw noise grid by applying contrast and sharpening.

    This processing step applies a box blur to blend tile seams, increases the 
    contrast to utilize the full dynamic range, and optionally applies an 
    Unsharp Mask filter to highlight ridge lines and details.

    Args:
        noise_grid (NDArray[np.float32]): The raw stitched noise grid.

    Returns:
        NDArray[np.float32]: The enhanced and sharpened noise grid.
    """
    # Initial blur to blend the seams between stitched tiles
    noise_grid = cv2.blur(noise_grid, KERNEL_02)

    # Increase contrast to make the terrain more dramatic
    noise_grid *= CONTRAST_FACTOR
    np.clip(noise_grid, -1.0, 1.0, out=noise_grid)

    amount: float = UNSHARP_PERCENT / 100.0 if UNSHARP_PERCENT > 1 else float(UNSHARP_PERCENT)
    
    # Apply Unsharp Masking to sharpen details (peaks and ridges)
    if amount != 0:
        blurred_mask: NDArray[np.float32] = cv2.blur(noise_grid, KERNEL_03)
        cv2.addWeighted(
            noise_grid, 1.0 + amount, 
            blurred_mask, -amount, 
            0, 
            dst=noise_grid
        )
        np.clip(noise_grid, -1.0, 1.0, out=noise_grid)
    
    return noise_grid
    
def _resize(noise_grid: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    Resizes the noise grid to the final target resolution.

    Ensures that all noise layers, regardless of their original 'multiplier' 
    scale, match the final dimensions required for stacking.

    Args:
        noise_grid (NDArray[np.float32]): The input noise grid of arbitrary size.

    Returns:
        NDArray[np.float32]: The grid resized to the global RESIZE dimension.
    """
    target_size: int = int(RESIZE)
    # Upscale or downscale the grid to the final desired resolution
    if noise_grid.shape[0] != target_size:
        noise_grid = cv2.resize(
            noise_grid, 
            (target_size, target_size), 
            interpolation=cv2.INTER_LINEAR
        )
    return noise_grid

def _join_tiles(multiplier: int, tiles: NDArray[np.floating]) -> NDArray[np.float32]:
    """
    Executes the full pipeline to create a single noise layer at a specific scale.

    This helper function orchestrates the stitching, enhancing, and resizing 
    steps to produce one coherent noise layer (frequency) from the base tiles.

    Args:
        multiplier (int): The scale factor for this specific noise layer.
        tiles (NDArray[np.floating]): The source tiles to use for generation.

    Returns:
        NDArray[np.float32]: A fully processed 2D noise layer ready for stacking.
    """
    # Pipeline: Stitch small tiles -> Enhance/Sharpen -> Resize to target resolution
    noise_grid: NDArray[np.float32] = _stitch(multiplier, tiles)
    noise_grid = _enhance(noise_grid)
    noise_grid = _resize(noise_grid)

    save_noise_image(noise_grid, "noises", f"noise_g{multiplier}")

    return noise_grid

def gen_noises(tiles: List[NDArray[np.floating]]) -> List[NDArray[np.float32]]:
    """
    Generates multiple noise layers at different frequencies (octaves).

    Iterates through the configured MULTIPLIERS list to create a collection of 
    noise maps, each representing terrain features at different scales (e.g., 
    large continents vs. small hills).

    Args:
        tiles (List[NDArray[np.floating]]): The base terrain tiles.

    Returns:
        List[NDArray[np.float32]]: A list of noise layers, all resized to the 
        same final resolution.
    """
    tiles_array: NDArray[np.floating] = np.array(tiles) 

    # Generate multiple noise layers at different frequencies (scales)
    noises: List[NDArray[np.float32]] = []
    for multiplier in MULTIPLIERS:
        noise_grid: NDArray[np.float32] = _join_tiles(multiplier, tiles_array)
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
    noise_stack: NDArray[np.float32] = np.array(noises, dtype=np.float32)
    weights: NDArray[np.float32] = np.asarray(WEIGHTS, dtype=np.float32)

    # Perform weighted sum (Dot product) to combine all noise layers
    # This creates the fractal "Perlin-like" effect where some layers provide large shapes and others provide detail
    noise_sum: NDArray[np.float32] = np.tensordot(weights, noise_stack, axes=1)

    # Identify "water" or "lowland" areas
    mask: NDArray[np.bool_] = (noise_sum <= THRESHOLD)
    
    if mask.any():
        # Apply extra smoothing to low areas to simulate sediment or water
        blurred_section: NDArray[np.float32] = cv2.blur(noise_sum, KERNEL_04)
        noise_sum[mask] = blurred_section[mask]

    # Final overall smooth to remove any remaining artifacts
    cv2.blur(noise_sum, KERNEL_05, dst=noise_sum)

    save_noise_image(noise_sum, "fnoises", "noise_f")

    # Scale the normalized heightmap to the final physical height (Amplitude)
    return noise_sum * AMPLITUDE