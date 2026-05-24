import numpy as np
import random
import cv2
import os, time, logging
from typing import List

from numpy.typing import NDArray

from utils.config import (
    SAVE, TILES, GRID_SIZE, KERNEL_01, KERNEL_02, KERNEL_03, 
    KERNEL_04, KERNEL_05, CONTRAST_FACTOR, UNSHARP_PERCENT, 
    RESIZE, MULTIPLIERS, THRESHOLD, WEIGHTS, AMPLITUDE,
    STIT_RNG, SMOO_RNG, SEED, STRIDE, OVERLAP
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
        clipped: NDArray[np.floating] = np.clip(data, -1.0, 1.0) 
        shifted: NDArray[np.floating] = clipped + 1.0 
        
        # Scale to 0-255 for standard 8-bit image format
        norm_img: NDArray[np.uint8] = (shifted * 127.5).astype(np.uint8) 
        
        filename: str = os.path.join(folder, f"{filename}.png") 
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
    noise_batch: NDArray[np.float32] = SMOO_RNG.uniform(0.01, 1, size=(TILES, GRID_SIZE, GRID_SIZE)).astype(np.float32)

    mask_above: NDArray[np.bool_] = (matrices_arr == 1)

    # Apply noise selectively based on the cellular automata mask
    # Cells that are '1' get positive noise (mountains), '0' get negative noise (valleys)
    noise_batch = np.where(mask_above, noise_batch, -noise_batch)

    smoothed_batch: List[NDArray[np.floating]] = []
    for i in range(TILES):
        # Apply a strong Gaussian blur to smooth the sharp white noise into rolling terrain
        blurred: NDArray[np.floating] = cv2.GaussianBlur(noise_batch[i], KERNEL_01, 0)
        smoothed_batch.append(blurred)
        
        save_noise_image(blurred, "tiles", f"tile_{SEED}_{i}")
    
    return smoothed_batch

def _stitch(multiplier: int, tiles: NDArray[np.floating]) -> NDArray[np.float32]:
    """
    Creates a large noise grid by stitching together random base tiles.
    Completely vectorized using tensor transposition to eliminate Python loops.
    """
    total_tiles: int = multiplier * multiplier
    
    # 1. Native sampling of the 3D tiles array along Axis 0
    selected_tiles = STIT_RNG.choice(tiles, size=total_tiles, replace=True)

    # 2. Vectorized block-rearrangement (Eliminates the 404/4096 loop entirely!)
    # Reshape to 4D: (tile_row, tile_col, pixel_row, pixel_col)
    tensor_4d = selected_tiles.reshape(multiplier, multiplier, GRID_SIZE, GRID_SIZE)
    
    # Transpose to block tile_row with pixel_row, and tile_col with pixel_col
    transposed = tensor_4d.transpose(0, 2, 1, 3)
    
    # Collapse back into a continuous 2D stitched canvas 
    return transposed.reshape(multiplier * GRID_SIZE, multiplier * GRID_SIZE)

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
    t0: int = time.time()
    # Initial blur to blend the seams between stitched tiles
    cv2.blur(noise_grid, KERNEL_02, dst=noise_grid)
    t1: int = time.time()
    logging.debug(f"Time for initial blur: {t1-t0}s")
    # 2. High-Contrast Algebraic Sigmoid (Zero-Allocation Register Math)
    # Computes: noise_grid / sqrt(0.12 + noise_grid^2)
    denom = noise_grid ** 2
    
    # TUNING KNOB: 
    # Lower this value (e.g., 0.05) for harsher contrast and wider flat plateaus.
    # Raise this value (e.g., 0.25) to soften the cliffs into rolling hills.
    denom += 0.065 
    
    np.sqrt(denom, out=denom)
    noise_grid /= denom
    t2: int = time.time()
    logging.debug(f"Time for contrasting and tanh: {t2-t1}s")

    amount: float = UNSHARP_PERCENT / 100.0 if UNSHARP_PERCENT > 1 else float(UNSHARP_PERCENT)    
    # Apply Unsharp Masking to sharpen details (peaks and ridges)
    if amount != 0:
        blurred_mask: NDArray[np.float32] = cv2.blur(noise_grid, KERNEL_03)
        t3: int = time.time()
        logging.debug(f"Time for secondary blur: {t3-t2}s")
        cv2.addWeighted(
            noise_grid, 1.0 + amount, 
            blurred_mask, -amount, 
            0, 
            dst=noise_grid
        )
        np.clip(noise_grid, -1.0, 1.0, out=noise_grid)
        t4: int = time.time()
        logging.debug(f"Time for sharpening: {t4-t3}s")
    
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
    t0: int = time.time()
    noise_grid: NDArray[np.float32] = _stitch(multiplier, tiles)
    t1: int = time.time()
    logging.debug(f"Time for full _stitch: {t1-t0}s")
    noise_grid = _enhance(noise_grid)
    t2: int = time.time() 
    logging.debug(f"Time for full _enhance: {t2-t1}s") 
    noise_grid = _resize(noise_grid)   
    t3: int = time.time() 
    logging.debug(f"Time for full _resize: {t3-t2}s") 
    t4: int = time.time()
    
    logging.debug(f"Time for full _join_tiles for {multiplier}: {t4-t0}s")
    save_noise_image(noise_grid, "noises", f"noise_g{multiplier}_{SEED}")

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

    # Identify "water" or "lowland" areas and map them to 1.0 and 0.0
    mask: NDArray[np.float32] = (noise_sum <= THRESHOLD).astype(np.float32)
    
    if mask.any():
        # Soften the hard binary edges by blurring the mask itself
        smooth_alpha = cv2.GaussianBlur(mask, (15, 15), 0)
        # Apply extra smoothing to low areas to simulate sediment or water
        blurred_section: NDArray[np.float32] = cv2.blur(noise_sum, KERNEL_04)
        # Linearly blend across the entire map using the alpha channel
        # Deep valleys get the full 85% blur mix.
        # Highlands get 0% blur mix (completely untouched).
        # The borders transition perfectly smoothly, eliminating derivative creases.
        blend_factor = smooth_alpha * 0.85
        noise_sum = (blurred_section * blend_factor) + (noise_sum * (1.0 - blend_factor))
    # Final overall smooth to remove any remaining artifacts
    cv2.blur(noise_sum, KERNEL_05, dst=noise_sum)

    save_noise_image(noise_sum, "fnoises", f"noise_{SEED}")

    # Scale the normalized heightmap to the final physical height (Amplitude)
    return noise_sum * AMPLITUDE
