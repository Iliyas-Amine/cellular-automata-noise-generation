import numpy as np
import cv2
import os
from typing import List, Any

from numpy.typing import NDArray

def hash_func(shape, seed=0):
    nx, ny, nz = shape
    
    x = np.arange(nx, dtype=np.uint32)[:, None, None]
    y = np.arange(ny, dtype=np.uint32)[None, :, None]
    z = np.arange(nz, dtype=np.uint32)[None, None, :]

    h = (x * np.uint32(0x1B873593)) ^ (y * np.uint32(0x85EBCA6B)) ^ (z * np.uint32(0xC2B2AE35))
    
    if seed:
        seed_hash = (int(seed) * 0x9E3779B9) & 0xFFFFFFFF
        h ^= np.uint32(seed_hash)

    h ^= (h >> np.uint32(16))
    h *= np.uint32(0x7FEB352D)
    h ^= (h >> np.uint32(13))
    h *= np.uint32(0x846CA68B)
    h ^= (h >> np.uint32(16))

    return h.astype(np.float32) * (1.0 / 4294967296.0)

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
    os.makedirs(folder, exist_ok=True)
    # Normalize the float data (-1.0 to 1.0) to 0.0 to 2.0 range
    clipped = np.clip(data, -1.0, 1.0) 
    shifted = clipped + 1.0

    # Scale to 0-255 for standard 8-bit image format
    norm_img = (shifted * 127.5).astype(np.uint8) 
    filename = os.path.join(folder, f"{filename}.png") 
    cv2.imwrite(filename, norm_img)

def batch_tilemap(matrices: NDArray[np.int16], config: dict[str, Any]) -> NDArray[np.floating]:
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

    m_float = matrices.astype(np.float32)

    min_val = m_float.min()
    max_val = m_float.max()

    noise_batch = (m_float - min_val) / (max_val - min_val)

    mask_below = (matrices == 0)
    noise_batch[mask_below] = -1.0 * hash_func(matrices.shape)[mask_below]

    for i in range(config["TILES"]):
        if config["SAVE"]:
            save_noise_image(noise_batch[i], "tiles", f"tile_{config['SEED']}_{i}")
    
    return noise_batch

def _stitch(multiplier: int, tiles: NDArray[np.floating], config: dict[str, Any]) -> NDArray[np.float32]:
    GRID_SIZE = config["GRID_SIZE"]
    stride = GRID_SIZE // 2         

    hann_1d = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(GRID_SIZE) / (GRID_SIZE - 1)))
    hann_2d = np.outer(hann_1d, hann_1d).astype(np.float32)

    tiles_per_axis = multiplier + 1
    canvas_dim = (tiles_per_axis - 1) * stride + GRID_SIZE
    
    elev_canvas = np.zeros((canvas_dim, canvas_dim), dtype=np.float32)
    weight_canvas = np.zeros((canvas_dim, canvas_dim), dtype=np.float32)

    for r in range(tiles_per_axis):
        for c in range(tiles_per_axis):
            tile = config["STIT_RNG"].choice(tiles)
            y, x = r * stride, c * stride
            
            elev_canvas[y:y+GRID_SIZE, x:x+GRID_SIZE] += tile * hann_2d
            weight_canvas[y:y+GRID_SIZE, x:x+GRID_SIZE] += hann_2d

    normalized = elev_canvas / np.maximum(weight_canvas, 1e-7)

    return normalized[stride:-stride, stride:-stride]

def _enhance(noise_grid: NDArray[np.float32], scratch_grid: NDArray[np.float32], config: dict[str, Any]) -> None:
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

    np.multiply(noise_grid, noise_grid, out=scratch_grid)
    scratch_grid += config["CONTRAST_FACTOR"]/100 
    np.sqrt(scratch_grid, out=scratch_grid)
    noise_grid /= scratch_grid

def _resize(noise_grid: NDArray[np.float32], RESIZE: int) -> NDArray[np.float32]:
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

def _join_tiles(multiplier: int, tiles: NDArray[np.floating], scratch_grid: NDArray[np.float32], config: dict[str, Any]) -> NDArray[np.float32]:
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
    noise_grid = _stitch(multiplier, tiles, config)

    noise_grid = _resize(noise_grid, config["RESIZE"])

    _enhance(noise_grid, scratch_grid, config)
    
    if config["SAVE"]:
        save_noise_image(noise_grid, "noises", f"noise_g{multiplier}_{config['SEED']}")

    return noise_grid

def gen_noises(tiles: NDArray[np.floating], config: dict[str, Any]) -> List[NDArray[np.float32]]:
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
    scratch_grid = np.empty((config["RESIZE"], config["RESIZE"]), dtype=np.float32)

    noises: List[NDArray[np.float32]] = []
    for multiplier in config["MULTIPLIERS"]:
        noise_grid = _join_tiles(multiplier, tiles, scratch_grid, config)
        noises.append(noise_grid)
    return noises

def stacking(noises: List[NDArray[np.float32]], config: dict[str, Any]) -> NDArray[np.float32]:
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
    weights = np.asarray(config["WEIGHTS"], dtype=np.float32)

    # Perform weighted sum (Dot product) to combine all noise layers
    # This creates the fractal "Perlin-like" effect where some layers provide large shapes and others provide detail
    noise_sum = np.tensordot(weights, noise_stack, axes=1)

    # Identify "water" or "lowland" areas and map them to 1.0 and 0.0
    mask = (noise_sum <= config["THRESHOLD"]).astype(np.float32)
    
    if mask.any():
        # Soften the hard binary edges by blurring the mask itself
        smooth_alpha = cv2.GaussianBlur(mask, config["KERNEL_01"], 0)
        # Apply extra smoothing to low areas to simulate sediment or water
        blurred_section: NDArray[np.float32] = cv2.blur(noise_sum, config["KERNEL_02"])
        # Linearly blend across the entire map using the alpha channel
        blend_factor = smooth_alpha * config["BLEND_PERCENT"]
        noise_sum = (blurred_section * blend_factor) + (noise_sum * (1.0 - blend_factor))
    
    if config["SAVE"]:
        save_noise_image(noise_sum, "fnoises", f"noise_{config['SEED']}")

    # Scale the normalized heightmap to the final physical height (Amplitude)
    noise_sum *= float(config["AMPLITUDE"])
    return noise_sum
