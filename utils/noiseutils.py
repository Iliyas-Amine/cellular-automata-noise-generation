from PIL import Image, ImageDraw, ImageEnhance
from scipy.ndimage import zoom
import numpy as np, random, logging, cv2
from os import urandom
from utils.config import (GRID_SIZE, UNSHARP_PERCENT, SAVE, RESIZE, CONTRAST_FACTOR, PATTERN, AMPLITUDE)

def _conv_tilemap(matrix):
    logging.debug("Starting conversion...")

    noise_map = np.zeros((GRID_SIZE, GRID_SIZE))

    above = 1 if np.count_nonzero(matrix == 1) < np.count_nonzero(matrix == 0) else 0
    mask_above = (matrix == above)

    logging.debug("Applying mask to create noise.")
    
    noise_map[mask_above] = np.random.uniform(0.01, 1, size=np.count_nonzero(mask_above))
    noise_map[~mask_above] = np.random.uniform(-1, -0.01, size=np.count_nonzero(~mask_above))

    logging.debug(f"Noise Matrix: {noise_map}")

    logging.debug("Smoothing the noise matrix")

    for i in range(5, 10):
        noise_map = cv2.blur(noise_map, (i,i))
    smoothed_noise_map = cv2.blur(noise_map, (10,10))

    logging.debug(f"Smooth Noise Matrix: {smoothed_noise_map}")

    if SAVE:
        logging.debug("Converting matrix to image")
        noise_image = Image.new('L', (64, 64))
        draw = ImageDraw.Draw(noise_image)
        for x in range(0, 64):
            for y in range(0, 64):
                z_value = smoothed_noise_map[x, y]
                color = int((z_value + 1) * 127.5)
                draw.rectangle([(x, y), (x, y)], fill=color)

        logging.debug("Applying contrast...")
        contrast_factor = 1.4
        noise_image = ImageEnhance.Contrast(noise_image).enhance(contrast_factor)
        logging.debug("Saving noise tile...")
        noise_image.save(f"tiles/tile_{urandom(8).hex()}.png")
    
    return smoothed_noise_map


def _join_tiles(multiplier, tiles):
    current_size = multiplier*GRID_SIZE

    noise_grid = np.zeros((current_size, current_size), dtype=np.float32)

    tile_grid = multiplier ** 2
    if len(tiles) < tile_grid:
        tiles *= int(tile_grid/len(tiles)) + 1
    random.shuffle(tiles)
    for i in range(multiplier):
        random.shuffle(tiles)
        for j in range(multiplier):
            tile_idx = i * multiplier + j
            if tile_idx < len(tiles):
                y_start, y_end = i * GRID_SIZE, (i + 1) * GRID_SIZE
                x_start, x_end = j * GRID_SIZE, (j + 1) * GRID_SIZE
                noise_grid[y_start:y_end, x_start:x_end] = tiles[tile_idx]

    for i in range(5, 11):
        noise_grid = cv2.blur(noise_grid, (i,i))

    noise_grid = noise_grid * CONTRAST_FACTOR
    noise_grid = np.clip(noise_grid, -1.0, 1.0)

    blurred_mask = cv2.blur(noise_grid, (10,10))
    mask = noise_grid - blurred_mask
    amount = UNSHARP_PERCENT / 100.0 if UNSHARP_PERCENT > 1 else UNSHARP_PERCENT
    noise_grid = noise_grid + (mask * amount)
    noise_grid = np.clip(noise_grid, -1.0, 1.0)

    zoom_ratio = RESIZE / current_size

    noise_grid = zoom(noise_grid, zoom_ratio, order=1)

    if SAVE:
        normalized_grid = ((noise_grid + 1) / 2.0) * 255.0
        img_data = normalized_grid.astype(np.uint8)
        
        img_obj = Image.fromarray(img_data, mode='L')
        
        filename = f'noises/noise_g{multiplier}_{urandom(4).hex()}.png'
        img_obj.save(filename)

    return noise_grid


def stacking(noises):
    noise_sum = np.zeros((RESIZE, RESIZE), dtype=np.float32)

    for grid, weight in zip(noises, PATTERN):
        noise_sum += grid * weight


    mask = (noise_sum <= -0.55)
    
    for i in range(15, 31):
        agg_blur = cv2.blur(noise_sum, (i,i))
    
    noise_sum[mask] = agg_blur[mask]

    noise_sum = cv2.blur(noise_sum, (3,3))

    if SAVE:
        normalized = ((noise_sum + 1) / 2.0) * 255.0
        img_data = np.clip(normalized, 0, 255).astype(np.uint8)
        noise_map = Image.fromarray(img_data, mode='L')
        
        filename = f"fnoises/noise_f_{urandom(4).hex()}.png"
        noise_map.save(filename)

    return noise_sum * AMPLITUDE
