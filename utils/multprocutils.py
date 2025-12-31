from concurrent.futures import ProcessPoolExecutor, wait
from utils.noiseutils import _conv_tilemap, _join_tiles
from utils.meshutils import _gen_mesh
from utils.tilegen import _gen_pop
from utils.config import (MULTIPLIERS, TILES)
from os import cpu_count
import logging


def _gen_tile():
    logging.debug("Generating a single tile...")
    matrix = _gen_pop()
    logging.debug(f"Matrix: {matrix}")
    tile = _conv_tilemap(matrix)
    logging.debug("Generated tile")
    return tile


def gen_tiles():
    logging.info(f"Generating {TILES} tiles...")
    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        futures = [executor.submit(_gen_tile) for _ in range(TILES)]
        wait(futures)
    tiles = list()
    for future in futures:
        tile = future.result()
        tiles.append(tile)
    logging.info(f"All {TILES} tiles generated")
    return tiles


def gen_noises(tiles):
    logging.info("Generating noise maps...")
    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        futures = [executor.submit(_join_tiles, multiplier, tiles) for multiplier in MULTIPLIERS]
        wait(futures)
    noises = list()
    for future in futures:
        noise = future.result()
        noises.append(noise)
    logging.info("All 5 noise maps generated")
    return noises


def gen_meshes(noises):
    logging.info("Generating VTK meshes...")
    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        futures = [executor.submit(_gen_mesh, noise) for noise in noises]
        wait(futures)
    logging.info(f"All {len(noises)} VTK meshes generated")