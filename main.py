import logging
import time
import shutil
import os
from typing import List

# Third-party imports
import numpy as np
from numpy.typing import NDArray

# Local application imports
from utils.noiseutils import stacking, gen_noises
from utils.tilegen import gen_tiles
from utils.meshutils import gen_mesh
from utils.config import SAVE

# Configure logging to write to a timestamped file in the 'logs' directory
os.makedirs("logs", exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(funcName)s - %(levelname)s - %(message)s',
                    filename=f"logs/{time.strftime('%Y%m%d_%H%M%S')}.log")

def clean_dirs():
    """
    Cleans up intermediate output directories before a new run.

    This utility removes existing output directories ('tiles', 'noises', 
    'fnoises', 'meshes') and recreates them empty. This prevents disk bloat 
    from accumulating thousands of files over multiple executions (e.g., 
    preventing 1,000+ PNGs from filling the disk after 50 runs).
    """
    targets = ["tiles", "noises", "fnoises", "meshes"]
    for t in targets:
        if os.path.exists(t):
            shutil.rmtree(t)
        os.makedirs(t)
    logging.info("Cleared Directories")

def run() -> None:
    """
    Executes the full cellular meshing pipeline.

    This main driver function orchestrates the procedural generation process in four stages:
    1.  **Tile Generation**: Creates base patterns using cellular automata.
    2.  **Noise Synthesis**: Stitches these patterns into multi-scale noise layers.
    3.  **Stacking**: Blends the noise layers into a single cohesive heightmap.
    4.  **Mesh Generation**: Converts the 2D heightmap into a 3D VTK mesh.

    It also logs performance metrics (execution time) for each stage to the logs directory.
    """
    # Start the timer for performance tracking
    t0: float = time.time()
    logging.info(f"Saving Status: {SAVE}")
    
    # Step 1: Generate the base cellular automata tiles
    # These are small, tileable patterns generated via simulation
    tiles: List[NDArray[np.floating]] = gen_tiles()
    
    t1: float = time.time()
    logging.info(f"Time in seconds: Tile Generation in {t1-t0}s")
    
    # Step 2: Create multi-scale noise layers
    # Stitch the base tiles together at different frequencies (multipliers) to form larger grids
    noises: List[NDArray[np.float32]] = gen_noises(tiles)
    
    # Step 3: Blend the noise layers
    # Combine the different frequency layers into a single heightmap using weighted stacking
    noise: NDArray[np.float32] = stacking(noises)
    
    t2: float = time.time()
    logging.info(f"Time in seconds: One Noise Map Generation in {t2-t1}s")
    
    # Step 4: Generate 3D Geometry
    # Convert the final 2D heightmap into a structured VTK mesh
    gen_mesh(noise)
    
    t3: float = time.time()
    logging.info(f"Time in seconds: VTK Mesh Generation in {t3-t2}s")
    logging.info(f"Time in seconds: Full Program in {t3-t0}s")
    logging.info("Program exit")


# Entry point of the script
if __name__ == "__main__":
    clean_dirs()
    run()