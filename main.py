# Local application imports
from utils.noiseutils import stacking, gen_noises
from utils.tilegen import gen_tiles
from utils.meshutils import gen_mesh
import utils.configutils as cutils

# Helping imports
import time, logging, os

# Configure logging to write to a timestamped file in the 'logs' directory
os.makedirs("logs", exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(funcName)s - %(levelname)s - %(message)s',
                    filename=f"logs/{time.strftime('%Y%m%d_%H%M%S')}.log")


def run(config_file=None) -> None:
    """
    Executes the full cellular meshing pipeline.

    This main driver function orchestrates the procedural generation process in four stages:
    1.  Tile Generation: Creates base patterns using cellular automata.
    2.  Noise Synthesis: Stitches these patterns into multi-scale noise layers.
    3.  Stacking: Blends the noise layers into a single cohesive heightmap.

    It also logs performance metrics (execution time) for each stage to the logs directory.
    """

    # Loading config file if exists, else use default values
    if config_file:
        config = cutils.load_config(config_file)
    else:
        config = cutils.load_default()

    # Start the timer for performance tracking
    t0 = time.time()
    logging.info(f"Saving Status: {config['SAVE']}")
    
    # Step 1: Generate the base cellular automata tiles
    # These are small, tileable patterns generated via simulation
    tiles = gen_tiles(config)
    
    t1 = time.time()
    logging.info(f"Time in seconds: Tile Generation in {t1-t0}s")
    
    # Step 2: Create multi-scale noise layers
    # Stitch the base tiles together at different frequencies (multipliers) to form larger grids
    noises = gen_noises(tiles, config)
    
    # Step 3: Blend the noise layers
    # Combine the different frequency layers into a single heightmap using weighted stacking
    noise = stacking(noises, config)
    
    t2 = time.time()
    logging.info(f"Time in seconds: One Noise Map Generation in {t2-t1}s")

    logging.info(f"Time in seconds: Full Program in {t2-t0}s")
    logging.info("Program exit")


if __name__ == "__main__":
    run("config.toml")
