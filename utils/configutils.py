import numpy as np
import tomllib
import random
from numpy.typing import NDArray
from typing import Any


def load_config(filename: str) -> dict[str, Any]:
    with open(filename, "rb") as f:
        data = tomllib.load(f)
    
    seed_str = str(data["SEED"]).zfill(32)

    config = {
    "SEED": seed_str,
    "POS_RNG": np.random.default_rng(seed=int(seed_str[:8])),
    "PROB_RNG": np.random.default_rng(seed=int(seed_str[8:16])),
    "SMOO_RNG": np.random.default_rng(seed=int(seed_str[16:24])),
    "STIT_RNG": np.random.default_rng(seed=int(seed_str[24:32])),
    "NEIGHBOR_KERNEL": np.array(data["kernels"]["NEIGHBOR_KERNEL"], dtype=np.int8),
    "KERNEL_01": tuple(data["kernels"]["KERNEL_1"]),
    "KERNEL_02": tuple(data["kernels"]["KERNEL_2"]),
    "KERNEL_03": tuple(data["kernels"]["KERNEL_3"]),
    "KERNEL_04": tuple(data["kernels"]["KERNEL_4"]),
    "KERNEL_05": tuple(data["kernels"]["KERNEL_5"]),
    "TILES": int(data["params"]["TILES"]),
    "GRID_SIZE": int(data["params"]["GRID_SIZE"]),
    "INITIAL_SEEDS": int(data["params"]["SEEDS_PER_TILE"]) * int(data["params"]["TILES"]),
    "UPDATE_ITERATIONS": int(data["params"]["UPDATE_ITERATIONS"]),
    "NEIGHBOR_ACTIVATION_FACTOR": float(data["params"]["NEIGHBOR_ACTIVATION_FACTOR"]),
    "THRESHOLD": float(data["params"]["THRESHOLD"]),
    "MULTIPLIERS": [int(m) for m in data["params"]["MULTIPLIERS"]],
    "AMPLITUDE": int(data["params"]["AMPLITUDE"]),
    "WEIGHTS": np.array(data["params"]["WEIGHTS"], dtype=np.float64),
    "RESIZE": int(data["params"]["RESIZE"]),
    "CONTRAST_FACTOR": float(data["params"]["CONTRAST_FACTOR"]),
    "BLEND_PERCENT": float(data["params"]["BLEND_PERCENT"]),
    "SAVE": bool(data["params"]["SAVE"])
    }

    return config

def load_default() -> dict[str, Any]:
    SEED = ''.join(random.choices('0123456789', k=32))
    TILES = 20

    config = {
    "NEIGHBOR_KERNEL": np.array([[
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]], dtype=np.int8),
    "KERNEL_01": (31, 31),
    "KERNEL_02": (15, 15),
    "KERNEL_03": (15, 15),
    "KERNEL_04": (31, 31),
    "KERNEL_05": (3, 3),
    "SEED": SEED,
    "POS_RNG": np.random.default_rng(seed=int(SEED[:8])),
    "PROB_RNG": np.random.default_rng(seed=int(SEED[8:16])),
    "SMOO_RNG": np.random.default_rng(seed=int(SEED[16:24])),
    "STIT_RNG": np.random.default_rng(seed=int(SEED[24:32])),
    "TILES": TILES,
    "GRID_SIZE": 64,
    "INITIAL_SEEDS": 5 * TILES,
    "UPDATE_ITERATIONS": 25,
    "NEIGHBOR_ACTIVATION_FACTOR": 0.11,
    "THRESHOLD": -0.55,
    "MULTIPLIERS": [2, 4, 8, 16, 32],
    "AMPLITUDE": 48,
    "WEIGHTS": np.array([0.50, 0.25, 0.09, 0.08, 0.08], dtype=np.float64),
    "RESIZE": 512,
    "CONTRAST_FACTOR": 3.0,
    "BLEND_PERCENT": 0.85,
    "SAVE": True
    }
    return config

