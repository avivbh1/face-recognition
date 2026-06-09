import numpy as np
from pathlib import Path

REFERENCE_PATH = Path("reference.npy")


def save_reference(embedding):
    np.save(REFERENCE_PATH, embedding)


def load_reference():
    if not REFERENCE_PATH.exists():
        return None
    return np.load(REFERENCE_PATH)
