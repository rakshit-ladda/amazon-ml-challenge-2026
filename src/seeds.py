"""Fixed seeds + deterministic GPU kernels for every training step (SEED env, default 42)."""
import os
import random

import numpy as np

SEED = int(os.environ.get("SEED", 42))


def set_all(seed: int = SEED):
    """Seed python/numpy/torch and request deterministic cuDNN/cuBLAS kernels."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass
