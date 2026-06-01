import os
import random
import warnings

import torch


def set_global_seed(seed: int) -> None:
    # Request deterministic CUDA kernels when available.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    warnings.filterwarnings(
        "ignore",
        message=r"Memory Efficient attention defaults to a non-deterministic algorithm\..*",
        category=UserWarning,
    )
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.allow_tf32 = False
