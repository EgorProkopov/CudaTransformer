import torch

from src.modules.softmax import SafeSoftmax


def torch_softmax_forward_benchmarks(x: torch.Tensor):
    """
    Compute torch softmax benchmark and
    """
    
