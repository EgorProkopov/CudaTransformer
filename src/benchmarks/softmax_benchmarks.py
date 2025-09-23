from typing import Optional

import torch

from src.modules.softmax import SafeSoftmax


WARM_UP_RUNS = 30
ITERATIONS = 200


def torch_softmax_warpup():
    pass


def custom_softmax_warmup():
    pass


def torch_softmax_forward_benchmarks(x: torch.Tensor, return_res=True) -> tuple[float, Optional[torch.Tensor]]:
    """
    Compute torch forward softmax benchmark (cpu and cuda)
    """
    pass


def torch_softmax_backward_benchmarks(dy: torch.Tensor, y: torch.Tensor, return_res=True) -> tuple[float, Optional[torch.Tensor]]:
    pass


def custom_softmax_forward_benchmarks(x: torch.Tensor, return_res=True) -> tuple[float, Optional[torch.Tensor]]:
    pass


def custom_softmax_backward_benchmarks(dy: torch.Tensor, y: torch.Tensor, return_res=True) -> tuple[float, Optional[torch.Tensor]]:
    pass
