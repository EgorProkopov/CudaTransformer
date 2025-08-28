import torch
import torch.nn as nn

from modules import rmsnorm_custom_cuda as rmsnorm_cuda


class RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward():
        pass


    @staticmethod
    def backward():
        pass


class RMSNorm(nn.Module):
    def __init__(self):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


if __name__ == "__main__":
    pass