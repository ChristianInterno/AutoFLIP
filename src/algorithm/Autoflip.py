import torch
from .basealgorithm import BaseOptimizer

from .fedavg import FedavgOptimizer



class AutoflipOptimizer(FedavgOptimizer):
    def __init__(self, params, **kwargs):
        super(AutoflipOptimizer, self).__init__(params=params, **kwargs)
