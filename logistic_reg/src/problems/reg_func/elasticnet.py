import numpy as np

def _prox_func(u, p1, p2):
    _value = 0.0
    if u > p1:
        _value = p2 * (u - p1)
    elif u < -p1:
        _value = p2 * (u + p1)
    return _value

class ElasticNet:
    def __init__(self, lambda1, lambda2):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
    
    def func_value(self, x):
        return np.sum(self.lambda1 * np.abs(x) + (self.lambda2 / 2) * x**2)
    
    def grad(self, x):
        return self.lambda1 * np.sign(x) + self.lambda2 * x
    
    def grad_block(self, x, j):
        return self.lambda1 * np.sign(x[j]) + self.lambda2 * x[j]
    
    
    def prox_opr_block(self, u, τ):
        p1 = τ * self.lambda1
        p2 = 1.0 / (1.0 + τ * self.lambda2)
        return _prox_func(u, p1, p2)

    def prox_opr(self, u, τ):
        p1 = τ * self.lambda1
        p2 = 1.0 / (1.0 + τ * self.lambda2)
        prox = p2 * np.sign(u) * np.maximum(0, np.abs(u) - p1)
        return prox
