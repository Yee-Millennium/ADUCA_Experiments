import numpy as np

class Ridge:
    def __init__(self, lambda_value):
        self.lambda_value = lambda_value
    
    def func_value(self, x):
        return 0.5 * self.lambda_value * np.linalg.norm(x)**2
    
    def grad(self, x):
        return self.lambda_value * x
    
    def grad_block(self, x, j):
        return self.lambda_value * x[j]
    
    def prox_opr_block(self, u, τ):
        return u / (τ * self.lambda_value + 1)
