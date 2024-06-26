import numpy as np

class CompositeFunc:
    def __init__(self, loss_func, reg_func):
        self.d = loss_func.d
        self.loss_func = loss_func
        self.reg_func = reg_func

    def func_value(self, x):
        return self.loss_func.func_value(x) + self.reg_func.func_value(x)

    def grad(self, x):
        return self.loss_func.grad(x) + self.reg_func.grad(x)

    def grad_block(self, x, j):
        return self.loss_func.grad_block(x, j) + self.reg_func.grad_block(x, j)


