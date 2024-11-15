import numpy as np

class SVMElasticGFunc:
    def __init__(self, d, n, lambda1, lambda2):
        self.lambda1 = lambda1  # Lasso regularization parameter
        self.lambda2 = lambda2  # Ridge regularization parameter
        self.d = d
        self.n = n

    def func_value(self, x):
        assert len(x) == self.d + self.n

        ret_1 = np.sum(np.abs(x[:self.d]))  # L1 regularization part
        ret_2 = np.sum(x[:self.d] ** 2)     # L2 regularization part
        ret = self.lambda1 * ret_1 + (self.lambda2 / 2) * ret_2

        # Constraint check for elements of x[d+1:] to be in [-1, 0]
        if not np.all((-1.0 <= x[self.d:]) & (x[self.d:] <= 0.0)):
            return -np.inf

        return ret

    def prox_opr_block(self, j, u, tau):
        assert 1 <= j <= self.n + self.d

        if j <= self.d:
            p1 = tau * self.lambda1
            p2 = 1.0 / (1.0 + tau * self.lambda2)
            return self._prox_func(u, p1, p2)
        else:
            return min(0.0, max(-1.0, u))

    @staticmethod
    def _prox_func(_x0, p1, p2):
        if _x0 > p1:
            return p2 * (_x0 - p1)
        elif _x0 < -p1:
            return p2 * (_x0 + p1)
        else:
            return 0.0
        
    def prox_opr(self, u, τ):
        p1 = τ * self.lambda1
        p2 = 1.0 / (1.0 + τ * self.lambda2)
        prox = p2 * np.sign(u) * np.maximum(0, np.abs(u) - p1)

        p = np.minimum(0, np.maximum(-1, prox))
        return p