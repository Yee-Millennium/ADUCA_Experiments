import numpy as np

class LogisticLoss:
    def __init__(self, data):
        self.n = len(data.values)
        self.d = data.features.shape[1]
        self.A = data.features
        self.b = data.values

    def func_value(self, x):
        A_x = self.A @ x
        b_A_x = self.b * A_x
        tmp = np.log(np.exp(-b_A_x) + 1)
        return np.sum(tmp) / self.n

    def grad(self, x):
        A_x = self.A @ x
        b_A_x = self.b * A_x
        tmp = -self.b * np.exp(-b_A_x) / (np.exp(-b_A_x) + 1) / self.n
        return tmp @ self.A

    def grad_block_update(self, *args):
        # (x)
        if len(args) == 1:
            x = args[0]
            A_x = self.A @ x
            b_A_x = self.b * A_x
            tmp = -self.b * np.exp(-b_A_x) / (np.exp(-b_A_x) + 1) / self.n
            return tmp @ self.A, b_A_x
        # (x, j)
        elif len(args) == 2:
            x= args[0]
            j = args[1]
            A_x = self.A @ x
            b_A_x = self.b * A_x
            tmp = -self.b * np.exp(-b_A_x) / (np.exp(-b_A_x) + 1) / self.n
            return (tmp @ self.A)[j], b_A_x
        else:
            # (b_A_x, update_x, j)
            b_A_x = args[0]
            i, delta_x_i = args[1]
            j = args[2]
            b_A_x += self.b * delta_x_i * self.A[:, i]
            grad_block_x = np.sum(-self.b * np.exp(-b_A_x) / (1 + np.exp(-b_A_x)) * self.A[:, j]) / self.n
            return grad_block_x, b_A_x

    def grad_block(self, x, j):
        return self._grad(x)[j]

    def _grad_block_sample(self, x, j, t):
        a_x = 0.0
        for i in range(self.d):
            a_x += self.A[t, i] * x[i]
        b_a_x = self.b[t] * a_x
        return -self.b[t] * np.exp(-b_a_x) / (1 + np.exp(-b_a_x)) * self.A[t, j]