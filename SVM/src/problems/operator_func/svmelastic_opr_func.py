import numpy as np

class SVMElasticOprFunc:
    def __init__(self, data):
        self.d = data.features.shape[1]
        self.n = len(data.values)
        self.A = data.features
        self.b = data.values
        # print(f"!!! A.shape: {self.A.shape}")

    def func_value(self, x):
        assert len(x) == self.d + self.n
        _x = x[:self.d]
        res = 1 - (self.b * (self.A @ _x))  # Element-wise multiplication
        return np.sum(np.maximum(res, 0)) / self.n

    def func_map(self, x):
        assert len(x) == self.d + self.n
        ret = np.zeros(self.d + self.n)
        _x = x[:self.d]
        for t in range(self.n):
            ret[:self.d] += x[self.d + t] * self.b[t] * self.A[t, :]
            ret[self.d + t] = - (self.b[t] * (self.A[t, :] @ _x) - 1)
        return ret / self.n

    def func_map_block(self, j, x):
        assert len(x) == self.d + self.n
        assert 1 <= j <= self.d + self.n

        if j <= self.d:
            ret = 0.0
            for t in range(self.n):
                ret += x[self.d + t] * self.b[t] * self.A[t, j - 1]
            return ret / self.n
        else:
            t = j - self.d - 1
            return -(self.b[t] * (self.A[t, :] @ x[:self.d]) - 1) / self.n

    def func_map_block_sample(self, j, t, x):
        assert len(x) == self.d + self.n
        assert 1 <= j <= self.d + self.n
        assert 1 <= t <= self.n

        if j <= self.d:
            return x[self.d + t - 1] * self.b[t - 1] * self.A[t - 1, j - 1]
        elif j - self.d == t:
            return - (self.b[t - 1] * (self.A[t - 1, :] @ x[:self.d]) - 1)
        else:
            return 0.0