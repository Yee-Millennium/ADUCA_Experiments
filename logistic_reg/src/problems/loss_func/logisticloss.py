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
        elif len(args) == 3:
            # (b_A_x, (j-1, update_x), j)
            b_A_x = args[0]
            i, delta_x_i = args[1]
            j = args[2]
            b_A_x += self.b * delta_x_i * self.A[:, i]
            grad_block_x = np.sum(-self.b * np.exp(-b_A_x) / (1 + np.exp(-b_A_x)) * self.A[:, j]) / self.n
            return grad_block_x, b_A_x
        elif len(args) == 5:
            # (b_A_x, update_x, j, blocksize_u, block_size_F)
            # print(f"!!! A's shape: {self.A.shape}")
            b_A_x = args[0]
            delta_u_j_1 = args[1]
            j = args[2] 
            block_size_u = args[3]
            block_size_F = args[4]
            i = (j-1) * block_size_u
            # print(f"!!! block_size: {block_size}")
            # print(f"!!! delta_u_j_1.shape: {delta_u_j_1.shape}")
            # print(f"!!! A[:, i:(i+block_size_u)]'s shape: {self.A[:, i:(i+block_size_u)].shape}")
            b_A_x += self.b *  (self.A[:, i:(i+block_size_u)] @ delta_u_j_1)
            # print(f"If the b_A_x the same? {np.array_equal(self.b *  (self.A[:, i:(i+block_size_u)] @ delta_u_j_1), self.b * delta_u_j_1 * self.A[:, i])}")
            # print(f"!!! b_A_x's shape: {b_A_x.shape}")
            # print(f"!!! A the same? {np.array_equal(np.reshape(self.A[:, j:(j+block_size_F)].T, -1), self.A[:, j])}")
            grad_block_u = ((-self.b * np.exp(-b_A_x) / (1 + np.exp(-b_A_x))) @ self.A[:, j:(j+block_size_F)]) / self.n
            # print(f"A[:,j:(j+block_size_F)]'s shape]: {self.A[:,j:(j+block_size_F)].shape}")
            # print(f"!!! The sum of current version: {grad_block_u}")
            # print(f"!!! The sum of previous version: {(np.sum(-self.b * np.exp(-b_A_x) / (1 + np.exp(-b_A_x)) * self.A[:, j]) / self.n)} ")
            # print(f"If sums' is the same? {(np.sum(-self.b * np.exp(-b_A_x) / (1 + np.exp(-b_A_x)) * self.A[:, j]) / self.n) == grad_block_u}")
            return grad_block_u, b_A_x

    def grad_block(self, x, j):
        return self._grad(x)[j]

    def _grad_block_sample(self, x, j, t):
        a_x = 0.0
        for i in range(self.d):
            a_x += self.A[t, i] * x[i]
        b_a_x = self.b[t] * a_x
        return -self.b[t] * np.exp(-b_a_x) / (1 + np.exp(-b_a_x)) * self.A[t, j]