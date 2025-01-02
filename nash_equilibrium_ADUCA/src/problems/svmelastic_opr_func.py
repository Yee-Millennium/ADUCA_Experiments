import numpy as np
# from scipy.linalg.blas import dgemm
from scipy.sparse import csr_matrix

class SVMElasticOprFunc:
    def __init__(self, n, gamma, beta, c , L):
        self.n = n
        self.gamma = gamma
        self.beta = beta
        self.c = c
        self.L = L

    def f(self, q):
        t = 1/self.beta
        res = self.c * q + 1/(1+t)*(self.L**t * q**(1+t))
        return res
    
    def f_block(self, q, block:range):
        t = 1./self.beta[block]
        res = self.c[block] * q[block] + 1./(1.+t[block])*(self.L[block]**t * q[block]**(1+t))
        return res

    def df(self, q):
        t = 1./self.beta
        res = self.c + (self.L* q)**t
        return res
    
    def df_block(self, q_block, block:range):
        t = 1./self.beta[block]
        res = self.c[block] + (self.L[block]* q_block)**t
        return res
    
    def p(self, Q):
        return (5000**(1/self.gamma)) * (Q**(-1/self.gamma))
    
    def dp(self, Q):
        res = -1./self.gamma * (5000**(1./self.gamma)) * (Q**(-1./self.gamma -1))
        return res
    
    def func_map(self, q):
        Q = np.sum(q)
        res = self.df(q) - self.p(Q) - q*self.dp(Q)
        return res

    def func_map_block(self, q_block, Q, block):
        res = self.df_block(q_block, block) - self.p(Q) - q_block*self.dp(Q)
        return res
    
    def func_map_block_update(self, F, q_block, Q, block:range):
        F[block] = self.func_map_block(q_block, Q, block)
        return F

    # def func_map_block_sample(self, j, t, x):
    #     assert len(x) == self.d + self.n
    #     assert 1 <= j <= self.d + self.n
    #     assert 1 <= t <= self.n

    #     if j <= self.d:
    #         return x[self.d + t - 1] * self.b[t - 1] * self.A[t - 1, j - 1]
    #     elif j - self.d == t:
    #         return - (self.b[t - 1] * (self.A[t - 1, :] @ x[:self.d]) - 1)
    #     else:
    #         return 0.0