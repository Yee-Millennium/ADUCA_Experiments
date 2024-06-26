import numpy as np

class Data:
    def __init__(self, features, values):
        self.features = np.array(features, dtype=np.float64)
        self.values = np.array(values, dtype=np.float64)