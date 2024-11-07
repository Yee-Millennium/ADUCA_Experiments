import numpy as np
from src.problems.utils.data import Data

def libsvm_parser(path, n, d):
    features = np.zeros((n, d), dtype=np.float64)
    values = np.zeros(n, dtype=np.float64)

    with open(path, 'r') as file:
        data_str = file.readlines()

    for i in range(n):
        feature_value_str = data_str[i].strip().split(" ")
        value = float(feature_value_str[0])
        values[i] = value

        for j in range(1, len(feature_value_str)-1):
            idx_feature_pair = feature_value_str[j].split(':')
            idx = int(idx_feature_pair[0])
            feature = float(idx_feature_pair[1])
            features[i, idx] = feature

    return Data(features, values)

