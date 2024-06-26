import numpy as np

class Results:
    def __init__(self):
        self.iterations = []
        self.times = []
        self.optmeasures = []

def logresult(results, current_iter, elapsed_time, opt_measure):
    """
    Append execution measures to Results.
    """
    results.iterations.append(current_iter)
    results.times.append(elapsed_time)
    results.optmeasures.append(opt_measure)
    return 
