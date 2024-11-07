import numpy as np
import time
import logging
from src.algorithms.utils.results import Results, logresult
from src.problems.composite_func import CompositeFunc
from src.algorithms.utils.exitcriterion import ExitCriterion, CheckExitCondition

def gd(problem: CompositeFunc, exit_criterion: ExitCriterion, parameters, x_0=None):
    # Init of GD
    L = parameters['L']
    if x_0 is None:
        x_0 = np.zeros(problem.d)
    x = np.copy(x_0)

    # Run init
    iteration = 0
    exit_flag = False
    start_time = time.time()
    results = Results()
    init_opt_measure = problem.func_value(x_0)
    logresult(results, 1, 0.0, init_opt_measure)

    while not exit_flag:
        x = x - problem.grad(x) * (1/L)

        iteration += 1
        if iteration % exit_criterion.loggingfreq == 0:
            elapsed_time = time.time() - start_time
            opt_measure = problem.func_value(x)
            logging.info(f"elapsed_time: {elapsed_time}, iteration: {iteration}, opt_measure: {opt_measure}")
            logresult(results, iteration, elapsed_time, opt_measure)
            exit_flag = CheckExitCondition(exit_criterion, iteration, elapsed_time, opt_measure)

    return results, x
