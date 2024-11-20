import numpy as np
import time
import logging
from src.algorithms.utils.results import Results, logresult
from src.problems.GMVI_func import GMVIProblem
from src.algorithms.utils.exitcriterion import ExitCriterion, CheckExitCondition


def gr(problem: GMVIProblem, exit_criterion: ExitCriterion, parameters, x_0=None):
    # Init of Golden-Ratio
    d = problem.operator_func.d
    n = problem.operator_func.n
    beta = parameters["beta"]
    rho = beta + beta**2
    if x_0 is None:
        x_0 = np.zeros(problem.d)
    x_1 = x_0 - np.full(shape=problem.d, fill_value=(-0.1))

    x = np.copy(x_1)
    x_ = np.copy(x_0)
    v = np.copy(x_1)
    v_ = np.copy(x_1)


    a = 1
    a_ = 1
    A = 1

    x_hat = a * x

    F = problem.operator_func.func_map(x)
    F_ = problem.operator_func.func_map(x_)

    # Stepsize selection function
    def gr_stepsize(a , a_, x, x_, F, F_ ):
        step_1 = rho * a
 
        F_norm = np.linalg.norm(F - F_)
        if F_norm == 0 :
            return step_1
        
        x_norm = np.linalg.norm(x - x_)
        step_2 = (x_norm)**2 / ((4 * beta**2 * a_) * F_norm ** 2)

        step = min(step_1, step_2)
        # print(f"!!! step: {step}")
        return step


    # Run init
    iteration = 0
    exit_flag = False
    start_time = time.time()
    results = Results()
    init_opt_measure = problem.func_value(x_0)
    logresult(results, 1, 0.0, init_opt_measure)

    while not exit_flag:
        step = gr_stepsize(a, a_, x, x_, F, F_)
        a_ = a
        a = step
        A += a

        v = (1-beta) * x + beta * v_

        v_ = np.copy(v)
        x_ = np.copy(x)

        x = problem.g_func.prox_opr(v - a * F, a, d)

        F_ = np.copy(F)
        F = problem.operator_func.func_map(x)

        x_hat = (A - a)/A * x_hat + a/A * x

        iteration += problem.d
        if iteration % ( problem.d *  exit_criterion.loggingfreq) == 0:
            elapsed_time = time.time() - start_time
            opt_measure = problem.func_value(x_hat)
            logging.info(f"elapsed_time: {elapsed_time}, iteration: {iteration}, opt_measure: {opt_measure}")
            logresult(results, iteration, elapsed_time, opt_measure)
            exit_flag = CheckExitCondition(exit_criterion, iteration, elapsed_time, opt_measure)

    return results, x
