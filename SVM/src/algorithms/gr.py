import numpy as np
import time
import logging
from src.algorithms.utils.results import Results, logresult
from src.problems.GMVI_func import GMVIProblem
from src.algorithms.utils.exitcriterion import ExitCriterion, CheckExitCondition


def gr(problem: GMVIProblem, exit_criterion: ExitCriterion, parameters, x_0=None):
    # Init of Golden-Ratio
    beta = parameters["beta"]
    rho = beta + beta**2
    if x_0 is None:
        x_0 = np.zeros(problem.d)
    x_1 = x_0 - np.full(shape=problem.d, fill_value=(-0.005))

    x = x_1
    x_minus_1 = x_0
    a = np.ones(2)
    v = np.copy(x)
    v_minus_1 = np.copy(v)

    A = 0
    x_sum = np.copy(x_0)
    x_hat = np.zeros(problem.d)

    # Stepsize selection function
    def gr_stepsize(x, x_minus_1, a_step):
        step_1 = rho * a_step[-1]

        delta_grad = problem.operator_func.func_map(x) - problem.operator_func.func_map(x_minus_1)
        if not np.any(delta_grad) :
            return step_1
            
        delta_x = (x - x_minus_1)
        step_2 = (1 / (4 * beta**2 * a_step[-2])) * (delta_x @ delta_x) / (delta_grad @ delta_grad)

        return min([step_1, step_2])


    # Run init
    iteration = 0
    exit_flag = False
    start_time = time.time()
    results = Results()
    init_opt_measure = problem.func_value(x_0)
    logresult(results, 1, 0.0, init_opt_measure)

    while not exit_flag:
        # Step 1
        step = gr_stepsize(x, x_minus_1, a)
        a = np.append(a[1:], step)

        x_minus_1 = np.copy(x)
        v_minus_1 = np.copy(v)

        # Step 2
        v = (1-beta) * x_minus_1 + beta * v_minus_1
        x = problem.g_func.prox_opr((v - step * problem.operator_func.func_map(x_minus_1)), step)
        
        ## Compute x_hat
        A += step
        x_sum += step * x
        x_hat = 1/A * x_sum

        iteration += 1
        if iteration % exit_criterion.loggingfreq == 0:
            elapsed_time = time.time() - start_time
            opt_measure = problem.func_value(x)
            logging.info(f"elapsed_time: {elapsed_time}, iteration: {iteration}, opt_measure: {opt_measure}")
            logresult(results, iteration, elapsed_time, opt_measure)
            exit_flag = CheckExitCondition(exit_criterion, iteration, elapsed_time, opt_measure)

    return results, x
