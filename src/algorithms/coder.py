import numpy as np
import time
import logging
from src.algorithms.utils.exitcriterion import ExitCriterion, CheckExitCondition
from src.problems.composite_func import CompositeFunc
from src.algorithms.utils.results import Results, logresult

# Setup logging
logging.basicConfig(level=logging.INFO)

def coder_stepsize(a_minus_1, A_minus_1, L, gamma):
    return (1 + gamma * A_minus_1) / (2 * L)


def coder(problem: CompositeFunc, exit_criterion: ExitCriterion, parameters, x0=None):
    # Init of CODER
    L, gamma = parameters["L"], parameters["gamma"]
    a, A = 0, 0

    if x0 is None:
        x0 = np.zeros(problem.d)
    
    x= np.copy(x0)
    x_minus_1= np.copy(x0)
    q, w = np.zeros_like(x0), np.zeros_like(x0)
    p = problem.loss_func.grad(x0)
    p_minus_1 = np.copy(p)
    z = np.zeros(problem.d)
    z_minus_1 = np.copy(z)
    loss_func_grad_x = np.copy(p)
    b_A_x = np.zeros(problem.loss_func.n)
    m = problem.d  # Assume that each block is simply a coordinate for now

    # Run init
    iteration = 0
    exit_flag = False
    start_time = time.time()
    results = Results()
    init_opt_measure = problem.func_value(x0)
    logresult(results, 1, 0.0, init_opt_measure)

    while not exit_flag:
        np.copyto(x_minus_1, x)
        np.copyto(p_minus_1, p)
        np.copyto(z_minus_1, z)
        
        # Step 4
        A_minus_1 = A
        a_minus_1 = a
        a = coder_stepsize(a_minus_1, A_minus_1, L, gamma)
        A = A_minus_1 + a


        np.copyto(w, x)
        loss_func_grad_x_minus_1 = loss_func_grad_x

        # Step 5
        for j in range(0, m, 1):
            # Step 6
            if j == 0:
                loss_func_grad_x, b_A_x = problem.loss_func.grad_block_update(x)
                p[0] = loss_func_grad_x[0]
            else:
                p[j], b_A_x = problem.loss_func.grad_block_update(b_A_x, (j-1, x[j-1] - x_minus_1[j-1]), j)

            # Step 7
            q[j] = p[j] + (a_minus_1 / a) * (loss_func_grad_x_minus_1[j] - p_minus_1[j])
            
            # Step 8
            z[j] = z_minus_1[j] + a * q[j]

            # Step 9
            x[j] = problem.reg_func.prox_opr_block(x0[j] - z[j], A)
            
        iteration += 1
        if iteration % exit_criterion.loggingfreq == 0:
            elapsed_time = time.time() - start_time
            opt_measure = problem.func_value(x)
            logging.info(f"elapsed_time: {elapsed_time}, iteration: {iteration}, opt_measure: {opt_measure}")
            logresult(results, iteration, elapsed_time, opt_measure)
            exit_flag = CheckExitCondition(exit_criterion, iteration, elapsed_time, opt_measure)

    return results, x
