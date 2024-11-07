import numpy as np
import time
import logging
from src.algorithms.utils.exitcriterion import ExitCriterion, CheckExitCondition
from src.problems.composite_func import CompositeFunc
from src.algorithms.utils.results import Results, logresult

# Setup logging
logging.basicConfig(level=logging.INFO)

def acoder_stepsize(a_minus1, A_minus1, L, gamma):
    # Largest value such that Line 4 of the algorithm is satisfied
    _r = 0.4 * (1 + A_minus1 * gamma) / L
    _b, _c = -_r, -A_minus1 * _r
    return 0.5 * (-_b + np.sqrt(_b**2 - 4 * _c))

def acoder(problem: CompositeFunc, exit_criterion: ExitCriterion, parameters, x0=None):
    # Initialization of ACODER
    L, gamma = parameters["L"], parameters["gamma"]
    a, A = 0.0, 0.0
    if x0 is None:
        x0 = np.zeros(problem.d)
    v = np.copy(x0)
    x = np.copy(x0)
    y = np.copy(x0)
    v_minus1 = np.copy(x0)
    x_minus1 = np.copy(x0)
    y_minus1 = np.copy(x0)
    q = np.zeros_like(x0)
    w = np.zeros_like(x0)
    p = problem.loss_func.grad(x0)
    p_minus1 = np.copy(p)
    z = np.zeros(problem.d)
    z_minus1 = np.copy(z)
    loss_func_grad_x = np.copy(p)
    b_A_x = np.zeros(problem.loss_func.n)
    m = problem.d  # Assume that each block is a coordinate for now

    # Run initialization
    iteration = 0
    last_logged_iter = 0
    exit_flag = False
    start_time = time.time()
    results = Results()
    init_opt_measure = problem.func_value(x0)
    logresult(results, 1, 0.0, init_opt_measure)

    while not exit_flag:
        # Save previous variables
        v_minus1[:] = v
        x_minus1[:] = x
        y_minus1[:] = y
        p_minus1[:] = p
        z_minus1[:] = z

        # Step 4
        A_minus1 = A
        a_minus1 = a
        a = acoder_stepsize(a_minus1, A_minus1, L, gamma)
        A = A_minus1 + a

        # Step 5
        x = (A_minus1 / A) * y + (a / A) * v

        w[:] = x
        loss_func_grad_x_minus1 = np.copy(loss_func_grad_x)
        for j in range(m, 0, -1):  # Loop from m down to 1
            idx = j - 1  # Adjust for zero-based indexing
            # Step 7
            if j == m:
                loss_func_grad_x, b_A_x = problem.loss_func.grad_block_update(x)
                p[idx] = loss_func_grad_x[idx]
            else:
                # Update gradient block
                delta = y[idx + 1] - x[idx + 1]
                p[idx], b_A_x = problem.loss_func.grad_block_update(b_A_x, (idx + 1, delta), idx)

            # Step 8
            q[idx] = p[idx] + (a_minus1 / a) * (loss_func_grad_x_minus1[idx] - p_minus1[idx])

            # Step 9
            z[idx] = z_minus1[idx] + a * q[idx]

            # Step 10
            v[idx] = problem.reg_func.prox_opr_block(x0[idx] - z[idx], A)

            # Step 11
            y[idx] = (A_minus1 / A) * y_minus1[idx] + (a / A) * v[idx]

        iteration += 2
        if (iteration - last_logged_iter) >= exit_criterion.loggingfreq:
            last_logged_iter = iteration
            elapsed_time = time.time() - start_time
            opt_measure = problem.func_value(y)
            logging.info(f"elapsed_time: {elapsed_time}, iteration: {iteration}, opt_measure: {opt_measure}")
            logresult(results, iteration, elapsed_time, opt_measure)
            exit_flag = CheckExitCondition(exit_criterion, iteration, elapsed_time, opt_measure)

    return results, v, y