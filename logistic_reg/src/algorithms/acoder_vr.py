import numpy as np
import time
import logging
import random
from src.algorithms.utils.exitcriterion import ExitCriterion, CheckExitCondition
from src.problems.composite_func import CompositeFunc
from src.algorithms.utils.results import Results, logresult

# Setup logging
logging.basicConfig(level=logging.INFO)

def acodervr_stepsize(A_minus1, L, gamma, K):
    _ret = K * A_minus1 * (1 + A_minus1 * gamma) / (8 * L)
    return np.sqrt(_ret)

def acodervr(problem: CompositeFunc, exit_criterion: ExitCriterion, parameters, x0=None):
    # Short names
    grad_tj_f = problem.loss_func._grad_block_sample
    grad_f = problem.loss_func.grad
    prox_opr_block_g = problem.reg_func.prox_opr_block

    # Init of ACODERVR
    L, gamma, K = parameters["L"], parameters["gamma"], parameters["K"]
    a_minus1, A_minus1 = 0.0, 0.0
    a, A = 1 / (4 * L), 1 / (4 * L)
    if x0 is None:
        x0 = np.zeros(problem.d)
    w = np.copy(x0)
    v = np.copy(x0)
    x = np.copy(x0)
    y = np.copy(x0)
    w_minus1 = np.copy(x0)
    v_minus1 = np.copy(x0)
    x_minus1 = np.copy(x0)
    y_minus1 = np.copy(x0)
    q = np.zeros(problem.d)
    z_minus1 = np.zeros(problem.d)
    z = np.zeros(problem.d)
    y_tilde_minus1 = np.copy(x0)
    y_tilde = np.copy(x0)
    y_tilde_sum = np.zeros(problem.d)
    m = problem.d  # Assume that each block is simply a coordinate for now
    n = list(range(problem.loss_func.n))  # n = [0, 1, ..., n-1]

    # Seeding of ACODERVR
    z = z_minus1 + grad_f(x0)
    v = np.array([prox_opr_block_g(x0_i - z_i, A) for x0_i, z_i in zip(x0, z)])
    y_tilde = np.copy(v)
    y = np.copy(v)

    # Run init
    iteration = 0
    exit_flag = False
    start_time = time.time()
    results = Results()
    init_opt_measure = problem.func_value(y)
    logresult(results, 1, 0.0, init_opt_measure)

    while not exit_flag:
        np.copyto(y_tilde_minus1, y_tilde)
        y_tilde_sum.fill(0.0)

        # Step 9 and 10
        a_minus1 = a
        A_minus1 = A
        a = acodervr_stepsize(A_minus1, L, gamma, K)
        A = A_minus1 + a

        # Step 12
        mu = grad_f(y_tilde_minus1)

        for k in range(1, K+1):
            np.copyto(v_minus1, v)
            np.copyto(x_minus1, x)
            np.copyto(y_minus1, y)
            np.copyto(z_minus1, z)

            # Step 14
            x = (A_minus1 * y_tilde_minus1 + a * v_minus1) / A

            np.copyto(w, x)
            np.copyto(w_minus1, x_minus1)

            for j in range(m-1, -1, -1):  # From m-1 down to 0
                # Step 16
                if j <= m - 2:
                    w[j+1] = y[j+1]
                    w_minus1[j+1] = y_minus1[j+1]

                # Step 17
                t = random.choice(n)

                # Step 18 and 19
                if k == 1:
                    a0_minus1 = a_minus1
                else:
                    a0_minus1 = a
                _grad_vr = grad_tj_f(w, j, t) - grad_tj_f(y_tilde_minus1, j, t) + mu[j]
                q[j] = _grad_vr + a0_minus1 / a * (grad_tj_f(x_minus1, j, t) - grad_tj_f(w_minus1, j, t))

                # Step 20
                z[j] = z_minus1[j] + a * q[j]

                # Step 21
                v[j] = prox_opr_block_g(x0[j] - z[j] / K, A_minus1 + a * k / K)

                # Step 22
                y[j] = (A_minus1 * y_tilde_minus1[j] + a * v[j]) / A

            y_tilde_sum += y

        y_tilde = y_tilde_sum / K

        iteration += 1 + 4 * K / problem.loss_func.n  # TODO: Check and verify
        if iteration % exit_criterion.loggingfreq == 0:
            elapsed_time = time.time() - start_time
            opt_measure = problem.func_value(y_tilde)
            logging.info(f"elapsed_time: {elapsed_time}, iteration: {iteration}, opt_measure: {opt_measure}")
            logresult(results, iteration, elapsed_time, opt_measure)
            exit_flag = CheckExitCondition(exit_criterion, iteration, elapsed_time, opt_measure)

    return results, v, y_tilde