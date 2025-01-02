import numpy as np
import time
import numpy as np
import time
import logging
from src.algorithms.utils.exitcriterion import ExitCriterion, CheckExitCondition
from src.problems.GMVI_func import GMVIProblem
from src.algorithms.utils.results import Results, logresult
from src.algorithms.utils.helper import construct_block_range

def coder(problem: GMVIProblem, exitcriterion: ExitCriterion, parameters, x0=None):
    # Initialize parameters and variables
    n = problem.operator_func.n
    L = parameters["L"]
    block_size = parameters['block_size']
    blocks = construct_block_range(begin=0, end=n, block_size=block_size)
    m = len(blocks)
    logging.info(f"m = {m}")

    a, A = 0, 0
    x0 = np.ones(n) if x0 is None else x0
    x, x_prev = x0.copy(), x0.copy()
    Q = np.sum(x)
    x_tilde = x0.copy()
    x_tilde_sum = np.zeros(n)

    p = problem.operator_func.func_map(x0)
    p_prev = p.copy()
    F_store = np.copy(p)

    z, z_prev, q = np.zeros(n), np.zeros(n), np.zeros(n)

    # Initialization
    iteration = 0
    exitflag = False
    starttime = time.time()
    results = Results()  # Assuming Results is defined elsewhere
    init_optmeasure = problem.residual(x0)
    logresult(results, 1, 0.0, init_optmeasure)

    # Main loop
    while not exitflag:
        p_prev = np.copy(p)
        z_prev = np.copy(z)

        # Update steps
        A_prev = A
        a_prev = a
        a = 1 / (2 * L)
        A = A_prev + a

        F_x_prev = np.copy(F_store)

        for idx, block in enumerate(blocks):
            if idx != 0:
                problem.operator_func.func_map_block_update(F_store, x[block], Q, block)

            # Step 6
            p_prev[block] = p[block]
            p[block] = F_store[block]

            # Step 7
            q[block] = p[block] + (a_prev / a) * (F_x_prev[block] - p_prev[block])

            # Step 8
            z[block] = z_prev[block] + a * q[block]

            # Step 9
            x_prev[block] = x[block]
            x[block] = problem.g_func.prox_opr_block(x0[block] - z[block])

            Q += np.sum(x[block] - x_prev[block])
            
        x_tilde_sum += a * x
        iteration += m

        # Logging and exit condition
        if iteration % (m * exitcriterion.loggingfreq) == 0:
            x_tilde = x_tilde_sum / A
            elapsed_time = time.time() - starttime
            opt_measure = problem.residual(x_tilde)
            print(f"Elapsed time: {elapsed_time}, Iteration: {iteration}, Opt measure: {opt_measure}")
            logresult(results, iteration, elapsed_time, opt_measure)
            exitflag = CheckExitCondition(exitcriterion, iteration, elapsed_time, opt_measure)

    #  x_tilde
    return results, x




def coder_linesearch(problem: GMVIProblem, exitcriterion: ExitCriterion, parameters, x0=None):
    # Initialize parameters and variables
    n = problem.operator_func.n
    L = parameters["L"]
    block_size = parameters['block_size']
    blocks = construct_block_range(begin=0, end=n, block_size=block_size)
    m = len(blocks)
    logging.info(f"m = {m}")

    a, A = 0, 0
    x0 = np.ones(n) if x0 is None else x0
    x, x_prev = x0.copy(), x0.copy()
    Q = np.sum(x)
    x_tilde = x0.copy()
    x_tilde_sum = np.zeros(n)

    p = problem.operator_func.func_map(x0)
    p_prev = p.copy()
    F_store = np.copy(p)

    z, z_prev, q = np.zeros(n), np.zeros(n), np.zeros(n)

    # Initialization
    iteration = 0
    exitflag = False
    starttime = time.time()
    results = Results()  # Assuming Results is defined elsewhere
    init_optmeasure = problem.residual(x0)
    logresult(results, 1, 0.0, init_optmeasure)

    # Main loop
    while not exitflag:
        np.copyto(x_prev, x)
        np.copyto(p_prev, p)
        np.copyto(z_prev, z)
        F_x_prev = np.copy(F_store)
        Q = np.sum(x)
        A_prev = A
        a_prev = a
        L_ = L

        # Step 5
        L = L_ / 4

        # Step 6
        while{True}:
            # Step 7
            L = 2 * L
            temp_x = np.copy(x)
            temp_p = np.copy(p)
            temp_p_prev = np.copy(p_prev)
            temp_F_store = np.copy(F_store)
            temp_Q = Q

            # Step 8
            a = 1 / (2 * L)
            A = A_prev + a

            # Step 9
            for idx, block in enumerate(blocks):
                if idx != 0:
                    problem.operator_func.func_map_block_update(temp_F_store, temp_x[block], temp_Q, block)

                # Step 10
                temp_p_prev[block] = temp_p[block]
                temp_p[block] = F_store[block]

                # Step 11
                q[block] = temp_p[block] + (a_prev / a) * (F_x_prev[block] - temp_p_prev[block])

                # Step 12
                z[block] = z_prev[block] + a * q[block]

                # Step 13
                temp_x[block] = problem.g_func.prox_opr_block(x0[block] - z[block])

                temp_Q += np.sum(temp_x[block] - x_prev[block])
                
                
            # Step 15
            norm_F_p = np.linalg.norm(temp_F_store - temp_p)
            norm_x = np.linalg.norm(temp_x - x_prev)
            if norm_F_p <= L * norm_x:
                x = np.copy(temp_x)
                p = np.copy(temp_p)
                p_prev = np.copy(temp_p_prev)
                F_store = np.copy(temp_F_store)
                Q = temp_Q
                break

        x_tilde_sum += a * x
        iteration += m

        # Logging and exit condition
        if iteration % (m * exitcriterion.loggingfreq) == 0:
            print(f"!!! L: {L}")
            x_tilde = x_tilde_sum / A
            elapsed_time = time.time() - starttime
            opt_measure = problem.residual(x_tilde)
            print(f"Elapsed time: {elapsed_time}, Iteration: {iteration}, Opt measure: {opt_measure}")
            logresult(results, iteration, elapsed_time, opt_measure, L=L)
            exitflag = CheckExitCondition(exitcriterion, iteration, elapsed_time, opt_measure)

    #  x_tilde
    return results, x