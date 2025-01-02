import numpy as np
import time
import numpy as np
import time
import logging
from src.algorithms.utils.exitcriterion import ExitCriterion, CheckExitCondition
from src.problems.GMVI_func import GMVIProblem
from src.algorithms.utils.results import Results, logresult
from src.algorithms.utils.helper import construct_block_range

def pccm(problem: GMVIProblem, exitcriterion: ExitCriterion, parameters, x0=None):
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
            q[block] = p[block]

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