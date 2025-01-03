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
    L = parameters["L"] * 2
    block_size = parameters['block_size']
    blocks = construct_block_range(begin=0, end=n, block_size=block_size)
    m = len(blocks)
    logging.info(f"m = {m}")

    a, A = 0, 0
    x0 = np.ones(n) if x0 is None else x0
    x, x_prev = x0.copy(), x0.copy()
    Q = np.sum(x)
    p = problem.operator_func.p(Q)
    p_ = p
    dp = problem.operator_func.dp(Q)
    dp_ = dp
    x_tilde = x0.copy()
    x_tilde_sum = np.zeros(n)

    F_tilde = problem.operator_func.func_map(x0)
    F_tilde_prev = F_tilde.copy()
    F_store = np.copy(F_tilde)

    z, z_prev, F_bar = np.zeros(n), np.zeros(n), np.zeros(n)

    # Initialization
    iteration = 0
    exitflag = False
    starttime = time.time()
    results = Results()  # Assuming Results is defined elsewhere
    init_optmeasure = problem.residual(x0)
    logresult(results, 1, 0.0, init_optmeasure)

    # Main loop
    while not exitflag:
        F_tilde_prev = np.copy(F_tilde)
        z_prev = np.copy(z)

        # Update steps
        A_prev = A
        a_prev = a
        a = 1 / (2 * L)
        A = A_prev + a

        F_x_prev = np.copy(F_store)

        for idx, block in enumerate(blocks):

            # Step 6
            F_tilde_prev[block] = F_tilde[block]
            F_tilde[block] = F_store[block]

            # Step 7
            F_bar[block] = F_tilde[block]

            # Step 8
            z[block] = z_prev[block] + a * F_bar[block]

            # Step 9
            x_prev[block] = x[block]
            x[block] = problem.g_func.prox_opr_block(x0[block] - z[block])

            Q += np.sum(x[block] - x_prev[block])
            p_ = p
            p = problem.operator_func.p(Q)
            dp_ = dp
            dp = problem.operator_func.dp(Q)
            problem.operator_func.func_map_block_update(F_store, x, p, p_, dp, dp_, block)
            
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