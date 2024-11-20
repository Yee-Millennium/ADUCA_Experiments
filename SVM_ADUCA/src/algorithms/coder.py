import numpy as np
import time
import numpy as np
import time
import logging
from src.algorithms.utils.exitcriterion import ExitCriterion, CheckExitCondition
from src.problems.GMVI_func import GMVIProblem
from src.algorithms.utils.results import Results, logresult

class CODERParams:
    def __init__(self, L, gamma):
        self.L = L
        self.gamma = gamma

def coder(problem: GMVIProblem, exitcriterion: ExitCriterion, parameters, x0=None):
    # Initialize parameters and variables
    L, gamma = parameters.L, parameters.gamma
    a, A = 0, 0
    x0 = np.zeros(problem.d) if x0 is None else x0
    # print(f"problem.d: {problem.d}")
    # print(f"!!! x0's shape: {x0.shape}")
    x, x_prev = x0.copy(), x0.copy()
    x_tilde_sum = np.zeros(problem.d)
    x_tilde = x0.copy()
    # print(f"!!! x0's shape: {x0.shape}")
    p = problem.operator_func.func_map(x0)
    p_prev = p.copy()
    z, z_prev, q = np.zeros(problem.d), np.zeros(problem.d), np.zeros(problem.d)
    m = problem.d  # Assuming each block is a coordinate

    # Initialization
    iteration = 0
    exitflag = False
    starttime = time.time()
    results = Results()  # Assuming Results is defined elsewhere
    init_optmeasure = problem.func_value(x0)
    logresult(results, 1, 0.0, init_optmeasure)

    # Main loop
    while not exitflag:
        x_prev[:] = x
        p_prev[:] = p
        z_prev[:] = z

        # Update steps
        A_prev = A
        a_prev = a
        a = (1 + gamma * A_prev) / (2 * L)
        A = A_prev + a

        F_x_prev = problem.operator_func.func_map(x_prev)
        for j in range(m):
            # Step 6
            p[j] = problem.operator_func.func_map_block(j + 1, x)

            # Step 7
            q[j] = p[j] + (a_prev / a) * (F_x_prev[j] - p_prev[j])

            # Step 8
            z[j] = z_prev[j] + a * q[j]

            # Step 9
            x[j] = problem.g_func.prox_opr_block(j + 1, x0[j] - z[j], A)

        x_tilde_sum += a * x
        iteration += m

        # Logging and exit condition
        if iteration % (m * exitcriterion.loggingfreq) == 0:
            x_tilde = x_tilde_sum / A
            elapsed_time = time.time() - starttime
            opt_measure = problem.func_value(x_tilde)
            print(f"Elapsed time: {elapsed_time}, Iteration: {iteration}, Opt measure: {opt_measure}")
            logresult(results, iteration, elapsed_time, opt_measure)
            exitflag = CheckExitCondition(exitcriterion, iteration, elapsed_time, opt_measure)

    #  x_tilde
    return results, x




def coder_linesearch(problem: GMVIProblem, exitcriterion: ExitCriterion, parameters, x0=None):
    # Initialize parameters and variables
    gamma = parameters.gamma
    L = 0.5
    L_ = 0.5
    a, A = 0, 0

    x0 = np.zeros(problem.d) if x0 is None else x0
    # print(f"problem.d: {problem.d}")
    # print(f"!!! x0's shape: {x0.shape}")
    x, x_prev = x0.copy(), x0.copy()
    x_tilde_sum = np.zeros(problem.d)
    x_tilde = x0.copy()
    # print(f"!!! x0's shape: {x0.shape}")
    p = problem.operator_func.func_map(x0)
    p_prev = p.copy()
    z, z_prev, q = np.zeros(problem.d), np.zeros(problem.d), np.zeros(problem.d)
    m = problem.d  # Assuming each block is a coordinate

    # Initialization
    iteration = 0
    exitflag = False
    starttime = time.time()
    results = Results()  # Assuming Results is defined elsewhere
    init_optmeasure = problem.func_value(x0)
    logresult(results, 1, 0.0, init_optmeasure)

    # Main loop
    while not exitflag:
        np.copyto(x_prev, x)
        np.copyto(p_prev, p)
        np.copyto(z_prev, z)
        F_x_prev = problem.operator_func.func_map(x_prev)
        A_prev = A
        a_prev = a
        L_ = L

        # Step 5
        L = L_ / 2

        # Step 6
        while{True}:
            # Step 7
            L = 2 * L

            temp_x = np.copy(x)

            # Step 8
            a = (1 + gamma * A_prev) / (2 * L)
            A = A_prev + a

            # Step 9
            for j in range(m):
                # Step 10
                p[j] = problem.operator_func.func_map_block(j + 1, x)

                # Step 11
                q[j] = p[j] + (a_prev / a) * (F_x_prev[j] - p_prev[j])

                # Step 12
                z[j] = z_prev[j] + a * q[j]

                # Step 13
                temp_x[j] = problem.g_func.prox_opr_block(j + 1, x0[j] - z[j], A)
            # Step 15
            norm_F_p = np.linalg.norm(problem.operator_func.func_map(temp_x) - p)
            norm_x = np.linalg.norm(temp_x - x_prev)
            if norm_F_p <= L * norm_x:
                x = np.copy(temp_x)
                break

        x_tilde_sum += a * x
        iteration += m

        # Logging and exit condition
        if iteration % (m * exitcriterion.loggingfreq) == 0:
            x_tilde = x_tilde_sum / A
            elapsed_time = time.time() - starttime
            opt_measure = problem.func_value(x_tilde)
            print(f"Elapsed time: {elapsed_time}, Iteration: {iteration}, Opt measure: {opt_measure}")
            logresult(results, iteration, elapsed_time, opt_measure)
            exitflag = CheckExitCondition(exitcriterion, iteration, elapsed_time, opt_measure)

    #  x_tilde
    return results, x