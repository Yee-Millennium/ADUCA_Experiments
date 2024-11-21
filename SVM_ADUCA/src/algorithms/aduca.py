import numpy as np
import time
import logging
from src.algorithms.utils.exitcriterion import ExitCriterion, CheckExitCondition
from src.problems.GMVI_func import GMVIProblem
from src.algorithms.utils.results import Results, logresult
from src.algorithms.utils.helper import construct_block_range

# Setup logging
logging.basicConfig(level=logging.INFO)

def aduca(problem: GMVIProblem, exit_criterion: ExitCriterion, parameters, u_0=None):
    # Init of adapCODER
    d = problem.operator_func.d
    n = problem.operator_func.n
    beta = parameters["beta"]
    c = parameters["c"]

    block_size = parameters['block_size']
    blocks = construct_block_range(dimension=(d+n), block_size=block_size)
    m = len(blocks)

    phi_1 = 2 * c * beta * (1+beta)
    phi_2 = (((1-2*c) * (1 + beta))  /  7 * beta)**0.5 * 0.5
    phi_3 = 1 / (7 * beta)

    alpha = min(phi_1, 4 / (7 * beta * (1+beta) * (1-2 * c) ))

    def aduca_stepsize(u, u_, a, a_, F, F_, F_tilde):
        step_1 = alpha * a 

        ### we can heuristically scale the step
        L_hat_k = np.linalg.norm(F - F_tilde) / (np.linalg.norm(u-u_)) 
        if L_hat_k == 0:
            step_2 = 100
        else:    
            step_2 = (phi_2 / L_hat_k) * (a / a_)**0.5    

        L_k = np.linalg.norm(F - F_) / (np.linalg.norm(u - u_)) 
        # print(f"!!! L_k: {L_k}")
        if L_k == 0:
            step_3 = 100
        else:
            step_3 = (phi_3 ** 2) / (a_ * L_k**2)  
            # print(f"!!! step_3: {step_3}")
        
        step = min(step_1, step_2, step_3)
        # print(f" !!! Stepsize: {step}")
        return step

    a = 0
    a_ = 0
    A = 0

    u = np.zeros(problem.d)
    u_ = np.copy(u)
    # u_new = np.copy(u)
    u_hat = np.copy(u)
    v = np.zeros(problem.d)
    v_ = np.zeros(problem.d)

    F = np.zeros(problem.d)
    F_ = np.zeros(problem.d)
    F_tilde = np.zeros(problem.d)
    F_tilde_ = np.zeros(problem.d)
    F_bar = np.copy(F_tilde)

    if u_0 is None:
        u_0 = np.full(shape=problem.d, fill_value=-0.0001)
        # u_0 = np.zeros(problem.d)
    u_1 = u_0
    
    F_0 = problem.operator_func.func_map(u_0)
    F_tilde_0 = np.copy(F_0)
    F_tilde_1 = np.copy(F_tilde_0)

    ## line-search for the first step
    a_0 = 10 * phi_1
    while True:
        F_store = np.copy(F_0)
        a_0 = a_0 / 2
        u_1 = problem.g_func.prox_opr(u_0 - a_0 * F_0, a_0, d)

        for block in blocks:
            F_tilde_1[block] = F_store[block]
            F_store = problem.operator_func.func_map_block_update(F_store, u_1[block], u_0[block], block)
        
        F_1 = np.copy(F_store)
        norm_F = np.linalg.norm((F_1 - F_0))
        norm_F_tilde = np.linalg.norm((F_1 - F_tilde_1))
        norm_u = np.linalg.norm((u_1 - u_0))

        # print(f"phi_2: {phi_2}")
        # print(f"a_0: {a_0}")
        if (a_0 * norm_F <= phi_2 * norm_u) and (a_0 * norm_F_tilde <= phi_2 * norm_u):
            break

    a_ = a_0
    a = a_0
    A = 0

    u = np.copy(u_1)
    u_ = np.copy(u_0)
    v_ = np.copy(u_)
    u_hat = A * u_

    F = np.copy(F_1)
    F_ = np.copy(F_0)
    F_tilde = np.copy(F_tilde_1)
    F_tilde_ = np.copy(F_tilde_0)
    F_bar = np.zeros(problem.d)

    # Run init
    iteration = 0
    exit_flag = False
    start_time = time.time()
    results = Results()
    init_opt_measure = problem.func_value(u_)
    logresult(results, 1, 0.0, init_opt_measure)

    while not exit_flag:

        # Step 6
        step = aduca_stepsize(u, u_, a, a_, F, F_, F_tilde)
        a_ = a
        a = step
        A += a

        # Step 7
        for block in blocks:

            # Step 8
            F_bar[block] = F_tilde[block] + (a_ / a) * (F_[block] - F_tilde_[block])
            
            # Step 9
            v[block] = (1-beta) * u[block] + beta * v_[block]

            # Step 10
            u_[block] = u[block]
            u[block] = problem.g_func.prox_opr_block(block ,v[block] - a * F_bar[block], a)

            # Step 11
            F_tilde_[block] = F_tilde[block]
            F_tilde[block] = F_store[block]
            F_store = problem.operator_func.func_map_block_update(F_store, u[block], u_[block], block) 

        np.copyto(F_, F)
        F = np.copy(F_store)
        # print(f"If F equal to F_tilde")
        np.copyto(v_, v)

        # print(f"!!! (a / A): {(a / A)}")
        u_hat = ((A - a) * u_hat / A) + (a*u_ / A)


        iteration += m
        if iteration % (m *  exit_criterion.loggingfreq) == 0:
            elapsed_time = time.time() - start_time
            opt_measure = problem.func_value(u_hat)
            logging.info(f"elapsed_time: {elapsed_time}, iteration: {iteration}, opt_measure: {opt_measure}")
            logresult(results, iteration, elapsed_time, opt_measure)
            exit_flag = CheckExitCondition(exit_criterion, iteration, elapsed_time, opt_measure)

    return results, u






### aduca with restart
def aduca_restart(problem: GMVIProblem, exit_criterion: ExitCriterion, parameters, u_0=None):
    # Init of adapCODER
    d = problem.operator_func.d
    n = problem.operator_func.n
    beta = parameters["beta"]
    c = parameters["c"]
    restartfreq = parameters["restartfreq"]

    block_size = parameters['block_size']
    blocks = construct_block_range(dimension=(d+n), block_size=block_size)
    m= len(blocks)

    phi_1 = 2 * c * beta * (1+beta)
    phi_2 = (((1-2*c) * (1 + beta))  /  7 * beta)**0.5 * 0.5
    phi_3 = 1 / (7 * beta)
    alpha = min(phi_1, 4 / (7 * beta * (1+beta) * (1-2 * c) ))
    # Stepsize selection function
    def aduca_stepsize(u, u_, a, a_, F, F_, F_tilde):
        step_1 = alpha * a 

        ### we can heuristically scale the step
        L_hat_k = np.linalg.norm(F - F_tilde) / (np.linalg.norm(u-u_)) 
        if L_hat_k == 0:
            step_2 = 1000
        else:    
            step_2 = (phi_2 / L_hat_k) * (a / a_)**0.5    

        L_k = np.linalg.norm(F - F_) / (np.linalg.norm(u - u_)) 
        # print(f"!!! L_k: {L_k}")
        if L_k == 0:
            step_3 = 1000
        else:
            step_3 = (phi_3 ** 2) / (a_ * L_k**2)  
            # print(f"!!! step_3: {step_3}")
        
        step = min(step_1, step_2, step_3)
        # print(f" !!! Stepsize: {step}")
        return step

    a = 0
    a_ = 0
    A = 0

    if u_0 is None:
        u_0 = np.full(shape=problem.d, fill_value=-0.0001)
        # u_0 = np.zeros(problem.d)
    u_ = np.copy(u_0)
    u_hat = np.zeros(problem.d)
    v = np.zeros(problem.d)
    v_ = np.zeros(problem.d)

    F = np.zeros(problem.d)
    F_ = np.zeros(problem.d)
    F_tilde = np.zeros(problem.d)
    F_tilde_ = np.zeros(problem.d)
    F_bar = np.copy(F_tilde)


    outer_k = 0
    exit_flag = False
    start_time = time.time()
    results = Results()
    init_opt_measure = problem.func_value(u_)
    logresult(results, 1, 0.0, init_opt_measure)

    while not exit_flag:
        u = np.copy(u_0)
        u_ = np.copy(u_0)
    
        F_0 = problem.operator_func.func_map(u_0)
        F_tilde_0 = np.copy(F_0)
        F_tilde_1 = np.copy(F_tilde_0)

        ## line-search for the first step
        a_0 = 10 * phi_1
        while True:
            F_store = np.copy(F_0)
            a_0 = a_0 / 2
            u_1 = problem.g_func.prox_opr(u_0 - a_0 * F_0, a_0, d)

            for block in blocks:
                F_tilde_1[block] = F_store[block]
                F_store = problem.operator_func.func_map_block_update(F_store, u_1[block], u_0[block], block)
            
            F_1 = np.copy(F_store)
            norm_F = np.linalg.norm((F_1 - F_0))
            norm_F_tilde = np.linalg.norm((F_1 - F_tilde_1))
            norm_u = np.linalg.norm((u_1 - u_0))

            # print(f"phi_2: {phi_2}")
            # print(f"a_0: {a_0}")
            if (a_0 * norm_F <= phi_2 * norm_u) and (a_0 * norm_F_tilde <= phi_2 * norm_u):
                break

        a_ = a_0
        a = a_0
        A = 0

        u = np.copy(u_1)
        u_ = np.copy(u_0)
        v_ = np.copy(u_)
        u_hat = A * u_

        F = np.copy(F_1)
        F_ = np.copy(F_0)
        F_tilde = np.copy(F_tilde_1)
        F_tilde_ = np.copy(F_tilde_0)
        F_bar = np.zeros(problem.d)
        

        k = 0
        restart_flag = False
        while not exit_flag and not restart_flag:

            # Step 6
            step = aduca_stepsize(u, u_, a, a_, F, F_, F_tilde)
            a_ = a
            a = step
            A += a

            for block in blocks:
                # Step 8
                F_bar[block] = F_tilde[block] + (a_ / a) * (F_[block] - F_tilde_[block])
                
                # Step 9
                v[block] = (1-beta) * u[block] + beta * v_[block]

                # Step 10
                u_[block] = u[block]
                u[block] = problem.g_func.prox_opr_block(block ,v[block] - a * F_bar[block], a)

                # Step 11
                F_tilde_[block] = F_tilde[block]
                F_tilde[block] = F_store[block]
                F_store = problem.operator_func.func_map_block_update(F_store, u[block], u_[block], block) 
            np.copyto(F_, F)
            F = np.copy(F_store)
            # print(f"If F equal to F_tilde")
            np.copyto(v_, v)

            # print(f"!!! (a / A): {(a / A)}")
            u_hat = ((A - a) * u_hat / A) + (a*u_ / A)

            # Increment iteration counters
            outer_k += m
            k += m
            
            if outer_k % (m *  exit_criterion.loggingfreq) == 0:
                # Compute averaged variables
                step = aduca_stepsize(u,u_,a,a_,F,F_,F_tilde)
                a_ = a
                a = step
                A += a      
                u_hat = ((A - a) * u_hat / A) + (a*u / A)
                elapsed_time = time.time() - start_time
                opt_measure = problem.func_value(u_hat)
                logging.info(f"elapsed_time: {elapsed_time}, iteration: {outer_k}, opt_measure: {opt_measure}")
                logresult(results, outer_k, elapsed_time, opt_measure)
                exit_flag = CheckExitCondition(exit_criterion, outer_k, elapsed_time, opt_measure)
                if exit_flag:
                    break
            
            if (k >= restartfreq):
                elapsed_time = time.time() - start_time
                opt_measure = problem.func_value(u_hat)
                logging.info("<===== RESTARTING")
                logging.info(f"k: {k}")
                logging.info(f"elapsed_time: {elapsed_time}, iteration: {outer_k}, opt_measure: {opt_measure}")
                # Compute averaged variables
                step = aduca_stepsize(u,u_,a,a_,F,F_,F_tilde)
                a_ = a
                a = step
                A += a      
                u_hat = ((A - a) * u_hat / A) + (a*u / A)
                u_0 = np.copy(u_hat)
                # Update x0 and y0 for restart
                init_opt_measure = opt_measure
                restart_flag = True
                break
            
    return results, u




