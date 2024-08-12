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
        x0 = np.random.rand(problem.d)
    
    x= np.copy(x0)
    x_= np.copy(x0)

    q= np.zeros_like(x0)
    p = problem.loss_func.grad(x0)
    p_ = np.copy(p)
    loss_func_grad_x = np.copy(p) 

    z = np.copy(x)
    z_ = np.copy(z)

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
        np.copyto(x_, x)
        np.copyto(p_, p)
        np.copyto(z_, z)
        
        # Step 4
        A_ = A
        a_ = a
        a = coder_stepsize(a_, A_, L, gamma)
        A = A_ + a

        # Step 5
        for j in range(0, m, 1):
            # Step 6
            if j == 0:
                loss_func_grad_x, b_A_x = problem.loss_func.grad_block_update(x)
                p[0] = loss_func_grad_x[0]
            else:
                p[j], b_A_x = problem.loss_func.grad_block_update(b_A_x, (j-1, x[j-1] - x_[j-1]), j)

            # Step 7
            q[j] = p[j] + (a_ / a) * (loss_func_grad_x[j] - p_[j])
            
            # Step 8
            z[j] = z_[j] + a * q[j]

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

def coder_linesearch(problem: CompositeFunc, exit_criterion: ExitCriterion, parameters, x0=None):
    # Init 
    gamma = parameters["gamma"]
    L = 1
    L_ = 1
    a, A = 0, 0

    if x0 is None:
        x0 = np.random.randn(problem.d)
    
    x= np.copy(x0)
    x_= np.copy(x0)
    z = np.copy(x)
    z_ = np.copy(z)

    p = problem.loss_func.grad(x0)
    p_ = np.copy(p)
    q= np.zeros_like(x0)
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
        np.copyto(x_, x)
        np.copyto(p_, p)
        np.copyto(z_, z)
        A_ = A
        a_ = a
        L_ = L

        # Step 5
        L = L_ / 2

        # Step 6
        while(True):
            # Step 7
            L = 2 * L

            temp_x = np.copy(x)

            # Step 8
            a = coder_stepsize(a_, A_, L, gamma)
            A = A_ + a

            # Step 9
            for j in range(0, m, 1):
                # Step 10
                if j == 0:
                    loss_func_grad_x, b_A_x = problem.loss_func.grad_block_update(x)
                    p[0] = loss_func_grad_x[0]
                else:
                    p[j], temp_b_A_x = problem.loss_func.grad_block_update(b_A_x, (j-1, temp_x[j-1] - x_[j-1]), j)

                # Step 11
                q[j] = p[j] + (a_ / a) * (loss_func_grad_x[j] - p_[j])
                
                # Step 12
                z[j] = z_[j] + a * q[j]

                # Step 13
                temp_x[j] = problem.reg_func.prox_opr_block(x0[j] - z[j], A)

            # Step 15
            if np.linalg.norm(problem.loss_func.grad(temp_x) - p) <= L * np.linalg.norm(temp_x - x_):
                x = np.copy(temp_x)
                break

        iteration += 1
        if iteration % exit_criterion.loggingfreq == 0:
            elapsed_time = time.time() - start_time
            opt_measure = problem.func_value(x)
            logging.info(f"elapsed_time: {elapsed_time}, iteration: {iteration}, opt_measure: {opt_measure}")
            logresult(results, iteration, elapsed_time, opt_measure)
            exit_flag = CheckExitCondition(exit_criterion, iteration, elapsed_time, opt_measure)

    return results, x
