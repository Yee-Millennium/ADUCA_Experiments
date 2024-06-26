import numpy as np
import time
import logging
from src.algorithms.utils.exitcriterion import ExitCriterion, CheckExitCondition
from src.problems.composite_func import CompositeFunc
from src.algorithms.utils.results import Results, logresult

# Setup logging
logging.basicConfig(level=logging.INFO)
    

### This is the version of lambda = 1
def adapCoder(problem: CompositeFunc, exit_criterion: ExitCriterion, parameters, x_0=None):
    # Init of adapCODER
    L = parameters["L"]
    beta1, beta2, beta3 = parameters["beta1"], parameters["beta2"], parameters["beta3"]
    rho_1 = (1 + beta1 * beta3 /(1-beta3)) * beta3
    rho_2 = (1-beta2)**2 / beta3**2
    a = np.ones(4)
    A = 0

    if x_0 is None:
        x_0 = np.zeros(problem.d)
    x_1 = np.copy(x_0)
    x_1[0] = 1
    count = 0
    while np.array_equal( problem.grad(x_0),  problem.grad(x_1)):
        count += 1
        x_1[count] = 1 
    x = np.zeros((4, problem.d))
    x[-2], x[-1] = x_0, x_1
    x_sum = np.zeros(problem.d)
    x_hat = np.zeros(problem.d)
    v = np.zeros(problem.d)
    v_minus_1 = np.copy(v)

    loss_grad = np.zeros((3, problem.d))
    loss_grad[-3], loss_grad[-2], loss_grad[-1] = problem.loss_func.grad(x_0), problem.loss_func.grad(x_0), problem.loss_func.grad(x_1)
    p= np.copy(loss_grad)
    q = problem.loss_func.grad(x_0)

    b_A_x = np.zeros(problem.loss_func.n)
    m = problem.d  # Assume that each block is simply a coordinate for now


    # Stepsize selection function
    def adapCoder_stepsize(x, a_step, grad, tildeGrad):
        step_vec = np.zeros(6)
        step_1 = rho_1 * a_step[-1]
        step_vec[0] = step_1

        delta_x = x[-1] - x[-2]
        delta_grad = grad[-1] - grad[-2]
        if np.any(delta_grad):
            inv_Lsquare = (delta_x @ delta_x) / (delta_grad @ delta_grad)
            step_2 = rho_2 / (300 * a_step[-2] ) * inv_Lsquare
            step_vec[1] = step_2
        else:
            step_vec[1] = step_1
        
        delta_x_minus_1 = x[-2] - x[-3]
        delta_tildeGrad_minus_1 = grad[-2] - tildeGrad[-2]
        if np.any(delta_tildeGrad_minus_1):
            invHat_Lsquare_minus_1 =  (delta_x_minus_1 @ delta_x_minus_1) / (delta_tildeGrad_minus_1 @ delta_tildeGrad_minus_1)
            step_3 = rho_2 * a_step[-2] / (300 * a_step[-3] * a_step[-1]) * invHat_Lsquare_minus_1
            step_vec[2] = step_3
        else:
            step_vec[2] = step_1
        
        step_4 = (a_step[-1]**2 *(1-beta2)) / (10 * a_step[-2] * beta1 * beta2)
        step_vec[3] = step_4

        delta_tildeGrad = grad[-1] - tildeGrad[-1]
        if np.any(delta_tildeGrad):
            invHat_L = ((delta_x @ delta_x) / (delta_tildeGrad @ delta_tildeGrad))**0.5
            step_5 = 1/invHat_L * ((a_step[-1]*beta1*beta2*(1-beta2)) / (10*a_step[-2]*beta3*(1-beta3))) **0.5
            step_vec[4] = step_5
        else:
            step_vec[4] = step_1

        delta_x_minus_2 = x[-3] - x[-4]
        delta_tildeGrad_minus_2 = grad[-3] - tildeGrad[-3]
        if np.any(delta_tildeGrad_minus_2):
            invHat_Lsquare_minus_2 = (delta_x_minus_2 @ delta_x_minus_2) / (delta_tildeGrad_minus_2 @ delta_tildeGrad_minus_2)
            step_6 = rho_2 * (  (a_step[-3] * a_step[-1]) / (60 * a_step[-4] *a_step[-2]**2) ) * invHat_Lsquare_minus_2
            step_vec[5] = step_6
        else:
            step_vec[5] = step_1
        
        min_step = min(step_vec)
        if min_step > 1/L:
            return min_step
        else:
            return 1/L



    # Run init
    iteration = 0
    exit_flag = False
    start_time = time.time()
    results = Results()
    init_opt_measure = problem.func_value(x_0)
    logresult(results, 1, 0.0, init_opt_measure)

    while not exit_flag:
        np.copyto(v_minus_1, v)
        
        # Step 4
        step = adapCoder_stepsize(x, a, loss_grad, p)
        a = np.append(a[1:], step)

        # Step 5
        for j in range(0, m):
            # Step 6
            if j == 0:
                p = np.append(p[1:], np.array([loss_grad[-2]]), axis=0)
            else:
                p[-1][j], b_A_x = problem.loss_func.grad_block_update(b_A_x, (j-1, x[-1][j-1] - x[-2][j-1]), j)

            # Step 7
            q[j] = p[-1][j] + (a[-2] / a[-1]) * (loss_grad[-2][j] - p[-2][j])
            
            # Step 8
            if j == 0:
                z_0 = problem.reg_func.prox_opr_block(v_minus_1[j] - step * q[j], step)
                x = np.append(x[1:], np.zeros((1, problem.d)), axis=0) 
                x[-1][0] = z_0
            else:
                x[-1][j] = problem.reg_func.prox_opr_block(v_minus_1[j] - step * q[j], step)

            # Step 9
            v[j] = beta1 * x[-1][j] + beta2 * x[-2][j] + beta3 * v_minus_1[j]
            
        loss_grad = np.append(loss_grad[1:], np.array([problem.loss_func.grad(x[-1])]), axis=0)

        ### Compute x_hat
        # A += step
        # x_sum += step * x[-2]
        # x_hat = 1/A * x_sum

        iteration += 1
        if iteration % exit_criterion.loggingfreq == 0:
            elapsed_time = time.time() - start_time
            opt_measure = problem.func_value(x[-1])
            logging.info(f"elapsed_time: {elapsed_time}, iteration: {iteration}, opt_measure: {opt_measure}")
            logresult(results, iteration, elapsed_time, opt_measure)
            exit_flag = CheckExitCondition(exit_criterion, iteration, elapsed_time, opt_measure)

    return results, x