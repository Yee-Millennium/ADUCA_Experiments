import numpy as np
import time
import logging
from src.algorithms.utils.exitcriterion import ExitCriterion, CheckExitCondition
from src.problems.composite_func import CompositeFunc
from src.algorithms.utils.results import Results, logresult

# Setup logging
logging.basicConfig(level=logging.INFO)
    

### three points extrapolation
def adapCODER(problem: CompositeFunc, exit_criterion: ExitCriterion, parameters, x_0=None):
    # Init of adapCODER
    L = parameters["L"]
    beta1, beta2, beta3 = parameters["beta1"], parameters["beta2"], parameters["beta3"]
    rho_1 = (1 + (beta1 * beta3)/(1-beta3)) * beta3
    rho_2 = (1-beta2) / beta3
    rho_3 = (1-beta2) / beta1 * beta2 * 10
    rho_4 = ((beta1 * beta2 *(1-beta2)) / ((1-beta3)*beta3)) **0.5

    theta = np.ones(3)

    if x_0 is None:
        x_0 = np.random.randn(problem.d)
    perturbation = np.random.uniform(-0.0001, 0.0001, size=x_0.shape)
    x_1 = x_0 + perturbation
    while np.array_equal( problem.grad(x_0),  problem.grad(x_1)):
        perturbation = np.random.uniform(-0.0001, 0.0001, size=x_0.shape)
        x_1 += perturbation

    x = np.zeros((4, problem.d))
    x[-2], x[-1] = x_0, x_1

    v = np.copy(x[-1])
    v_ = np.copy(v)

    loss_grad = np.zeros((3, problem.d))
    loss_grad[-2], loss_grad[-1] = problem.loss_func.grad(x_0), problem.loss_func.grad(x_1)

    b_A_x = np.zeros(problem.loss_func.n)
    m = problem.d  # Assume that each block is simply a coordinate for now

    p= np.zeros_like(loss_grad)
    for j in range(0, m, 1):
        if j == 0:
            loss_func_grad_x, b_A_x = problem.loss_func.grad_block_update(x[-2])
            p[-1][0] = loss_func_grad_x[0]
        else:
            p[-1][j], b_A_x = problem.loss_func.grad_block_update(b_A_x, (j-1, x[-1][j-1] - x[-2][j-1]), j)
    p[-2] = np.copy(loss_grad[-2])

    q = np.zeros(problem.d)

    step = np.linalg.norm(x[-1] - x[-2]) / np.linalg.norm(loss_grad[-1] - loss_grad[-2])
    a = step
    a_ = step
    A = 0

    theta = np.ones(3)


    # Stepsize selection function
    def adapCoder_stepsize(x, a, a_, theta, grad, tildeGrad, iter=3):
        # step_1 = rho_1 * a
        step_1 = rho_1 * a * 1.2

        ### we can heuristically decrease the constant on the denominator
        L_k = np.linalg.norm(grad[-1] - grad[-2]) / np.linalg.norm(x[-1] - x[-2])
        # step_2 = (rho_2**2 * theta[-1]) / (300 * a * L_k**2)
        step_2 = (rho_2**2 * theta[-1]) / (50 * a * L_k**2)

        if iter <= 1:
            step_3 = 10000
        else:
            L_hat_ = np.linalg.norm(tildeGrad[-2] - grad[-2]) / np.linalg.norm (x[-2] - x[-3])
            # step_3 = (rho_2**2 * theta[-2]) / (300 * a * L_hat_**2)
            step_3 = (rho_2**2 * theta[-2]) / (50 * a * L_hat_**2)
        
        # step_4 = rho_3 * theta[-1] * a
        step_4 = rho_3 * theta[-1] * a * 1.5

        L_hat = np.linalg.norm(tildeGrad[-1] - grad[-1]) / np.linalg.norm(x[-1] - x[-2])
        # step_5 = rho_4 * (theta[-1] / 10)**0.5 / L_hat
        step_5 = rho_4 * (theta[-1] / 5)**0.5 / L_hat

        if iter <= 2:
            step_6 = 10000
        else:
            L_hat__ = np.linalg.norm(tildeGrad[-3] - grad[-3]) / np.linalg.norm(x[-3] - x[-4])
            # step_6 = (rho_2**2 * theta[-1] * theta[-3]) / (60 * a_ * L_hat__**2)
            step_6 = (rho_2**2 * theta[-1] * theta[-3]) / (10 * a_ * L_hat__**2)
        
        step = min(step_1, step_2, step_3, step_4, step_5, step_6)
        # print(f" !!! Stepsize: {step}")
        return step



    # Run init
    iteration = 0
    exit_flag = False
    start_time = time.time()
    results = Results()
    init_opt_measure = problem.func_value(x_0)
    logresult(results, 1, 0.0, init_opt_measure)

    while not exit_flag:
        np.copyto(v_, v)

        # Step 4
        if iteration == 0:
            step = adapCoder_stepsize(x, a, a_, theta, loss_grad, p, iter=1)
        elif iteration == 1:
            step = adapCoder_stepsize(x, a, a_, theta, loss_grad, p, iter=2)
        else:
            step = adapCoder_stepsize(x, a, a_, theta, loss_grad, p)
        a_ = a
        a = step
        theta = np.append(theta[1:], a  / a_)

        # Step 5
        p = np.append(p[1:], np.zeros((1, problem.d)), axis=0)
        for j in range(0, m):
            # Step 7
            q[j] = p[-2][j] + (a_ / a) * (loss_grad[-2][j] - p[-3][j])
            
            # Step 8
            if j == 0:
                z_0 = problem.reg_func.prox_opr_block(v_[j] - a * q[j], a)
                x = np.append(x[1:], np.zeros((1, problem.d)), axis=0) 
                x[-1][0] = z_0
            else:
                x[-1][j] = problem.reg_func.prox_opr_block(v_[j] - a * q[j], a)

            # Step 9
            v[j] = beta1 * x[-1][j] + beta2 * x[-2][j] + beta3 * v_[j]
            
            # Step 10
            if j == 0:
                loss_func_grad_x, b_A_x = problem.loss_func.grad_block_update(x[-2])
                p[-1][0] = loss_func_grad_x[0]
            else:
                p[-1][j], b_A_x = problem.loss_func.grad_block_update(b_A_x, (j-1, x[-1][j-1] - x[-2][j-1]), j)


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






def adapCODER_two_point_extrapolation(problem: CompositeFunc, exit_criterion: ExitCriterion, parameters, x_0=None):
    # Init of adapCODER
    beta = parameters["beta"]
    rho_1 = (2 * beta * (1+beta)) / 3
    rho_2 = ((1+beta) / (84 * beta) )**0.5
    rho_3 = 1 / (49 * beta**2)

    theta = np.ones(3)

    if x_0 is None:
        x_0 = np.random.randn(problem.d)
    perturbation = np.random.uniform(-0.0001, 0.0001, size=x_0.shape)
    x_1 = x_0 + perturbation
    while np.array_equal( problem.grad(x_0),  problem.grad(x_1)):
        perturbation = np.random.uniform(-0.0001, 0.0001, size=x_0.shape)
        x_1 += perturbation

    x = np.zeros((4, problem.d))
    x[-2], x[-1] = x_0, x_1

    v = np.copy(x[-1])
    v_ = np.copy(v)

    loss_grad = np.zeros((3, problem.d))
    loss_grad[-2], loss_grad[-1] = problem.loss_func.grad(x_0), problem.loss_func.grad(x_1)

    b_A_x = np.zeros(problem.loss_func.n)
    m = problem.d  # Assume that each block is simply a coordinate for now

    p= np.zeros_like(loss_grad)
    for j in range(0, m, 1):
        if j == 0:
            loss_func_grad_x, b_A_x = problem.loss_func.grad_block_update(x[-2])
            p[-1][0] = loss_func_grad_x[0]
        else:
            p[-1][j], b_A_x = problem.loss_func.grad_block_update(b_A_x, (j-1, x[-1][j-1] - x[-2][j-1]), j)
    p[-2] = np.copy(loss_grad[-2])

    q = np.zeros(problem.d)

    step = np.linalg.norm(x[-1] - x[-2]) / np.linalg.norm(loss_grad[-1] - loss_grad[-2])
    a = step
    a_ = step
    A = 0

    theta = np.ones(3)


    # Stepsize selection function
    def adapCoder_stepsize(x, a, a_, theta, grad, tildeGrad, iter=3):
        # step_1 = rho_1 * a
        step_1 = rho_1 * a * 3

        ### we can heuristically decrease the constant on the denominator
        L_hat = np.linalg.norm(tildeGrad[-1] - grad[-1]) / np.linalg.norm(x[-1] - x[-2])
        # step_2 = rho_2 * (theta[-1])**0.5 / L_hat
        step_2 = rho_2 * (theta[-1])**0.5 / L_hat * 8

        L_k = np.linalg.norm(grad[-1] - grad[-2]) / np.linalg.norm(x[-1] - x[-2])
        # step_3 = rho_3 * theta[-1] / (a * L_k**2)
        step_3 = rho_3 * theta[-1] / (a * L_k**2) * 8

        if iter <= 1:
            step_4 = 10000
        else:
            L_hat_ = np.linalg.norm(tildeGrad[-2] - grad[-2]) / np.linalg.norm (x[-2] - x[-3])
            # step_4 = rho_3 * theta[-2] / (a * L_hat_**2)
            step_4 = rho_3 * theta[-2] / (a * L_hat_**2) * 8

        if iter <= 2:
            step_5 = 10000
        else:
            L_hat__ = np.linalg.norm(tildeGrad[-3] - grad[-3]) / np.linalg.norm(x[-3] - x[-4])
            # step_5 = rho_3 * (theta[-3] * theta[-1]**2) / (a * L_hat__**2)
            step_5 = rho_3 * (theta[-3] * theta[-1]**2) / (a * L_hat__**2) * 8
        
        step = min(step_1, step_2, step_3, step_4, step_5)
        # print(f" !!! Stepsize: {step}")
        return step



    # Run init
    iteration = 0
    exit_flag = False
    start_time = time.time()
    results = Results()
    init_opt_measure = problem.func_value(x_0)
    logresult(results, 1, 0.0, init_opt_measure)

    while not exit_flag:
        np.copyto(v_, v)

        # Step 4
        if iteration == 0:
            step = adapCoder_stepsize(x, a, a_, theta, loss_grad, p, iter=1)
        elif iteration == 1:
            step = adapCoder_stepsize(x, a, a_, theta, loss_grad, p, iter=2)
        else:
            step = adapCoder_stepsize(x, a, a_, theta, loss_grad, p)
        a_ = a
        a = step
        theta = np.append(theta[1:], a  / a_)

        # Step 5
        p = np.append(p[1:], np.zeros((1, problem.d)), axis=0)
        for j in range(0, m):
            # Step 7
            q[j] = p[-2][j] + (a_ / a) * (loss_grad[-2][j] - p[-3][j])
            
            # Step 8
            if j == 0:
                z_0 = problem.reg_func.prox_opr_block(v_[j] - a * q[j], a)
                x = np.append(x[1:], np.zeros((1, problem.d)), axis=0) 
                x[-1][0] = z_0
            else:
                x[-1][j] = problem.reg_func.prox_opr_block(v_[j] - a * q[j], a)

            # Step 9
            v[j] = (1-beta) * x[-1][j] + beta * v_[j]
            
            # Step 10
            if j == 0:
                loss_func_grad_x, b_A_x = problem.loss_func.grad_block_update(x[-2])
                p[-1][0] = loss_func_grad_x[0]
            else:
                p[-1][j], b_A_x = problem.loss_func.grad_block_update(b_A_x, (j-1, x[-1][j-1] - x[-2][j-1]), j)


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