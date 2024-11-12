import numpy as np
import time
import logging
from src.algorithms.utils.exitcriterion import ExitCriterion, CheckExitCondition
from src.problems.composite_func import CompositeFunc
from src.algorithms.utils.results import Results, logresult

# Setup logging
logging.basicConfig(level=logging.INFO)


def aduca(problem: CompositeFunc, exit_criterion: ExitCriterion, parameters, x_0=None):
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
    def aduca_stepsize(x, a, a_, theta, grad, tildeGrad, iter=3):
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
            step = aduca_stepsize(x, a, a_, theta, loss_grad, p, iter=1)
        elif iteration == 1:
            step = aduca_stepsize(x, a, a_, theta, loss_grad, p, iter=2)
        else:
            step = aduca_stepsize(x, a, a_, theta, loss_grad, p)
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


# def aduca_lazy(problem: CompositeFunc, exit_criterion: ExitCriterion, parameters, u_0=None):
#     # Init of adapCODER
#     beta = parameters["beta"]
#     c = parameters["c"]
#     block_size = parameters['block_size']

#     phi_1 = 2 * c * beta * (1+beta)
#     phi_2 = (((1-2*c) * (1 + beta))  /  7 * beta)  ** 0.5 * 0.5
#     phi_3 = 1 / (7 * beta)

#     alpha = min(phi_1, (phi_2 / phi_3)**2)

#     a = 0
#     a_ = 0
#     A = 0

#     u = np.zeros(problem.d)
#     u_ = np.copy(u)
#     u_new = np.copy(u)
#     # u_hat = np.copy(u)
#     v = np.zeros(problem.d)
#     v_ = np.zeros(problem.d)

#     F = np.zeros(problem.d)
#     F_ = np.zeros(problem.d)
#     F_tilde = np.zeros(problem.d)
#     F_tilde_ = np.zeros(problem.d)
#     F_bar = np.copy(F_tilde)

#     if u_0 is None:
#         u_0 = np.random.randn(problem.d)
#     u_1 = u_0
    
#     F_0 = problem.loss_func.grad(u_0)
#     F_tilde_0 = np.copy(F_0)
#     ## line-search for the first step
#     a_0 = 2 * phi_1
#     while True:
#         a_0 = a_0 / 2
#         u_1 = problem.reg_func.prox_opr(u_0 - a_0 * F_0, a_0)
        
#         F_1 = problem.loss_func.grad(u_1)
#         F_tilde_1 = np.copy(F_tilde_0)
#         for j in range(0, problem.d, 1):
#             if j == 0:
#                 F_tilde_1[0], b_A_x = problem.loss_func.grad_block_update(u_0, 0)
#             else:
#                 F_tilde_1[j], b_A_x = problem.loss_func.grad_block_update(b_A_x, (j-1, u_1[j-1] - u_0[j-1]), j)
        
#         norm_F = np.linalg.norm((F_1 - F_0))
#         norm_F_tilde = np.linalg.norm((F_1 - F_tilde_1))
#         norm_u = np.linalg.norm((u_1 - u_0))

#         # print(f"phi_2: {phi_2}")
#         # print(f"a_0: {a_0}")
#         if (a_0 * norm_F <= phi_2 * norm_u) and (a_0 * norm_F_tilde <= phi_2 * norm_u):
#             break

#     a_ = a_0
#     a = a_0
#     A += a_0

#     u = np.copy(u_1)
#     u_ = np.copy(u_0)
#     # u_hat = u_

#     F = np.copy(F_1)
#     F_ = np.copy(F_0)
#     F_tilde = np.copy(F_tilde_1)
#     F_tilde = np.copy(F_tilde_0)
    


#     # Stepsize selection function
#     def aduca_stepsize(u, u_, a, a_, F, F_, F_tilde):
#         # step_1 = alpha * a
#         step_1 = alpha * a * 3

#         ### we can heuristically scale the step
#         L_hat_k = np.linalg.norm(F - F_tilde) / np.linalg.norm(u-u_)
#         # step_2 = (phi_2 / L_hat) * (a / a_)**0.5
#         if L_hat_k == 0:
#             step_2 = 1000000
#         else:    
#             step_2 = (phi_2 / L_hat_k) * (a / a_)**0.5    * 8

#         L_k = np.linalg.norm(F - F_) / np.linalg.norm(u - u_)
#         if L_k == 0:
#             step_3 = 1000000
#         else:
#             step_3 = (phi_3 ** 2) / (a_ * L_k**2)     * 8
        
#         step = min(step_1, step_2, step_3)
#         # print(f" !!! Stepsize: {step}")
#         return step


#     # Run init
#     iteration = 0
#     exit_flag = False
#     start_time = time.time()
#     results = Results()
#     init_opt_measure = problem.func_value(u_)
#     logresult(results, 1, 0.0, init_opt_measure)

#     while not exit_flag:

#         # Step 5
#         step = aduca_stepsize(u, u_, a, a_, F, F_, F_tilde)
#         a_ = a
#         a = step
#         A += a

#         # Step 8
#         for j in range(0, problem.d):
#             # Step 9
#             if j == 0:
#                 F_tilde[0] = F[0]
#             else:
#                 F_tilde[j], b_A_x = problem.loss_func.grad_block_update(b_A_x, (j-1, u[j-1] - u_[j-1]), j)
#             # Step 10
#             F_bar[j] = F_tilde[j] + (a_ / a) * (F_[j] - F_tilde_[j] )
            
#             # Step 11
#             v[j] = (1-beta) * u[j] + beta * v_[j]

#             # Step 12
#             u_new[j] = problem.reg_func.prox_opr_block(v_[j] - a * F_bar[j], a)
        
#         np.copyto(F_, F)
#         F, b_A_x = problem.loss_func.grad_block_update(u)
#         np.copyto(F_tilde_, F_tilde)
#         np.copyto(u_, u)
#         np.copyto(u, u_new)
#         np.copyto(v_, v)

#         # u_hat = ((A - a) * u_hat / A) + (a*u / A)


#         iteration += 1
#         if iteration % exit_criterion.loggingfreq == 0:
#             elapsed_time = time.time() - start_time
#             opt_measure = problem.func_value(u)
#             logging.info(f"elapsed_time: {elapsed_time}, iteration: {iteration}, opt_measure: {opt_measure}")
#             logresult(results, iteration, elapsed_time, opt_measure)
#             exit_flag = CheckExitCondition(exit_criterion, iteration, elapsed_time, opt_measure)

#     return results, u

def aduca_lazy(problem: CompositeFunc, exit_criterion: ExitCriterion, parameters, u_0=None):
    # Init of adapCODER
    beta = parameters["beta"]
    c = parameters["c"]
    block_size = parameters['block_size']
    block_number = problem.d // block_size 
    last_block_size = problem.d - (block_number-1) * block_size
    logging.info(f"block_number: {block_number}; block_size: {block_size}; last_block_size: {last_block_size}")

    phi_1 = 2 * c * beta * (1+beta)
    phi_2 = (((1-2*c) * (1 + beta))  /  7 * beta)  ** 0.5 * 0.5
    phi_3 = 1 / (7 * beta)

    alpha = min(phi_1, (phi_2 / phi_3)**2)

    a = 0
    a_ = 0
    A = 0

    u = np.zeros(problem.d)
    u_ = np.copy(u)
    u_new = np.copy(u)
    # u_hat = np.copy(u)
    v = np.zeros(problem.d)
    v_ = np.zeros(problem.d)

    F = np.zeros(problem.d)
    F_ = np.zeros(problem.d)
    F_tilde = np.zeros(problem.d)
    F_tilde_ = np.zeros(problem.d)
    F_bar = np.copy(F_tilde)

    if u_0 is None:
        u_0 = np.random.randn(problem.d)
    u_1 = u_0
    
    F_0 = problem.loss_func.grad(u_0)
    F_tilde_0 = np.copy(F_0)
    ## line-search for the first step
    a_0 = 2 * phi_1
    while True:
        a_0 = a_0 / 2
        u_1 = problem.reg_func.prox_opr(u_0 - a_0 * F_0, a_0)
        
        F_1 = problem.loss_func.grad(u_1)
        F_tilde_1 = np.copy(F_tilde_0)
        for j in range(0, block_number, 1):
            if j == 0:
                p, b_A_x = problem.loss_func.grad_block_update(u_0)
                F_tilde_1[0:(block_size)] = p[0:(block_size)]
            elif j != (block_number - 1):
                # print(f"block_size for the initialization: {j*block_size}")
                F_tilde_1[j*block_size:((j+1)*block_size)], b_A_x = problem.loss_func.grad_block_update(b_A_x, u_1[(j-1)*block_size:(j*block_size)] - u_0[(j-1)*block_size:(j*block_size)], j, block_size, block_size)
            else:
                F_tilde_1[j*block_size: (problem.d)], b_A_x = problem.loss_func.grad_block_update(b_A_x, u_1[(j-1)*block_size:(j*block_size)] - u_0[(j-1)*block_size:(j*block_size) ], j, block_size, last_block_size)
        
        norm_F = np.linalg.norm((F_1 - F_0))
        norm_F_tilde = np.linalg.norm((F_1 - F_tilde_1))
        norm_u = np.linalg.norm((u_1 - u_0))
        # print(f"!!! Norm of F: {norm_F}")
        # print(f"!!! Norm of F_tilde: {norm_F_tilde}")
        # print(f"!!! Norm of u: {norm_u}")
        # print(f"phi_2: {phi_2}")
        # print(f"a_0: {a_0}")
        if (a_0 * norm_F <= phi_2 * norm_u) and (a_0 * norm_F_tilde <= phi_2 * norm_u):
            break
        if (a_0 <= 1):
            break

    a_ = a_0
    a = a_0
    A += a_0

    u = np.copy(u_1)
    u_ = np.copy(u_0)
    # u_hat = u_

    F = np.copy(F_1)
    F_ = np.copy(F_0)
    F_tilde = np.copy(F_tilde_1)
    F_tilde = np.copy(F_tilde_0)
    


    # Stepsize selection function
    def aduca_stepsize(u, u_, a, a_, F, F_, F_tilde):
        # step_1 = alpha * a
        step_1 = alpha * a * 3

        ### we can heuristically scale the step
        L_hat_k = np.linalg.norm(F - F_tilde) / np.linalg.norm(u-u_)
        # step_2 = (phi_2 / L_hat) * (a / a_)**0.5
        if L_hat_k == 0:
            step_2 = 1000000
        else:    
            # print(f"a: {a}")
            # print(f"a_: {a_}")
            step_2 = (phi_2 / L_hat_k) * (a / a_)**0.5    * 6

        L_k = np.linalg.norm(F - F_) / np.linalg.norm(u - u_)
        if L_k == 0:
            step_3 = 1000000
        else:
            step_3 = (phi_3 ** 2) / (a_ * L_k**2)     * 6
        
        step = min(step_1, step_2, step_3)
        # print(f"!!! Stepsizes: {step_1, step_2, step_3}")
        # print(f" !!! Stepsize: {step}")
        return step


    # Run init
    iteration = 0
    exit_flag = False
    start_time = time.time()
    results = Results()
    init_opt_measure = problem.func_value(u_)
    logresult(results, 1, 0.0, init_opt_measure)

    while not exit_flag:

        # Step 5
        step = aduca_stepsize(u, u_, a, a_, F, F_, F_tilde)
        a_ = a
        a = step
        # A += a

        # Step 8
        for j in range(0, block_number, 1):
            if j == 0:
                F_tilde[0:(block_size)] = F[0:(block_size)]
                # Step 10
                F_bar[0:(block_size)] = F_tilde[0:(block_size)] + (a_ / a) * (F_[0:(block_size)] - F_tilde_[0:(block_size)] )
                
                # Step 11
                v[0:(block_size)] = (1-beta) * u[0:(block_size)] + beta * v_[0:(block_size)]

                # Step 12
                u_new[0:(block_size)] = problem.reg_func.prox_opr(v_[0:(block_size)] - a * F_bar[0:(block_size)], a)

            elif j != block_number - 1:
                F_tilde[j*block_size:((j+1)*block_size)], b_A_x = problem.loss_func.grad_block_update(b_A_x, u_1[(j-1)*block_size:(j*block_size)] - u_0[(j-1)*block_size:(j*block_size)], j, block_size, block_size)
                # Step 10
                F_bar[j*block_size:((j+1)*block_size)] = F_tilde[j*block_size:((j+1)*block_size)] + (a_ / a) * (F_[j*block_size:((j+1)*block_size)] - F_tilde_[j*block_size:((j+1)*block_size)] )
                
                # Step 11
                v[j*block_size:((j+1)*block_size)] = (1-beta) * u[j*block_size:((j+1)*block_size)] + beta * v_[j*block_size:((j+1)*block_size)]

                # Step 12
                u_new[j*block_size:((j+1)*block_size)] = problem.reg_func.prox_opr(v_[j*block_size:((j+1)*block_size)] - a * F_bar[j*block_size:((j+1)*block_size)], a)

            else:
                F_tilde[j*block_size: (problem.d)], b_A_x = problem.loss_func.grad_block_update(b_A_x, u_1[(j-1)*block_size:(j*block_size)] - u_0[(j-1)*block_size:(j*block_size)], j, block_size, last_block_size)
                # Step 10
                F_bar[j*block_size: (problem.d)] = F_tilde[j*block_size: (problem.d)] + (a_ / a) * (F_[j*block_size: (problem.d)] - F_tilde_[j*block_size: (problem.d)] )
                
                # Step 11
                v[j*block_size: (problem.d)] = (1-beta) * u[j*block_size: (problem.d)] + beta * v_[j*block_size: (problem.d)]

                # Step 12
                u_new[j*block_size: (problem.d)] = problem.reg_func.prox_opr(v_[j*block_size: (problem.d)] - a * F_bar[j*block_size: (problem.d)], a)
            
        
        np.copyto(F_, F)
        F, b_A_x = problem.loss_func.grad_block_update(u)
        np.copyto(F_tilde_, F_tilde)
        np.copyto(u_, u)
        np.copyto(u, u_new)
        np.copyto(v_, v)

        # u_hat = ((A - a) * u_hat / A) + (a*u / A)


        iteration += 1
        if iteration % exit_criterion.loggingfreq == 0:
            elapsed_time = time.time() - start_time
            opt_measure = problem.func_value(u)
            logging.info(f"elapsed_time: {elapsed_time}, iteration: {iteration}, opt_measure: {opt_measure}")
            logresult(results, iteration, elapsed_time, opt_measure)
            exit_flag = CheckExitCondition(exit_criterion, iteration, elapsed_time, opt_measure)

    return results, u





def aduca_lazy_restart(problem: CompositeFunc, exit_criterion: ExitCriterion, parameters, u_0=None):
    # Init of adapCODER
    beta = parameters["beta"]
    c = parameters["c"]
    params = {"beta": beta, "c": c}

    num_restarts = parameters["restarts"]
    iterations = exit_criterion.maxiter // num_restarts

    u = np.zeros(problem.d)
    u_output = np.copy(u)

    exit_criterion.maxiter = iterations

    for restart in range(num_restarts):
        if restart == 0:
            result, u = aduca_lazy(problem=problem, exit_criterion=exit_criterion, parameters=params, u_0 = u_0)
        else:
            # print(f"This is u: {u}")
            result, u_output = aduca_lazy(problem=problem, exit_criterion=exit_criterion, parameters=params, u_0 = u)
            u = np.copy(u_output)
    return u




# ### three points extrapolation
# def aduca(problem: CompositeFunc, exit_criterion: ExitCriterion, parameters, x_0=None):
#     # Init of adapCODER
#     L = parameters["L"]
#     beta1, beta2, beta3 = parameters["beta1"], parameters["beta2"], parameters["beta3"]
#     rho_1 = (1 + (beta1 * beta3)/(1-beta3)) * beta3
#     rho_2 = (1-beta2) / beta3
#     rho_3 = (1-beta2) / beta1 * beta2 * 10
#     rho_4 = ((beta1 * beta2 *(1-beta2)) / ((1-beta3)*beta3)) **0.5

#     theta = np.ones(3)

#     if x_0 is None:
#         x_0 = np.random.randn(problem.d)
#     perturbation = np.random.uniform(-0.0001, 0.0001, size=x_0.shape)
#     x_1 = x_0 + perturbation
#     while np.array_equal( problem.grad(x_0),  problem.grad(x_1)):
#         perturbation = np.random.uniform(-0.0001, 0.0001, size=x_0.shape)
#         x_1 += perturbation

#     x = np.zeros((4, problem.d))
#     x[-2], x[-1] = x_0, x_1

#     v = np.copy(x[-1])
#     v_ = np.copy(v)

#     loss_grad = np.zeros((3, problem.d))
#     loss_grad[-2], loss_grad[-1] = problem.loss_func.grad(x_0), problem.loss_func.grad(x_1)

#     b_A_x = np.zeros(problem.loss_func.n)
#     m = problem.d  # Assume that each block is simply a coordinate for now

#     p= np.zeros_like(loss_grad)
#     for j in range(0, m, 1):
#         if j == 0:
#             loss_func_grad_x, b_A_x = problem.loss_func.grad_block_update(x[-2])
#             p[-1][0] = loss_func_grad_x[0]
#         else:
#             p[-1][j], b_A_x = problem.loss_func.grad_block_update(b_A_x, (j-1, x[-1][j-1] - x[-2][j-1]), j)
#     p[-2] = np.copy(loss_grad[-2])

#     q = np.zeros(problem.d)

#     step = np.linalg.norm(x[-1] - x[-2]) / np.linalg.norm(loss_grad[-1] - loss_grad[-2])
#     a = step
#     a_ = step
#     A = 0

#     theta = np.ones(3)


#     # Stepsize selection function
#     def aduca_stepsize(x, a, a_, theta, grad, tildeGrad, iter=3):
#         # step_1 = rho_1 * a
#         step_1 = rho_1 * a * 1.2

#         ### we can heuristically decrease the constant on the denominator
#         L_k = np.linalg.norm(grad[-1] - grad[-2]) / np.linalg.norm(x[-1] - x[-2])
#         # step_2 = (rho_2**2 * theta[-1]) / (300 * a * L_k**2)
#         step_2 = (rho_2**2 * theta[-1]) / (50 * a * L_k**2)

#         if iter <= 1:
#             step_3 = 10000
#         else:
#             L_hat_ = np.linalg.norm(tildeGrad[-2] - grad[-2]) / np.linalg.norm (x[-2] - x[-3])
#             # step_3 = (rho_2**2 * theta[-2]) / (300 * a * L_hat_**2)
#             step_3 = (rho_2**2 * theta[-2]) / (50 * a * L_hat_**2)
        
#         # step_4 = rho_3 * theta[-1] * a
#         step_4 = rho_3 * theta[-1] * a * 1.5

#         L_hat = np.linalg.norm(tildeGrad[-1] - grad[-1]) / np.linalg.norm(x[-1] - x[-2])
#         # step_5 = rho_4 * (theta[-1] / 10)**0.5 / L_hat
#         step_5 = rho_4 * (theta[-1] / 5)**0.5 / L_hat

#         if iter <= 2:
#             step_6 = 10000
#         else:
#             L_hat__ = np.linalg.norm(tildeGrad[-3] - grad[-3]) / np.linalg.norm(x[-3] - x[-4])
#             # step_6 = (rho_2**2 * theta[-1] * theta[-3]) / (60 * a_ * L_hat__**2)
#             step_6 = (rho_2**2 * theta[-1] * theta[-3]) / (10 * a_ * L_hat__**2) 
        
#         step = min(step_1, step_2, step_3, step_4, step_5, step_6)
#         # print(f" !!! Stepsize: {step}")
#         return step



#     # Run init
#     iteration = 0
#     exit_flag = False
#     start_time = time.time()
#     results = Results()
#     init_opt_measure = problem.func_value(x_0)
#     logresult(results, 1, 0.0, init_opt_measure)

#     while not exit_flag:
#         np.copyto(v_, v)

#         # Step 4
#         if iteration == 0:
#             step = aduca_stepsize(x, a, a_, theta, loss_grad, p, iter=1)
#         elif iteration == 1:
#             step = aduca_stepsize(x, a, a_, theta, loss_grad, p, iter=2)
#         else:
#             step = aduca_stepsize(x, a, a_, theta, loss_grad, p)
#         a_ = a
#         a = step
#         theta = np.append(theta[1:], a  / a_)

#         # Step 5
#         p = np.append(p[1:], np.zeros((1, problem.d)), axis=0)
#         for j in range(0, m):
#             # Step 7
#             q[j] = p[-2][j] + (a_ / a) * (loss_grad[-2][j] - p[-3][j])
            
#             # Step 8
#             if j == 0:
#                 z_0 = problem.reg_func.prox_opr_block(v_[j] - a * q[j], a)
#                 x = np.append(x[1:], np.zeros((1, problem.d)), axis=0) 
#                 x[-1][0] = z_0
#             else:
#                 x[-1][j] = problem.reg_func.prox_opr_block(v_[j] - a * q[j], a)

#             # Step 9
#             v[j] = beta1 * x[-1][j] + beta2 * x[-2][j] + beta3 * v_[j]
            
#             # Step 10
#             if j == 0:
#                 loss_func_grad_x, b_A_x = problem.loss_func.grad_block_update(x[-2])
#                 p[-1][0] = loss_func_grad_x[0]
#             else:
#                 p[-1][j], b_A_x = problem.loss_func.grad_block_update(b_A_x, (j-1, x[-1][j-1] - x[-2][j-1]), j)


#         loss_grad = np.append(loss_grad[1:], np.array([problem.loss_func.grad(x[-1])]), axis=0)

#         ### Compute x_hat
#         # A += step
#         # x_sum += step * x[-2]
#         # x_hat = 1/A * x_sum

#         iteration += 1
#         if iteration % exit_criterion.loggingfreq == 0:
#             elapsed_time = time.time() - start_time
#             opt_measure = problem.func_value(x[-1])
#             logging.info(f"elapsed_time: {elapsed_time}, iteration: {iteration}, opt_measure: {opt_measure}")
#             logresult(results, iteration, elapsed_time, opt_measure)
#             exit_flag = CheckExitCondition(exit_criterion, iteration, elapsed_time, opt_measure)

#     return results, x



