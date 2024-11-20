import numpy as np
import time
import logging
from src.algorithms.utils.exitcriterion import ExitCriterion, CheckExitCondition
from src.problems.GMVI_func import GMVIProblem
from src.algorithms.utils.results import Results, logresult

# Setup logging
logging.basicConfig(level=logging.INFO)

def aduca_lazy(problem: GMVIProblem, exit_criterion: ExitCriterion, parameters, u_0=None):
    # Init of adapCODER
    d = problem.operator_func.d
    n = problem.operator_func.n
    beta = parameters["beta"]
    c = parameters["c"]
    # block_size = parameters['block_size']

    phi_1 = 2 * c * beta * (1+beta)
    phi_2 = (((1-2*c) * (1 + beta))  /  7 * beta)**0.5 * 0.5
    phi_3 = 1 / (7 * beta)
    # print(f"!!! phi_2: {phi_2}")
    # print(f"!!! phi_3: {phi_3}")

    alpha = min(phi_1, 4 / (7 * beta * (1+beta) * (1-2 * c) ))
    # print(f"alpha: {alpha}")

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
    ## line-search for the first step
    a_0 = 2 * phi_1
    while True:
        u = np.copy(u_0)
        a_0 = a_0 / 2
        u_1 = problem.g_func.prox_opr(u_0 - a_0 * F_0, a_0, d)
        
        F_1 = problem.operator_func.func_map(u_1)
        F_tilde_1 = np.copy(F_tilde_0)
        for j in range(0, problem.d, 1):
            F_tilde_1[j] = problem.operator_func.func_map_block(j+1, u)
            u[j] = u_1[j]
        
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
    w = np.copy(u) ### delayed vector
    v_ = np.copy(u_)
    u_hat = A * u_

    F = np.copy(F_1)
    F_ = np.copy(F_0)
    F_tilde = np.copy(F_tilde_1)
    F_tilde_ = np.copy(F_tilde_0)
    F_bar = np.zeros(problem.d)
    
    # Stepsize selection function
    def aduca_stepsize(u, u_, a, a_, F, F_, F_tilde):
        # step_1 = alpha * a
        # print(f"!!! alpha: {alpha}")
        step_1 = alpha * a 

        ### we can heuristically scale the step
        L_hat_k = np.linalg.norm(F - F_tilde) / (np.linalg.norm(u-u_)) 
        # step_2 = (phi_2 / L_hat) * (a / a_)**0.5
        # print(f"!!! norm of F - F_tilde: {np.linalg.norm(F - F_tilde)}")
        # print(f"!!! norm of u-u_: {np.linalg.norm(u - u_)}")
        # print(f"!!! L_hat_k: {L_hat_k}")
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
        for j in range(0, problem.d):

            # Step 8
            F_bar[j] = F_tilde[j] + (a_ / a) * (F_[j] - F_tilde_[j])
            
            # Step 9
            v[j] = (1-beta) * u[j] + beta * v_[j]

            # Step 10
            u_[j] = u[j]
            u[j] = problem.g_func.prox_opr_block(j+1 ,v[j] - a * F_bar[j], a)
            if j > 0:
                w[j-1] = u[j-1]

            # Step 11
            F_tilde_[j] = F_tilde[j]
            F_tilde[j] = problem.operator_func.func_map_block(j + 1, w)
        np.copyto(w, u)
        np.copyto(F_, F)
        F = problem.operator_func.func_map(u)
        # print(f"If F equal to F_tilde")
        np.copyto(v_, v)

        # print(f"!!! (a / A): {(a / A)}")
        u_hat = ((A - a) * u_hat / A) + (a*u_ / A)


        iteration += problem.d
        if iteration % (problem.d *  exit_criterion.loggingfreq) == 0:
            elapsed_time = time.time() - start_time
            opt_measure = problem.func_value(u_hat)
            logging.info(f"elapsed_time: {elapsed_time}, iteration: {iteration}, opt_measure: {opt_measure}")
            logresult(results, iteration, elapsed_time, opt_measure)
            exit_flag = CheckExitCondition(exit_criterion, iteration, elapsed_time, opt_measure)

    return results, u


# ### Ergodic Restart Scheme
# def aduca_lazy_restart(problem: GMVIProblem, exit_criterion: ExitCriterion, parameters, u_0=None):
#     # Init of adapCODER
#     beta = parameters["beta"]
#     c = parameters["c"]
#     params = {"beta": beta, "c": c}

#     num_restarts = parameters["restarts"]
#     iterations = exit_criterion.maxiter // num_restarts

#     if u_0 == None:
#         u = np.zeros(problem.d)
#     else:
#         u = np.copy(u_0)
#     u_output = np.copy(u)

#     exit_criterion.maxiter = iterations

#     iteration_total = 0
#     exit_flag = False
#     start_time = time.time()
#     results = Results()
#     init_opt_measure = problem.func_value(u)
#     logresult(results, 1, 0.0, init_opt_measure)

#     for restart in range(num_restarts):
#         if restart == 0:
#             result, u = aduca_lazy(problem=problem, exit_criterion=exit_criterion, parameters=params, u_0 = u_0)
#             iteration_total += iterations
#         if iteration_total % (iterations *  exit_criterion.loggingfreq) == 0:
#             elapsed_time = time.time() - start_time
#             opt_measure = problem.func_value(u)
#             logging.info(f"elapsed_time: {elapsed_time}, iteration: {iteration_total}, opt_measure: {opt_measure}")
#             logresult(results, iteration_total, elapsed_time, opt_measure)
#         else:
#             # print(f"This is u: {u}")
#             result, u_output = aduca_lazy(problem=problem, exit_criterion=exit_criterion, parameters=params, u_0 = u)
#             u = np.copy(u_output)
#             if iteration_total % (iterations *  exit_criterion.loggingfreq) == 0:
#                 elapsed_time = time.time() - start_time
#                 opt_measure = problem.func_value(u)
#                 logging.info(f"elapsed_time: {elapsed_time}, iteration: {iteration_total}, opt_measure: {opt_measure}")
#                 logresult(results, iteration_total, elapsed_time, opt_measure)
#     return results, u



# def aduca_lazy(problem: CompositeFunc, exit_criterion: ExitCriterion, parameters, u_0=None):
#     # Init of adapCODER
#     beta = parameters["beta"]
#     c = parameters["c"]
#     block_size = parameters['block_size']
#     block_number = problem.d // block_size 
#     last_block_size = problem.d - (block_number-1) * block_size
#     logging.info(f"block_number: {block_number}; block_size: {block_size}; last_block_size: {last_block_size}")

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
#         for j in range(0, block_number, 1):
#             if j == 0:
#                 p, b_A_x = problem.loss_func.grad_block_update(u_0)
#                 F_tilde_1[0:(block_size)] = p[0:(block_size)]
#             elif j != (block_number - 1):
#                 # print(f"block_size for the initialization: {j*block_size}")
#                 F_tilde_1[j*block_size:((j+1)*block_size)], b_A_x = problem.loss_func.grad_block_update(b_A_x, u_1[(j-1)*block_size:(j*block_size)] - u_0[(j-1)*block_size:(j*block_size)], j, block_size, block_size)
#             else:
#                 F_tilde_1[j*block_size: (problem.d)], b_A_x = problem.loss_func.grad_block_update(b_A_x, u_1[(j-1)*block_size:(j*block_size)] - u_0[(j-1)*block_size:(j*block_size) ], j, block_size, last_block_size)
        
#         norm_F = np.linalg.norm((F_1 - F_0))
#         norm_F_tilde = np.linalg.norm((F_1 - F_tilde_1))
#         norm_u = np.linalg.norm((u_1 - u_0))
#         # print(f"!!! Norm of F: {norm_F}")
#         # print(f"!!! Norm of F_tilde: {norm_F_tilde}")
#         # print(f"!!! Norm of u: {norm_u}")
#         # print(f"phi_2: {phi_2}")
#         # print(f"a_0: {a_0}")
#         if (a_0 * norm_F <= phi_2 * norm_u) and (a_0 * norm_F_tilde <= phi_2 * norm_u):
#             break
#         if (a_0 <= 1):
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
#             # print(f"a: {a}")
#             # print(f"a_: {a_}")
#             step_2 = (phi_2 / L_hat_k) * (a / a_)**0.5    * 6

#         L_k = np.linalg.norm(F - F_) / np.linalg.norm(u - u_)
#         if L_k == 0:
#             step_3 = 1000000
#         else:
#             step_3 = (phi_3 ** 2) / (a_ * L_k**2)     * 6
        
#         step = min(step_1, step_2, step_3)
#         # print(f"!!! Stepsizes: {step_1, step_2, step_3}")
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
#         # A += a

#         # Step 8
#         for j in range(0, block_number, 1):
#             if j == 0:
#                 F_tilde[0:(block_size)] = F[0:(block_size)]
#                 # Step 10
#                 F_bar[0:(block_size)] = F_tilde[0:(block_size)] + (a_ / a) * (F_[0:(block_size)] - F_tilde_[0:(block_size)] )
                
#                 # Step 11
#                 v[0:(block_size)] = (1-beta) * u[0:(block_size)] + beta * v_[0:(block_size)]

#                 # Step 12
#                 u_new[0:(block_size)] = problem.reg_func.prox_opr(v_[0:(block_size)] - a * F_bar[0:(block_size)], a)

#             elif j != block_number - 1:
#                 F_tilde[j*block_size:((j+1)*block_size)], b_A_x = problem.loss_func.grad_block_update(b_A_x, u_1[(j-1)*block_size:(j*block_size)] - u_0[(j-1)*block_size:(j*block_size)], j, block_size, block_size)
#                 # Step 10
#                 F_bar[j*block_size:((j+1)*block_size)] = F_tilde[j*block_size:((j+1)*block_size)] + (a_ / a) * (F_[j*block_size:((j+1)*block_size)] - F_tilde_[j*block_size:((j+1)*block_size)] )
                
#                 # Step 11
#                 v[j*block_size:((j+1)*block_size)] = (1-beta) * u[j*block_size:((j+1)*block_size)] + beta * v_[j*block_size:((j+1)*block_size)]

#                 # Step 12
#                 u_new[j*block_size:((j+1)*block_size)] = problem.reg_func.prox_opr(v_[j*block_size:((j+1)*block_size)] - a * F_bar[j*block_size:((j+1)*block_size)], a)

#             else:
#                 F_tilde[j*block_size: (problem.d)], b_A_x = problem.loss_func.grad_block_update(b_A_x, u_1[(j-1)*block_size:(j*block_size)] - u_0[(j-1)*block_size:(j*block_size)], j, block_size, last_block_size)
#                 # Step 10
#                 F_bar[j*block_size: (problem.d)] = F_tilde[j*block_size: (problem.d)] + (a_ / a) * (F_[j*block_size: (problem.d)] - F_tilde_[j*block_size: (problem.d)] )
                
#                 # Step 11
#                 v[j*block_size: (problem.d)] = (1-beta) * u[j*block_size: (problem.d)] + beta * v_[j*block_size: (problem.d)]

#                 # Step 12
#                 u_new[j*block_size: (problem.d)] = problem.reg_func.prox_opr(v_[j*block_size: (problem.d)] - a * F_bar[j*block_size: (problem.d)], a)
            
        
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




