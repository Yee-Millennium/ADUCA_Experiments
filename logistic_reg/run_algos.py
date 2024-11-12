# CODER_GR_code/run_algos.py

import argparse
import numpy as np
import os
import datetime
import json
import logging

from src.algorithms.gd import gd
from src.algorithms.coder import coder, coder_linesearch
from src.algorithms.gr import gr
from src.algorithms.aduca import aduca_lazy, aduca_lazy_restart
from src.algorithms.acoder_vr import acodervr
from src.algorithms.acoder import acoder

from src.problems.utils.data_parsers import libsvm_parser
from src.problems.loss_func.logisticloss import LogisticLoss
from src.problems.reg_func.elasticnet import ElasticNet
from src.problems.composite_func import CompositeFunc
from src.algorithms.utils.exitcriterion import ExitCriterion


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Simulating BLAS.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

## (Dimension, Sample size)
DATASET_INFO = { 
    "sonar_scale": (60, 208),
    "a1a": (123, 1605),
    "a9a": (123, 32561),
    "gisette_scale": (5000, 6000),
    "news20": (1355191, 19996),
    "rcv1": (47236, 20242),
    "phishing": (68, 11055),
    "colon-cancer": (2000, 62),
    "madelon": (500, 2000),
    "mushrooms": (112, 8124),
    "skin_nonskin": (3, 245057),
    "SUSY": (18,5000000)
}

def parse_commandline():
    parser = argparse.ArgumentParser(description='Run optimization algorithms.')
    parser.add_argument('--outputdir', required=True, help='Output directory')
    parser.add_argument('--maxiter', required=True, type=int, help='Max iterations')
    parser.add_argument('--maxtime', required=True, type=int, help='Max execution time in seconds')
    parser.add_argument('--targetaccuracy', required=True, type=float, help='Target accuracy')
    parser.add_argument('--optval', type=float, default=0.0, help='Known optimal value')
    parser.add_argument('--loggingfreq', type=int, default=100, help='Logging frequency')
    parser.add_argument('--dataset', required=True, help='Choice of dataset')
    parser.add_argument('--lossfn', default='logistic', help='Choice of loss function')
    parser.add_argument('--lambda1', type=float, default=0.0, help='Elastic net lambda 1')
    parser.add_argument('--lambda2', type=float, default=0.0, help='Elastic net lambda 2')
    parser.add_argument('--algo', required=True, help='Algorithm to run')
    parser.add_argument('--lipschitz', required=True, type=float, help='Lipschitz constant')
    parser.add_argument('--gamma', type=float, default=0.0, help='Gamma')
    parser.add_argument('--K', type=int, default=0, help='Variance reduction K')
    # parser.add_argument('--beta1', type = float, help='adapCoder constant parameter 1')
    # parser.add_argument('--beta2', type = float, help='adapCoder constant parameter 2')
    # parser.add_argument('--beta3', type = float, help='adapCoder constant parameter 3')
    parser.add_argument('--beta', type = float, help='aduca constant parameter')
    parser.add_argument('--c', type = float, help='aduca constant parameter')
    parser.add_argument('--restarts', type = int, help='aduca_restart constant parameter')
    parser.add_argument('--block_size', type = int, help='block_size parameter >= 1, <= n')

    return parser.parse_args()

def main():
    # Run setup
    args = parse_commandline()
    outputdir = args.outputdir
    algorithm = args.algo

    # Problem Setup
    dataset = args.dataset
    lambda1 = args.lambda1
    lambda2 = args.lambda2

    if dataset not in DATASET_INFO:
        raise ValueError("Invalid dataset name supplied.")
    
    d, n = DATASET_INFO[dataset]
    filepath = f"data/libsvm/{dataset}"
    data = libsvm_parser(filepath, n, d)
    loss = LogisticLoss(data)
    reg = ElasticNet(lambda1, lambda2)
    problem = CompositeFunc(loss, reg)

    logging.info(f"dataset: {dataset}, d: {d}, n: {n}")
    logging.info(f"elasticnet_λ₁ = {lambda1}; elasticnet_λ₂ = {lambda2}")
    logging.info("--------------------------------------------------")

    # Exit criterion
    maxiter = args.maxiter
    maxtime = args.maxtime
    targetaccuracy = args.targetaccuracy + args.optval
    loggingfreq = args.loggingfreq
    exitcriterion = ExitCriterion(maxiter, maxtime, targetaccuracy, loggingfreq)

    logging.info(f"maxiter = {maxiter}")
    logging.info(f"maxtime = {maxtime}")
    logging.info(f"targetaccuracy = {targetaccuracy}")
    logging.info(f"loggingfreq = {loggingfreq}")
    logging.info("--------------------------------------------------")


    # Runing
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    logging.info(f"timestamp = {timestamp}")
    logging.info("Completed initialization")
    outputfilename = f"{outputdir}/{dataset}-{lambda1}_{lambda2}-{algorithm}-{args.lipschitz}-output-{timestamp}.json"
    logging.info(f"outputfilename = {outputfilename}")
    logging.info("--------------------------------------------------")

    if algorithm == "ACODER":
        L = args.lipschitz
        gamma = args.gamma
        acoder_params = {"L": L, "gamma": gamma}
        output, output_v, output_x = acoder(problem, exitcriterion, acoder_params)

    elif algorithm == "CODER":
        logging.info("Running CODER...")
        L = args.lipschitz
        gamma = args.gamma
        logging.info(f"Setting L = {L}, gamma = {gamma}")
        coder_params = {"L": L, "gamma": gamma}
        output, output_x = coder(problem, exitcriterion, coder_params)

    elif algorithm == "ACODER-VR":
        L = args.lipschitz
        gamma = args.gamma
        K = args.K if args.K != 0 else n
        acodervr_params = {"L": L, "gamma": gamma, "K": K}
        output, output_v, output_y = acodervr(problem, exitcriterion, acodervr_params)

    elif algorithm == "CODER_linesearch":
        logging.info("Running CODER_linesearch...")
        L = args.lipschitz
        gamma = args.gamma
        logging.info(f"Setting L = {L}, gamma = {gamma}")
        coder_linesearch_params = {"L": L, "gamma": gamma}
        output, output_x = coder_linesearch(problem, exitcriterion, coder_linesearch_params)
        
    elif algorithm == "GR":
        logging.info("Running Golden Ratio...")
        beta = args.beta
        logging.info(f"Setting beta ={beta}")
        gr_params = {"beta": beta}
        output, output_x = gr(problem, exitcriterion, gr_params)
    
    elif algorithm == "ADUCA":
        logging.info("Running ADUCA...")
        beta = args.beta
        c = args.c
        aduca_params = {"beta": beta, "c": c}
        output, output_x = aduca(problem, exitcriterion, aduca_params)
    
    elif algorithm == "ADUCA_lazy":
        logging.info("Running ADUCA_lazy...")
        beta = args.beta
        c = args.c
        block_size = args.block_size
        aduca_params = {"beta": beta, "c": c, "block_size":block_size}
        output, output_x = aduca_lazy(problem, exitcriterion, aduca_params)
    elif algorithm == "ADUCA_lazy_restart":
        logging.info("Running ADUCA_lazy_restart...")
        beta = args.beta
        c = args.c
        restarts = args.restarts 
        aduca_params = {"beta": beta, "c": c, "restarts": restarts}
        output, output_x = aduca_lazy_restart(problem, exitcriterion, aduca_params)

    # elif algorithm == "RCDM":
    #     Ls = np.ones(d) * args.lipschitz
    #     alpha = 1.0
    #     rcdm_params = {"Ls": Ls, "alpha": alpha}
    #     output = rcdm(problem, exitcriterion, rcdm_params)

    #elif algorithm == "ACDM":
    #    Ls = np.ones(d) * args.lipschitz
    #    gamma = args.gamma
    #    acdm_params = {"Ls": Ls, "gamma": gamma}
    #    output = acdm(problem, exitcriterion, acdm_params)

    elif algorithm == "GD":
        logging.info("Running gradient descent...")
        L = args.lipschitz
        logging.info(f"Setting L = {L}")
        gd_params = {"L": L}
        output,output_x = gd(problem, exitcriterion, gd_params)

    else:
        raise ValueError("Wrong algorithm name supplied")

    with open(outputfilename, 'w') as outfile:
        json.dump({"args": vars(args), 
                   "output_x": output_x.tolist(),
                   "iterations": output.iterations, 
                   "times": output.times,
                   "optmeasures": output.optmeasures}, 
                   outfile)
        logging.info(f"output saved to {outputfilename}")


if __name__ == "__main__":
    main()

