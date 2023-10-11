import sys 
sys.path.insert(0, sys.path[0]+"/../")
import utils
import torch
import estimate
import random
import numpy as np
import math
import multiprocessing as mp
from functools import partial
import pandas as pd

def restore_from_per(per_ind, W):
    W_re = np.zeros(W.shape)
    for i in range(len(per_ind)):
        for j in range(len(per_ind)):
            W_re[i, j] = W[per_ind[i], per_ind[j]]
    return W_re

def estimate_once_notears_sgld(d, sem_type, graph_type, seed=1, lambda1=0, loss_type='l2', BB=100, noise=None, device=torch.device("cpu")):
    
    print("=============processing seed=", seed, "================")
    #utils.set_random_seed(seed)
    n = d-1
    s0 = n
    print("n, d, s0, graph_type, sem_type, lambda1, loss_type", "-", n, d, s0, graph_type, sem_type, lambda1, loss_type)
    
    B_true = utils.simulate_dag(d, s0, graph_type)
    W_true = utils.simulate_parameter(B_true)
    B = np.min([math.factorial(d), BB])
    results = np.zeros([6, B+2])

    if noise == "normal":
        noise_scale = np.abs(np.random.normal(1, 1, d))
    elif noise == "uni":
        noise_scale = np.random.uniform(1, 2, d)
    else:
        noise_scale = None
    
    X = utils.simulate_linear_sem(W_true, n, sem_type, noise_scale=noise_scale)

    M = X @ W_true
    if loss_type == 'l2':
        R = X - M
        results[0, 0] = 0.5 / X.shape[0] * (R ** 2).sum()
    elif loss_type == 'logistic':
        results[0, 0] = 1.0 / X.shape[0] * (np.logaddexp(0, M) - X * M).sum()
    elif loss_type == 'poisson':
        S = np.exp(M)
        results[0, 0] = 1.0 / X.shape[0] * (S - X * M).sum()
    else:
        raise ValueError('unknown loss type')
    
    #all_perm = list(permutations(range(X.shape[1])))
    #utils.set_random_seed(seed+d)
    inds = [i for i in range(d)]
    w_est, *losses = estimate.notears_linear(X, W_true=W_true, lambda1=0, loss_type=loss_type, lr=0.01)
    results[:5, 1] = losses
    results[5, 1] = np.linalg.norm(w_est-W_true.cpu().detach().numpy(), ord='fro')
    for i in range(B):
        random.shuffle(inds)
        w_est, *losses = estimate.notears_linear(X[:, inds], W_true=W_true, lambda1=0, loss_type=loss_type, lr=0.01)
        w_est_re = restore_from_per(inds, w_est)
        results[:5, i+2] = losses
        results[5, i+2] = np.linalg.norm(w_est_re-W_true.cpu().detach().numpy(), ord='fro')
        
    results = pd.DataFrame(results.T, columns=[
    "loss_est", "loss_l1", "obj_aug", "obj_dual", "h", "fro_dist"])
    
    results.to_csv(sys.path[0]+"./results_loss/result_"+str(seed)+"_"+sem_type+"_"+str(d)+".csv", index=False)

def worker(rank, ds, semtypes):
    torch.cuda.set_device(rank)
    device = torch.device("cuda") 
    
    for sm in range(len(semtypes)):
        for dd in range(len(ds)):
            if semtypes[sm] != "logistic":
                loss_type = "l2"
                graph_type = "ER"
            else:
                loss_type = "logistic"
                graph_type = "BP"
            estimate_once_notears_sgld(ds[dd], semtypes[sm], graph_type, seed=6000+rank, lambda1=0, loss_type=loss_type, BB=100, device=device)

        
if __name__ == '__main__':

    semtypes = ["gauss", "gumbel", "logistic"]
    ds = [5, 10, 15, 20, 30, 40]
    torch.multiprocessing.spawn(worker, args=(ds, semtypes), nprocs=torch.cuda.device_count(), join=True)

    # device = torch.device("cuda")
    
    # for sm in range(len(semtypes)):
    #     for dd in range(len(ds)):
    #         if semtypes[sm] != "logistic":
    #             loss_type = "l2"
    #             graph_type = "ER"
    #         else:
    #             loss_type = "logistic"
    #             graph_type = "BP"
    #         estimate_once_notears_sgld(ds[dd], semtypes[sm], graph_type, seed=6000, lambda1=0, loss_type=loss_type, BB=200, device=device)


    # pool = mp.Pool(processes=6)
    # with pool:
    #     for sm in range(len(semtypes)):
    #         pool.map(partial(estimate_once, 
    #         graph_type="ER", seed=5001, sem_type=semtypes[sm], lambda1=0, device=device), 
    #         ds)
    # pool.close()