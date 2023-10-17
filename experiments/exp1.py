import sys 
sys.path.insert(0, sys.path[0]+"/../")
import utils
import torch
import estimate
import numpy as np
import pandas as pd

if __name__ == '__main__':

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print("DEVICE:", device)
    # utils.set_random_seed(1)
    # B_true = utils.simulate_dag(30, 29, "ER", device=device)
    # W_true = utils.simulate_parameter(B_true, device=device)
    # X = utils.simulate_linear_sem(W_true, 29, "gauss", device=device)
    # result = estimate.notears_linear(X, W_true=W_true, lambda1=0, loss_type="l2", lr=0.01, rho_max=1e+2)
    # print("================================RESULT")
    # results = pd.DataFrame(result[1:])
    
    # results.to_csv(sys.path[0]+"./results_loss/test"+str(1)+".csv", index=False)
    

    # import os
    # print(os.listdir(sys.path[0]+"/../"))

    #print(result)
    #np.linalg.norm(result[0]-W_true.cpu().detach().numpy(), ord='fro')
    #np.save('w_est.npy', result[0])
    #np.save('w_true.npy', W_true)

