import torch
import torch.nn.functional as F
from scipy import linalg as slin
import numpy as np
import matplotlib.pyplot as plt
from SGLD import SGLD

def notears_linear(X, lambda1, loss_type, W_true, max_iter=100, h_tol=1e-8, loss_tol=1e-6,\
    rho_max=1e+16, w_threshold=0.3, device=torch.device("cpu"), lr=0.01):

    X = torch.tensor(X, dtype=torch.float32, device=device)
    n, d = X.shape

    def sigmoid(x):
        return 1 / (1 + torch.exp(-x))

    def _loss(W):
        M = torch.matmul(X, W)
        if loss_type == 'l2':
            R = X - M
            loss = 0.5 / n * torch.sum(R ** 2)
            G_loss = -1.0 / n * torch.matmul(X.T, R)
        elif loss_type == 'logistic':
            loss = 1.0 / n * torch.sum(F.logsigmoid(M) - X * M)
            G_loss = 1.0 / n * torch.matmul(X.T, sigmoid(M) - X)
        elif loss_type == 'poisson':
            S = torch.exp(M)
            loss = 1.0 / n * torch.sum(S - X * M)
            G_loss = 1.0 / n * torch.matmul(X.T, S - X)
        else:
            raise ValueError('unknown loss type')
        return loss, G_loss

    def _h(W):
        E = torch.matrix_exp(W * W)
        h = torch.trace(E) - d
        G_h = E.T * W * 2
        return h, G_h

    # def _adj(w):
    #     return (w[:d * d] - w[d * d:]).reshape([d, d])
    def _adj(w):
        return w.reshape([d, d])

    def _losses(W):
        """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
        loss_est, _ = _loss(W)
        h, _ = _h(W)
        loss_l1 = loss_est + lambda1 * torch.abs(W).sum()
        obj_new = loss_l1 + 0.5 * rho * h * h 
        obj_dual = obj_new + alpha * h
        return loss_est.item(), loss_l1.item(), obj_new.item(), obj_dual.item(), h.item()

    def _func(w):
        W = _adj(w)
        loss, G_loss = _loss(W)
        h, G_h = _h(W)

        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.abs().sum()
        G_smooth = G_loss + (rho * h + alpha) * G_h
        g_obj = torch.cat((G_smooth + lambda1, - G_smooth + lambda1), dim=0)
        return obj, g_obj

    w_est = torch.zeros(d * d, dtype=torch.float, device=device)
    rho = torch.tensor(1.0, dtype=torch.float, device=device)
    alpha = torch.tensor(0.0, dtype=torch.float, device=device)
    h = torch.tensor(float('inf'), dtype=torch.float, device=device)
    
    loss_values = []
    iters = []

    for i in range(max_iter):
           
        while rho < rho_max:
            # w_new = torch.zeros(d * d, dtype=torch.float, device=device).requires_grad_(True)
         
            w_new = w_est.detach().clone().requires_grad_(True)

            #optimizer = torch.optim.SGD([w_new], lr=0.01/rho)  # Use SGD as optimizer
            #optimizer = SGLD([w_new], lr=lr) #0.01/rho for SGD
            optimizer = torch.optim.Adam([w_new], lr=lr)
            loss = torch.tensor(float('inf'), dtype=torch.float, device=device)
            loss_new = torch.tensor(0.0, dtype=torch.float, device=device)
            
            sgd_step = 0

            while (loss - loss_new).abs() > loss_tol:
                #loss_last = loss.clone()
                optimizer.zero_grad()
                
                # Augmented Lagrangian loss
                loss, _ = _func(w_new)
                h_new, _ = _h(_adj(w_new))

                # back propagate
                loss.backward()

                # Add Gaussian noise to the gradient for SGLD
                with torch.no_grad():
                    for param in [w_new]:
                        noise = torch.randn_like(param.grad) * np.sqrt(2*lr)
                        param.grad.add_(noise)

                optimizer.step()

                loss_new, _ = _func(w_new)

                # constraints
                # diagonal_indices = [i + i*d for i in range(d)]
                # diagonal_indices = diagonal_indices + [i+d*d for i in diagonal_indices]
                # w_new.data[diagonal_indices] = 0
                # w_new.data = torch.clamp(w_new.data, min=0)
                # with torch.no_grad(): 
                #     diagonal_indices = [i + i*d for i in range(d)]
                #     diagonal_indices = diagonal_indices + [i+d*d for i in diagonal_indices]
                #     w_new.data[diagonal_indices] = 0
                #     w_new.data.clamp_(min=0)

                sgd_step += 1
                #iters.append(i)
                if sgd_step % 100 == 1:
                    loss_, _ = _loss(_adj(w_new))
                    loss_values.append(loss_.item())
                    print(f"Iteration {sgd_step}: Loss = {loss_.item()}, h = {h_new.item()}, rho = {rho.item()}, alpha = {alpha.item()}")
                    #print((loss - loss_last).abs().item())
                
                # if loss_new > 1000 * loss:
                #     break
            
            h_new, _ = _h(_adj(w_new))
       
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break

        w_est, h = w_new.detach().clone(), h_new.detach().clone()
        alpha += rho * h

        if h <= h_tol or rho >= rho_max:
            break

    W_est = _adj(w_est)
    # W_est2 = W_est.detach().clone()
    # W_est2[np.abs(W_est2) < w_threshold] = 0

    loss_est, loss_l1, obj_new, obj_dual, h = _losses(W_est)
    # loss_est2, loss_l12, obj_new2, obj_dual2, h2 = _losses(W_est2)
    # loss_est3, loss_l13, obj_new3, obj_dual3, h3 = _losses(W_true)

    
    # plt.plot(loss_values)
    # plt.xlabel('Iteration')
    # plt.ylabel('Loss')
    # plt.title('Loss Curve')
    # plt.savefig("loss.png")
    # plt.show()

    # plt.plot(iters)
    # plt.xlabel('Iteration')
    # plt.ylabel('Iter')
    # plt.title('Length of Loop')
    # plt.savefig("iter.png")
    # plt.show()
    loss_values2 = [np.nan] * 200
    if len(loss_values) > 200:
        loss_values2 = loss_values[:200]
    else:
        loss_values2[:len(loss_values)] = loss_values

    #return loss_values2

    return _adj(w_est).cpu().detach().numpy(), loss_est, loss_l1, obj_new, obj_dual, h