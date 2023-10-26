import torch
import torch.nn.functional as F
import random
import numpy as np
import igraph as ig

def set_random_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def is_dag(W):
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    return G.is_dag()


def simulate_dag(d, s0, graph_type, device=torch.device("cpu")):
    def _random_permutation(M):
        # torch.randperm permutes first axis only
        P = torch.eye(M.shape[0], device=device, dtype=torch.float)[torch.randperm(M.shape[0])]
        return P.t() @ M @ P

    def _random_acyclic_orientation(B_und):
        return torch.tril(_random_permutation(B_und), diagonal=-1)

    def _graph_to_adjmat(G):
        return torch.tensor(G.get_adjacency().data, device=device, dtype=torch.float)

    if graph_type == 'ER':
        # Erdos-Renyi
        G_und = ig.Graph.Erdos_Renyi(n=d, m=s0)
        B_und = _graph_to_adjmat(G_und)
        B = _random_acyclic_orientation(B_und)
    elif graph_type == 'SF':
        # Scale-free, Barabasi-Albert
        G = ig.Graph.Barabasi(n=d, m=int(round(s0 / d)), directed=True)
        B = _graph_to_adjmat(G)
    elif graph_type == 'BP':
        # Bipartite, Sec 4.1 of (Gu, Fu, Zhou, 2018)
        top = int(0.2 * d)
        G = ig.Graph.Random_Bipartite(top, d - top, m=s0, directed=True, neimode=ig.OUT)
        B = _graph_to_adjmat(G)
    else:
        raise ValueError('unknown graph type')

    B_perm = _random_permutation(B)
    assert ig.Graph.Adjacency(B_perm.cpu().numpy().tolist()).is_dag()
    
    return B_perm


def simulate_parameter(B, w_ranges=((-2.0, -0.5), (0.5, 2.0)), device=torch.device("cpu")):
    """Simulate SEM parameters for a DAG.
    Args:
        B (torch.Tensor): [d, d] binary adj matrix of DAG
        w_ranges (tuple): disjoint weight ranges
    Returns:
        W (torch.Tensor): [d, d] weighted adj matrix of DAG
    """
    W = torch.zeros(B.shape, device=device)
    S = torch.randint(len(w_ranges), B.shape, device=device)  # which range
    for i, (low, high) in enumerate(w_ranges):
        U = torch.rand(B.shape, device=device) * (high - low) + low
        W += B * (S == i) * U
    return W


def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


def simulate_linear_sem(W, n, sem_type, noise_scale=None, device=torch.device("cpu")):
    #W = torch.tensor(W, device=device)
    d = W.shape[0]
    
    if noise_scale is None:
        scale_vec = torch.ones(d, device=device)
    elif torch.tensor(noise_scale).numel() == 1:
        scale_vec = noise_scale * torch.ones(d, device=device)
    else:
        if len(noise_scale) != d:
            raise ValueError('noise scale must be a scalar or has length d')
        scale_vec = torch.tensor(noise_scale, device=device)
    
    if not is_dag(W):
        raise ValueError('W must be a DAG')
    
    if n == float('inf'):
        if sem_type == 'gauss':
            X = torch.sqrt(d) * torch.diag(scale_vec) @ torch.inverse(torch.eye(d, device=device) - W)
            return X
        else:
            raise ValueError('population risk not available')
    
    G = ig.Graph.Weighted_Adjacency(W.cpu().numpy().tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    X = torch.ones([n, d], device=device)
    
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        
        parent_tensor = X[:, parents]
        weight_tensor = W[parents, j]
        
        if sem_type == 'gauss':
            z = torch.normal(mean=0, std=scale_vec[j], size=(n,)).to(device)
            X[:, j] = parent_tensor @ weight_tensor + z
        
        elif sem_type == 'exp':
            z = torch.exponential(scale=scale_vec[j], size=(n,)).to(device)
            X[:, j] = parent_tensor @ weight_tensor + z
        
        elif sem_type == 'gumbel':
            z = torch.distributions.Gumbel(0, scale_vec[j]).sample((n,)).to(device)
            X[:, j] = parent_tensor @ weight_tensor + z
        
        elif sem_type == 'uniform':
            z = torch.distributions.Uniform(-scale_vec[j], scale_vec[j]).sample((n,)).to(device)
            X[:, j] = parent_tensor @ weight_tensor + z
        
        elif sem_type == 'logistic':
            X[:, j] = torch.distributions.Binomial(1, sigmoid(parent_tensor @ weight_tensor)).sample().to(device)
        
        elif sem_type == 'poisson':
            X[:, j] = torch.distributions.Poisson(torch.exp(parent_tensor @ weight_tensor).double()).sample().float().to(device)
        
        else:
            raise ValueError('unknown sem type')

    return X