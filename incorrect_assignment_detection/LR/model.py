import torch
import torch.nn as nn

import numpy as np
import scipy.sparse as sp
from scipy.special import iv
from sklearn import preprocessing
from sklearn.utils.extmath import randomized_svd

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        self.dim = input_dim
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self,feature):

        return torch.sigmoid(self.linear(feature))

def get_embedding_rand(matrix):
    # Sparse randomized tSVD for fast embedding
    smat = sp.csr_matrix(matrix.cpu().numpy())
    # smat = sp.csc_matrix(matrix)  # convert to sparse CSC format
    U, Sigma, VT = randomized_svd(
        smat, n_components=100, n_iter=5, random_state=None
    )
    # assert U.shape[1] == 100
    
    U = U * np.sqrt(Sigma)
    if U.shape[1]< 100:
        U_ext = np.zeros((U.shape[0],100))
        U_ext[:,:U.shape[1]] = U
        U = U_ext
    U = preprocessing.normalize(U, "l2")

    return torch.tensor(U, requires_grad=False, dtype =torch.float32)

def to_dense(edge_index, edge_attr,x):
    
    weighted_adjacency_matrix = torch.zeros((x.size(0), x.size(0)))
    weighted_adjacency_matrix[edge_index[0], edge_index[1]] = edge_attr
    weighted_adjacency_matrix = weighted_adjacency_matrix + torch.eye(x.size(0))
    weighted_adjacency_matrix = (weighted_adjacency_matrix + weighted_adjacency_matrix.T.contiguous())/2

    return weighted_adjacency_matrix
