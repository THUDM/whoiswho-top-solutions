import torch
import torch.nn as nn

import numpy as np
import scipy.sparse as sp
from scipy import linalg
from scipy.special import iv
from sklearn import preprocessing
from sklearn.utils.extmath import randomized_svd
import pickle
import multiprocessing as mp
import argparse

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

def to_dense(edge_index, edge_attr, n):
    
    weighted_adjacency_matrix = torch.zeros((n, n))
    weighted_adjacency_matrix[edge_index[0], edge_index[1]] = edge_attr
    weighted_adjacency_matrix = weighted_adjacency_matrix + torch.eye(n)
    weighted_adjacency_matrix = (weighted_adjacency_matrix + weighted_adjacency_matrix.T.contiguous())/2

    return weighted_adjacency_matrix

def get_embedding(g):
    Graph, _, author_id, pub_id  = g
    _, edge_index, edge_attr, label, _ = Graph.x, Graph.edge_index, Graph.edge_attr.squeeze(-1), Graph.y, Graph.batch
    features = get_embedding_rand(to_dense(edge_index, edge_attr,Graph.y.shape[0]))
    return features, label, author_id, pub_id

def build_dataset(src_path, dict_path):
    
    with open(src_path, 'rb') as f:
        data = pickle.load(f)
    with mp.Pool(processes=80) as pool:
        results = pool.map(get_embedding,data)
    with open(dict_path, "wb") as f:
        pickle.dump(results, f)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default='train.pkl')
    parser.add_argument('--eval_dir', type=str, default='eval.pkl')
    parser.add_argument('--test_dir', type=str, default='test.pkl')
    args = parser.parse_args()  

    build_dataset(args.train_dir, 'train_embedding.pkl')
    print('done train')
    build_dataset(args.eval_dir, 'eval_embedding.pkl')
    print('done eval')
    build_dataset(args.test_dir, 'test_embedding.pkl')
    print('all done')
