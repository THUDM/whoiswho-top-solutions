from tqdm import tqdm, trange
import random
from gensim.models import Word2Vec
import numpy as np
import _pickle as pickle
import os
import networkx as nx
import time
from copy import copy, deepcopy

def read_graph(input_file):
    '''
    Reads the input network in networkx.
    '''
    G = nx.readwrite.edgelist.read_weighted_edgelist(input_file, nodetype=str, create_using=nx.Graph())
    # for edge in tqdm(G.edges(), desc='read graph'):
    #     G[edge[0]][edge[1]]['weight'] = 1

    # G = G.to_undirected()

    return G

def alias_setup(probs):
    '''
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    '''
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)
    return J, q

def alias_draw(J, q):
    '''
    Draw sample from a non-uniform discrete distribution using alias sampling.
    '''
    K = len(J)

    kk = int(np.floor(np.random.rand()*K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]

import pdb
# random.seed(21)
# np.random.seed(31)
class Graph():
    def __init__(self, nx_G, p, q):
        self.G = nx_G
        self.p = p
        self.q = q

    def node2vec_walk(self, walk_length, start_node):
        '''
        Simulate a random walk starting from start node.
        '''
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = sorted(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    next_node = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0], alias_edges[(prev, cur)][1])]
                    walk.append(next_node)
            else:
                break

        return walk

    def simulate_walks(self, num_walks, walk_length):
        '''
        Repeatedly simulate random walks from each node.
        '''
        G = self.G
        walks = []
        nodes = list(G.nodes())
        # print('Walk iteration:')
        for walk_iter in trange(num_walks, desc='walk iteration: '):
            # print(str(walk_iter+1), '/', str(num_walks))
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))

        return walks

    def get_alias_edge(self, src, dst):
        '''
        Get the alias edge setup lists for a given edge.
        '''
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for dst_nbr in sorted(G.neighbors(dst)):
            if dst_nbr == src:
                unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(G[dst][dst_nbr]['weight'])
            else:
                unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)
        norm_const = sum(unnormalized_probs)
        normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]

        return alias_setup(normalized_probs)

    def preprocess_transition_probs(self, config, filetype):
        '''
        Preprocessing of transition probabilities for guiding the random walks.
        '''
        G = self.G

        alias_nodes = {}
        if not os.path.exists(os.path.join(config, 'alias_nodes_new_{}.pkl'.format(filetype))):
            for node in tqdm(G.nodes, desc='nodes '):
                unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
                norm_const = sum(unnormalized_probs)
                normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
                alias_nodes[node] = alias_setup(normalized_probs)
            pickle.dump(alias_nodes, open(os.path.join(config, 'alias_nodes_new_{}.pkl'.format(filetype)), 'wb'))
        else:
            alias_nodes = pickle.load(open(os.path.join(config, 'alias_nodes_new_{}.pkl'.format(filetype)), 'rb'))

        if not os.path.exists(os.path.join(config, 'alias_edges_new_{}.pkl'.format(filetype))):
            alias_edges = {}
            for edge in tqdm(G.edges(), desc='edges '):
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

            pickle.dump(alias_edges, open(os.path.join(config, 'alias_edges_new_{}.pkl'.format(filetype)), 'wb'))
        else:
            alias_edges = pickle.load(open(os.path.join(config, 'alias_edges_new_{}.pkl'.format(filetype)), 'rb'))

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return

def learn_embeddings(walks, dimensions, window_size, iter_nums, config, filetype):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    # walks = [map(str, walk) for walk in walks]
    model = Word2Vec(walks, size=dimensions, window=window_size, min_count=0, sg=1, workers=32, iter=iter_nums)
    model.save(os.path.join(config, 'node2vec_whoiswho_{}.bin'.format(filetype)))
    model.wv.save_word2vec_format(os.path.join(config, 'node2vec_whoiswho_{}.emb'.format(filetype)))


nx_G = read_graph('../resource/node2vec/train_graph.txt')
config = '../resource/node2vec'
G = Graph(nx_G, 4, 1)
G.preprocess_transition_probs(config, 'train')
walks = G.simulate_walks(20, 100)

start = time.time()
learn_embeddings(walks, 100, 10, 10, config, 'train')
end = time.time()
running_time = end-start
print('time cost : %.5f sec' %running_time)