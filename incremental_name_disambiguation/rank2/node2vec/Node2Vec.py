#!/usr/bin/env python
# encoding: utf-8


import networkx as nx
from nodevectors import Node2Vec

# ================================= 分开图 ====================================
for graph in ('train', 'test', 'valid_unass', 'test_unass'):
    nx_G = nx.readwrite.edgelist.read_weighted_edgelist('./resource/node2vec/{}_graph.txt'.format(graph),
                                                        nodetype=str, create_using=nx.Graph())

    g2v = Node2Vec(
        n_components=32,
        walklen=20,
        epochs=20
    )
    g2v.fit(nx_G)
    g2v.save_vectors("./resource/node2vec/node2vec_fast_{}.bin".format(graph))

# ================================== 合并图 ====================================
nx_G = nx.readwrite.edgelist.read_weighted_edgelist('./resource/node2vec/all_graph.txt', nodetype=str, create_using=nx.Graph())
g2v = Node2Vec(
    n_components=32,
    walklen=50,
    epochs=20
)
g2v.fit(nx_G)
g2v.save_vectors("./resource/node2vec/node2vec_fast_all_50_walklen.bin")
