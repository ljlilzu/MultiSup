#!/usr/bin/env python

# -*- coding: UTF-8 -*-

'''

@author: Hannah

@file: feature.py

@time: 2019/6/17 15:19

'''

import networkx as nx
import numpy as np

def sim(G, method, x, y):
    if method == 'CN':
        return CN_index(G, x, y)
    if method == 'RA':
        return RA_index(G, x, y)
    if method == 'JC':
        return JC_index(G, x, y)
    if method == 'PA':
        return PA_index(G, x, y)
    if method == 'local':
        return local_assortativity(G, x, y)
    if method == 'CC':
        return CC_index(G, x, y)
    if method == 'CCLP':
        return CCLP_index(G, x, y)
    if method == 'hasedge':
        return has_edge(G, x, y)
    if method == 'path3':
        return path3(G, x, y)

def CN_index(G,x,y):
    if x not in G:
        G.add_node(x)
    if y not in G:
        G.add_node(y)
    CN = len(list(nx.common_neighbors(G, x, y)))
    return CN

def RA_index(G,x,y):
    if x not in G:
        G.add_node(x)
    if y not in G:
        G.add_node(y)
    RA = 0
    for w in nx.common_neighbors(G, x, y):
        RA += 1 / nx.degree(G, w)
    return RA

def JC_index(G,x,y):
    if x not in G:
        G.add_node(x)
    if y not in G:
        G.add_node(y)
    CN = len(list(nx.common_neighbors(G,x,y)))
    if CN > 0:
        JC = CN / (nx.degree(G, x) + nx.degree(G, y) - CN)
    else:
        JC = 0
    return JC

def PA_index(G,x,y):
    if x not in G:
        G.add_node(x)
    if y not in G:
        G.add_node(y)
    PA = nx.degree(G, x) * nx.degree(G, y)
    return PA

def local_assortativity(G,x,y):
    if x not in G:
        G.add_node(x)
    if y not in G:
        G.add_node(y)
    if nx.degree(G, x) == 0 and nx.degree(G, y) == 0:
        local_assortativity = 0
    else:
        local_assortativity = (4 * nx.degree(G, x) * nx.degree(G, y) - nx.degree(G, x) - nx.degree(G, y)) \
                              / (2 * np.square(nx.degree(G, x)) + 2 * np.square(nx.degree(G, y)) - nx.degree(G,x) - nx.degree(G, y))
    return local_assortativity

def CC_index(G,x,y):
    if x not in G:
        G.add_node(x)
    if y not in G:
        G.add_node(y)
    mean_CC = (nx.clustering(G, x) + nx.clustering(G, y)) / 2
    return mean_CC

def CCLP_index(G,x,y):
    if x not in G:
        G.add_node(x)
    if y not in G:
        G.add_node(y)
    CCLP = 0
    for w in nx.common_neighbors(G, x, y):
        CCLP += nx.clustering(G, w)
    return CCLP

def path3(G,x,y):
    if x not in G:
        G.add_node(x)
    if y not in G:
        G.add_node(y)

    path3 = 0
    for u in nx.neighbors(G, x):
        for v in nx.neighbors(G, u):
            if x != v and u != y and y in nx.neighbors(G, v):
                path3 += 1
    return path3

def has_edge(G, x, y):
    if x not in G:
        G.add_node(x)
    if y not in G:
        G.add_node(y)
    if (x, y) in G.edges():
        return 1
    else:
        return 0

def pair(x, y):
    if (x < y):
        return (x, y)
    else:
        return (y, x)
