#!/usr/bin/env python
#encoding:utf-8

"""

@author: Hannah

@file: some_fun.py

@time: 2019/6/13 16:56

"""

import networkx as nx
import numpy as np
import feature

# 把一些需要反复调用的函数提取到这里来
def count_nodenum(file, layer_num):
    """
        使用networkx读网络时，edgelist文件里的孤立节点是读不到的。
        为了保持各层节点统一，向网络中addnode；这样做其实也是为了在存储节点特征时更加方便
    """
    a = []  # 统计每层网络中的最大节点号
    for i in range(layer_num):
        k = i + 1
        filename = file + str(k) + '.edgelist'
        G = nx.read_edgelist(filename, nodetype=int)
        b = []
        for j in nx.nodes(G):
            b.append(j)
        max1 = max(b)
        a.append(max1)
    node_index_max = max(a)  # 找到各层网络中节点的最大编号，目的是为了向网络中补填节点
    return node_index_max

# 将数字对应到节点对
def oneindex2xy(index, num_nodes):
    index = index + 1
    for i in range(1, num_nodes, 1):
        if index <= i * (i + 1) / 2:  # 第x位在第i行
            break
        else:
            continue
    y = i
    index = index - 1
    x = int((i - 1) - ((i * (i + 1) / 2 - 1) - index))
    return x, y


# 把一堆数字转换为节点对下标
def index2xy(random_list, num_nodes):
    # print(random_list)
    E = []  # EU中节点对的下标编号
    for index in random_list:
        a = []
        index = index + 1
        for i in range(1, num_nodes, 1):
            if index <= i * (i + 1) / 2:  # 第x位在第i行
                break
            else:
                continue
        y = i
        index = index - 1
        x = int((i - 1) - ((i * (i + 1) / 2 - 1) - index))
        a.append(x)
        a.append(y)
        # E的存储形式为：[[x1,y1],[x2,y2],[x3,y3],....]
        E.append(a)
    return E

def pair(x, y):
    if x < y:
        return (x, y)
    else:
        return (y, x)

def stats(value_list):
    value_array = np.array(value_list)
    avg = np.mean(value_array)
    std = np.std(value_array)
    return avg, std

# 原始特征集
def add_fea_label(Ga, training_graph, file, random_list, alpha, layer_num, method_mix, node_num, _flag):
    feature_list = []
    label_list = []
    for index in random_list:
        fea_list = []
        u, v = oneindex2xy(index, node_num)
        x = u + 1
        y = v + 1
        for layer_id in range(layer_num):  # 遍历各层
            beta = layer_id + 1
            filename_b = file + str(beta) + '.edgelist'
            Gb = nx.read_edgelist(filename_b, nodetype=int)
            for meth in method_mix:
                if beta == alpha and meth != 'hasedge':
                    fea = feature.sim(training_graph, meth, x, y)
                    fea_list.append(fea)
                elif beta == alpha and meth == 'hasedge':
                    continue
                else:
                    fea = feature.sim(Gb, meth, x, y)
                    fea_list.append(fea)
        feature_list.append(fea_list)
        if _flag == 0:
            if (x, y) in Ga.edges():
                label = 1
            else:
                label = 0
            label_list.append(label)
        if _flag == 1:
            if (x, y) in training_graph.edges():
                label = 1
            else:
                label = 0
            label_list.append(label)
    return feature_list, label_list