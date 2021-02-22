#!/usr/bin/env python
#encoding:utf-8


'''

@author: Hannah

@file: main.py

@time: 2019/7/23 9:00

'''
import lp

t = 10  # 迭代次数
p = 0.8  # 训练集边数占的比例
suf = str(p * 10)

DIR = [
    './data/Vickers',  # 0
    './data/CS',  # 1
    './data/CKM',  # 2
    './data/Lazega',  # 3
    './data/celegans',  # 4
    './data/TF'
]

name = [
    'Vickers',
    'cs',
    'ckm',
    'Lazega',
    'ce',
    'tf'
]

file = [
    '/Vickers-Chan-7thGraders_multiplex',
    '/CS-Aarhus_multiplex',
    '/CKM-Physicians-Innovation_multiplex',
    '/Lazega-Law-Firm_multiplex',
    '/celegans_connectome_multiplex',
    '/TF'
]

results = [
    './results/_Vickers',
    './results/_CS',
    './results/_CKM',
    './results/_Lazega',
    './results/_celegans',
    './results/_TF'
]

method_name = [
    'ours'
]

# 单个指标还是多个指标的结合
sim_method = [
    ['CN', 'RA', 'CCLP', 'JC', 'PA', 'local', 'CC', 'path3', 'hasedge'],  # 0
]

net_id = [0, 1, 2, 3, 4, 5]

graph_file_list = []  # 网络文件列表
file_list = []
name_list = []
result_file_list = []  # 结果文件列表

for i in net_id:
    graph_file_list.append(DIR[i])
    file_list.append(file[i])
    name_list.append(name[i])
    result_file_list.append(results[i])

# 实验中使用的方法的id
method_ids = [0]

# 按照数据集，分别计算
for i in range(len(graph_file_list)):
    _DIR = graph_file_list[i]
    _file = _DIR + file_list[i]
    result_file = result_file_list[i] + str(suf)
    out_file = open(result_file, 'w')
    print(_DIR)
    out_file.write(
        'Method\talpha\tacc_avg_svm\tpre_avg_svm\trecall_avg_svm\tf1_avg_svm\tMAE_avg_svm\tRMSE_avg_svm\t'
        'acc_avg_rf\tpre_avg_rf\trecall_avg_rf\tf1_avg_rf\tMAE_avg_rf\tRMSE_avg_rf\t'
        'acc_avg_Ada\tpre_avg_Ada\trecall_avg_Ada\tf1_avg_Ada\tMAE_avg_Ada\tRMSE_avg_Ada\t')

    out_file.write('\n')

    for index in method_ids:
        method = method_name[index]  # 方法名称
        method_mix = sim_method[index]
        print(method)  # 方法名称
        lp.LP(_DIR, _file, out_file, t, p, method, method_mix)
        out_file.flush()
    out_file.close()
