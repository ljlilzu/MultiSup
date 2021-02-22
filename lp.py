#!/usr/bin/env python
#encoding:utf-8
'''

@author: Hannah

@file: lp.py

@time: 2019/7/23 13:02

'''
import os
import math
import random
import warnings
import numpy as np
import networkx as nx
from sklearn import svm
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
import some_fun
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def stats(value_list):
    value_array = np.array(value_list)
    avg = np.mean(value_array)
    std = np.std(value_array)
    return avg,std

def LP(DIR, file, out_file, t, p, method_name, method_mix):
    layer_num = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
    node_num = some_fun.count_nodenum(file, layer_num)  # 节点数量

    for layer_id in range(layer_num):  # 遍历每一层
        alpha = layer_id + 1  # alpha目标层
        print('alpha = ', alpha)
        filename = file + str(alpha) + '.edgelist'
        Ga = nx.read_edgelist(filename, nodetype=int)
        # 将网络中存在的边分为ET和EP两部分，比例为8：2
        num_E = nx.number_of_edges(Ga)  # E的个数
        num_ET = int(num_E * p)
        num_EP = num_E - num_ET

        num_U = int(((node_num - 1) * node_num) / 2)  # 网络中所有可能存在的边的边数

        # 目标层网络中存在边的边编号
        random_list_E = []
        fr = open(filename)
        arrayOLines = fr.readlines()
        for line in arrayOLines:
            line = line.strip()  # 去掉首尾空格
            listFromLine = line.split(' ')
            x = int(listFromLine[0]) - 1
            y = int(listFromLine[1]) - 1
            index = int((y - 1) * y / 2 - 1 + x + 1)
            random_list_E.append(index)

        acc_list = []  # 存储svm算法得到的度量值
        pre_list = []
        recall_list = []
        f1_list = []
        MAE_list = []
        RMSE_list = []

        acc_list1 = []  # 存储随机森林算法得到的度量值
        pre_list1 = []
        recall_list1 = []
        f1_list1 = []
        MAE_list1 = []
        RMSE_list1 = []

        acc_list2 = []  # 存储Adaboost算法得到的度量值
        pre_list2 = []
        recall_list2 = []
        f1_list2 = []
        MAE_list2 = []
        RMSE_list2 = []

        for iter in range(t):  # 10
            print('iter = ', iter)
            # 有很令人讨厌的警告提示
            warnings.filterwarnings('ignore')

            # =================每次迭代都生成一次training_graph==================
            # =================将training_graph中的边用边编号表示==================
            seed = math.sqrt(num_E * node_num) + math.pow((1 + iter) * 10, 3)  # 随机数种子
            random.seed(seed)

            # 构造训练集
            # 所有可能存在的边U
            # 将网络中的存在边分为两部分：ET和EP，其中ET作为训练集的一部分；
            # 然后从U-EP里随机取|ET|条边——EU
            # 训练集 D = ET ∪ EU
            # 从U-D中随机抽取|EP|条边——EU_
            # 测试集 P = EP ∪ EU_
            # 从所有可能的边里随机选边的时候，可以按“链接编号”来取，再由链接编号转换为节点对编号

            random_list_ET = random.sample(random_list_E, num_ET)  # training_graph中的边
            random_list_EP = [item for item in random_list_E if item not in random_list_ET]  # Ga中删掉的边
            U_EP = [item for item in range(num_U) if item not in random_list_EP]  # U-EP
            random_list_EU = random.sample(U_EP, num_ET)
            random_list_D = list(set(random_list_EU).union(set(random_list_ET)))  # D = ET ∪ EU
            U_D = [item for item in range(num_U) if item not in random_list_D]  # U-D
            random_list_EU_ = random.sample(U_D, num_EP)
            random_list_P = list(set(random_list_EU_).union(set(random_list_EP)))
            random_list_D = sorted(random_list_D)
            random_list_P = sorted(random_list_P)

            # training_graph中的边只有ET
            training_graph = nx.Graph()
            training_graph.add_nodes_from(range(1, node_num+1, 1))  # 训练网络中的节点从1开始
            for edge_i in random_list_ET:
                u, v = some_fun.oneindex2xy(edge_i, node_num)
                x = u + 1
                y = v + 1
                training_graph.add_edge(x, y)
            training_graph.to_undirected()

            Xtrain, Ytrain = some_fun.add_fea_label(Ga, training_graph, file, random_list_D, alpha, layer_num, method_mix, node_num, 1)
            Xtest, Ytest = some_fun.add_fea_label(Ga, training_graph, file, random_list_P, alpha, layer_num, method_mix, node_num, 0)

            # 调用RandomForest分类器
            clf1 = RandomForestClassifier(random_state=4, max_features=0.8)
            clf1.fit(Xtrain, Ytrain)
            result1 = clf1.predict(Xtest)
            result1 = result1.tolist()
            result1 = list(map(int, result1))

            clf2 = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(random_state=4, max_depth=2), algorithm='SAMME', n_estimators= 50, learning_rate = 0.7 )
            clf2.fit(Xtrain, Ytrain)
            result2 = clf2.predict(Xtest)
            result2 = result2.tolist()
            result2 = list(map(int, result2))

            # 调用SVM方法前先对特征进行标准化处理
            _Xtrain = np.array(Xtrain)
            _Xtest = np.array(Xtest)
            max_train = np.amax(_Xtrain, axis=0)  # 列的最大值
            max_test = np.amax(_Xtest, axis=0)
            min_train = np.amin(_Xtrain, axis=0)
            min_test = np.amin(_Xtest, axis=0)

            for r_i in range(len(Xtrain)):
                for c_j in range(len(Xtrain[0])):
                    a = max_train[c_j]
                    i = min_train[c_j]
                    if a == i:
                        Xtrain[r_i][c_j] = 1
                    else:
                        Xtrain[r_i][c_j] = (Xtrain[r_i][c_j] - i) / (a - i)

            for r_i in range(len(Xtest)):
                for c_j in range(len(Xtest[0])):
                    a = max_test[c_j]
                    i = min_test[c_j]
                    if a == i:
                        Xtrain[r_i][c_j] = 1
                    else:
                        Xtest[r_i][c_j] = (Xtest[r_i][c_j] - i) / (a - i)

            # 调用SVM分类器
            # 高斯核函数的惩罚参数C和gamma参数非常重要
            from sklearn.model_selection import GridSearchCV
            clf = svm.SVC(random_state=4, kernel='rbf', probability=True)
            param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.01, 0.001, 0.0001]}
            grid_search = GridSearchCV(clf, param_grid, n_jobs=8, verbose=1)
            grid_search.fit(Xtrain, Ytrain)
            best_parameters = grid_search.best_estimator_.get_params()
            clf = svm.SVC(random_state=4, kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'],
                          probability=True)
            clf.fit(Xtrain, Ytrain)
            result = clf.predict(Xtest)
            result = result.tolist()
            result = list(map(int, result))

            acc = metrics.accuracy_score(Ytest, result)
            acc_list.append(acc)
            pre = metrics.precision_score(Ytest, result)
            pre_list.append(pre)
            recall = metrics.recall_score(Ytest, result)
            recall_list.append(recall)
            f1 = metrics.f1_score(Ytest, result)
            f1_list.append(f1)
            MAE = metrics.mean_absolute_error(Ytest, result)
            MAE_list.append(MAE)
            MSE = metrics.mean_squared_error(Ytest, result)
            RMSE = MSE ** 0.5
            RMSE_list.append(RMSE)

            acc1 = metrics.accuracy_score(Ytest, result1)
            acc_list1.append(acc1)
            pre1 = metrics.precision_score(Ytest, result1)
            pre_list1.append(pre1)
            recall1 = metrics.recall_score(Ytest, result1)
            recall_list1.append(recall1)
            f11 = metrics.f1_score(Ytest, result1)
            f1_list1.append(f11)
            MAE1 = metrics.mean_absolute_error(Ytest, result1)
            MAE_list1.append(MAE1)
            MSE1 = metrics.mean_squared_error(Ytest, result1)
            RMSE1 = MSE1 ** 0.5
            RMSE_list1.append(RMSE1)

            acc2 = metrics.accuracy_score(Ytest, result2)
            acc_list2.append(acc2)
            pre2 = metrics.precision_score(Ytest, result2)
            pre_list2.append(pre2)
            recall2 = metrics.recall_score(Ytest, result2)
            recall_list2.append(recall2)
            f12 = metrics.f1_score(Ytest, result2)
            f1_list2.append(f12)
            MAE2 = metrics.mean_absolute_error(Ytest, result2)
            MAE_list2.append(MAE2)
            MSE2 = metrics.mean_squared_error(Ytest, result2)
            RMSE2 = MSE2 ** 0.5
            RMSE_list2.append(RMSE2)

        # SVM分类算法所得结果
        acc_avg, acc_std = stats(acc_list)
        pre_avg, pre_std = stats(pre_list)
        recall_avg, recall_std = stats(recall_list)
        f1_avg, f1_std = stats(f1_list)
        MAE_avg, MAE_std = stats(MAE_list)
        RMSE_avg, RMSE_std = stats(RMSE_list)

        # 随机森林算法所得结果
        acc_avg1, acc_std1 = stats(acc_list1)
        pre_avg1, pre_std1 = stats(pre_list1)
        recall_avg1, recall_std1 = stats(recall_list1)
        f1_avg1, f1_std1 = stats(f1_list1)
        MAE_avg1, MAE_std1 = stats(MAE_list1)
        RMSE_avg1, RMSE_std1 = stats(RMSE_list1)

        # Adaboost算法所得结果
        acc_avg2, acc_std2 = stats(acc_list2)
        pre_avg2, pre_std2 = stats(pre_list2)
        recall_avg2, recall_std2 = stats(recall_list2)
        f1_avg2, f1_std2 = stats(f1_list2)
        MAE_avg2, MAE_std2 = stats(MAE_list2)
        RMSE_avg2, RMSE_std2 = stats(RMSE_list2)
        # auc_avg2, auc_std2 = stats(auc_list2)

        out_file.write(method_name + '\t')
        out_file.write('%d\t' % (alpha))

        out_file.write('%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t' % (
            acc_avg, pre_avg, recall_avg, f1_avg, MAE_avg, RMSE_avg))

        out_file.write('%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t' % (
            acc_avg1, pre_avg1, recall_avg1, f1_avg1, MAE_avg1, RMSE_avg1))

        out_file.write('%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t' % (
            acc_avg2, pre_avg2, recall_avg2, f1_avg2, MAE_avg2, RMSE_avg2))

        out_file.write('\n')
        print('========================================================================')
    out_file.write('\n')
    out_file.write('\n')