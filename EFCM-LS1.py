from collections import Counter
from copy import deepcopy
import sys
import pandas as pd
import numpy as np
from code.efcm.cluster import EFCM_LS1
from sklearn.metrics import adjusted_rand_score
from metrics.f_measure import FMeasure
from metrics.partition_entropy import PE
from metrics.modified_partition_coefficient import MPC
import os

def main():
    dataset_names =['dataset_01_raw_test','dataset_02_raw_test','dataset_03_raw_test']
    data_01 = pd.read_csv(os.path.join('./code/input/', '.'.join([dataset_names[0], 'csv'])))
    data_02 = pd.read_csv(os.path.join('./code/input/', '.'.join([dataset_names[1], 'csv'])))
    data_03 = pd.read_csv(os.path.join('./code/input/', '.'.join([dataset_names[2], 'csv'])))

    X_data_01 = data_01.drop(columns='Target').values.copy()
    y_data_01 = data_01.Target.values.copy()
    X_data_02 = data_02.drop(columns='Target').values.copy()
    y_data_02 = data_02.Target.values.copy()
    X_data_03 = data_03.drop(columns='Target').values.copy()
    y_data_03 = data_03.Target.values.copy()

    best_dataset1 = { 'objective': sys.maxsize, 'crisp': None, 'fuzzy': None, 'mpc': None, 'pe': None }
    best_dataset2 = { 'objective': sys.maxsize, 'crisp': None, 'fuzzy': None, 'mpc': None, 'pe': None }
    best_dataset3 = { 'objective': sys.maxsize, 'crisp': None, 'fuzzy': None, 'mpc': None, 'pe': None }

    NUM_EPOCHS = 2

    # Dataset1
    '''clf = EFCM_LS1(n_cluster=7,theta_u=.9 ,theta_v=30,epsilon=1e-10,tol_iter=150,init='dirichlet')
    u = []
    model = []
    _ = [model.append(deepcopy(clf)) for _ in range(NUM_EPOCHS)]
    for j in range(NUM_EPOCHS):
        model[j].fit(X_data_01) # X_data_01,X_data_02,X_data_03
        print('run:{} loss: {} '.format(j,model[j].loss))
        u.append(model[j].loss)

    best_dataset1['fuzzy'] = model[np.argmin(u)].fuzzy_matrix
    best_dataset1['crisp'] = model[np.argmin(u)].labels_

    # Dataset2
    clf = EFCM_LS1(n_cluster=7,theta_u=1.0 ,theta_v=1000.0,epsilon=1e-10,tol_iter=150,init='dirichlet')
    u = []
    model = []
    _ = [model.append(deepcopy(clf)) for _ in range(NUM_EPOCHS)]
    for j in range(NUM_EPOCHS):
        model[j].fit(X_data_02) # X_data_01,X_data_02,X_data_03
        print('run:{} loss: {} '.format(j,model[j].loss))
        u.append(model[j].loss)

    best_dataset2['fuzzy'] = model[np.argmin(u)].fuzzy_matrix
    best_dataset2['crisp'] = model[np.argmin(u)].labels_'''

    # Dataset3
    clf = EFCM_LS1(n_cluster=7,theta_u=5.0 ,theta_v=10,epsilon=1e-10,tol_iter=150,init='dirichlet')
    u = []
    model = []
    _ = [model.append(deepcopy(clf)) for _ in range(NUM_EPOCHS)]
    for j in range(NUM_EPOCHS):
        model[j].fit(X_data_03) # X_data_01,X_data_02,X_data_03
        print('run:{} loss: {} '.format(j,model[j].loss))
        u.append(model[j].loss)

    best_dataset3['fuzzy'] = model[np.argmin(u)].fuzzy_matrix
    best_dataset3['crisp'] = model[np.argmin(u)].labels_
    print(Counter(best_dataset3['crisp']))
    input()

    # Modified Partition Coefficient
    # Between 0 and 1. 
    # 0 = Maximum fuzziness
    # 1 = Hard partition
    #best_dataset1['mpc'] = MPC(best_dataset1['fuzzy'])
    #best_dataset2['mpc'] = MPC(best_dataset2['fuzzy'])
    best_dataset3['mpc'] = MPC(best_dataset3['fuzzy'])

    # Partition Entropy
    # Between 0 and log(C). 
    # The closer the value of PE to 0, the crisper the clustering is. 
    # The index value close to the upper bound indicates the absence of any
    # clustering structure in the datasets or inability of the algorithm to extract it.
    #best_dataset1['pe'] = PE(best_dataset1['fuzzy'])
    #best_dataset2['pe'] = PE(best_dataset2['fuzzy'])
    best_dataset3['pe'] = PE(best_dataset3['fuzzy'])

    # Rand
    # Between 0 and 1. 
    # 0 = Random independently of the number of clusters and samples.
    # 1 = When the clusterings are identical (up to a permutation).
    #rand_d1 = adjusted_rand_score(y_data_01, best_dataset1['crisp']) 
    #rand_d2 = adjusted_rand_score(y_data_02, best_dataset2['crisp']) 
    rand_d3 = adjusted_rand_score(y_data_03, best_dataset3['crisp'])

    # F-Measure
    # Between 0 and 1.
    # 0 = Worse
    # 1 = Better
    #fm_d1 = FMeasure(y_data_01, best_dataset1['crisp']) 
    #fm_d2 = FMeasure(y_data_02, best_dataset2['crisp']) 
    fm_d3 = FMeasure(y_data_03, best_dataset3['crisp'])

    # Print results
    #print("Dataset 1")
    #print("MPC: {}\tPE: {}".format(best_dataset1['mpc'], best_dataset1['pe']))
    # print("Dataset 2")
    # print("MPC: {}\tPE: {}".format(best_dataset2['mpc'], best_dataset2['pe']))
    print("Dataset 3")
    print("MPC: {}\tPE: {}".format(best_dataset3['mpc'], best_dataset3['pe']))

    print("\nAdjusted Rand Score")

    #print("Dataset 1: {}".format(rand_d1))
    # print("Dataset 2: {}".format(rand_d2))
    print("Dataset 3: {}".format(rand_d3))

    print("\nF-Measure")

    #print("Dataset 1: {}".format(fm_d1))
    # print("Dataset 2: {}".format(fm_d2))
    print("Dataset 3: {}".format(fm_d3))

if __name__ == "__main__":
    main()