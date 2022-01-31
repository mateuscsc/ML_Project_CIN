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
    dataset_names = ['dataset_01_scaled', 'dataset_02_scaled', 'dataset_03_scaled']
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

    for _ in range(50):
        clf = EFCM_LS1(n_cluster=7, theta_u=.43, theta_v=10, epsilon=1e-10, tol_iter=150, init='laplace')
        clf.fit(X_data_01)

        if (clf.loss < best_dataset1['objective']):
            best_dataset1['fuzzy'] = clf.fuzzy_matrix       # Shape to U=(N, C)
            best_dataset1['crisp'] = clf.labels_            # Get labels
            best_dataset1['objective'] = clf.loss           # Get objective function value

    '''for _ in range(50):
        _, u, _, _, jm, _, _ = fuzz.cmeans(dataset2.T, 7, 2, 0.005, 150)

        if (jm[-1] < best_dataset2['objective']):
            best_dataset2['fuzzy'] = u.T                    # Shape to U=(N, C)
            best_dataset2['crisp'] = u.T.argmax(axis=1)     # Get labels
            best_dataset2['objective'] = jm[-1]             # Get objective function value

    for _ in range(50):
        _, u, _, _, jm, _, _ = fuzz.cmeans(dataset3.T, 7, 2, 0.005, 150)

        if (jm[-1] < best_dataset3['objective']):
            best_dataset3['fuzzy'] = u.T                    # Shape to U=(N, C)
            best_dataset3['crisp'] = u.T.argmax(axis=1)     # Get labels
            best_dataset3['objective'] = jm[-1]             # Get objective function value
    '''
    # Modified Partition Coefficient
    # Between 0 and 1. 
    # 0 = Maximum fuzziness
    # 1 = Hard partition
    best_dataset1['mpc'] = MPC(best_dataset1['fuzzy'])
    #best_dataset2['mpc'] = MPC(best_dataset2['fuzzy'])
    #best_dataset3['mpc'] = MPC(best_dataset3['fuzzy'])

    # Partition Entropy
    # Between 0 and log(C). 
    # The closer the value of PE to 0, the crisper the clustering is. 
    # The index value close to the upper bound indicates the absence of any
    # clustering structure in the datasets or inability of the algorithm to extract it.
    best_dataset1['pe'] = PE(best_dataset1['fuzzy'])
    #best_dataset2['pe'] = PE(best_dataset2['fuzzy'])
    #best_dataset3['pe'] = PE(best_dataset3['fuzzy'])

    # Rand
    # Between 0 and 1. 
    # 0 = Random independently of the number of clusters and samples.
    # 1 = When the clusterings are identical (up to a permutation).
    #rand_d1_d2 = adjusted_rand_score(best_dataset1['crisp'], best_dataset2['crisp']) 
    #rand_d1_d3 = adjusted_rand_score(best_dataset1['crisp'], best_dataset3['crisp']) 
    #rand_d2_d3 = adjusted_rand_score(best_dataset2['crisp'], best_dataset3['crisp'])

    # F-Measure
    # Between 0 and 1.
    # 0 = Worse
    # 1 = Better
    #fm_d1_d2 = FMeasure(best_dataset1['crisp'], best_dataset2['crisp']) 
    #fm_d1_d3 = FMeasure(best_dataset1['crisp'], best_dataset3['crisp']) 
    #fm_d2_d3 = FMeasure(best_dataset2['crisp'], best_dataset3['crisp'])

    # Print results
    print("Dataset 1")
    print("MPC: {}\tPE: {}".format(best_dataset1['mpc'], best_dataset1['pe']))
    # print("Dataset 2")
    # print("MPC: {}\tPE: {}".format(best_dataset2['mpc'], best_dataset2['pe']))
    # print("Dataset 3")
    # print("MPC: {}\tPE: {}".format(best_dataset3['mpc'], best_dataset3['pe']))

    # print("\nAdjusted Rand Score")

    # print("Dataset 1 x Dataset 2: {}".format(rand_d1_d2))
    # print("Dataset 1 x Dataset 3: {}".format(rand_d1_d3))
    # print("Dataset 2 x Dataset 3: {}".format(rand_d2_d3))

    # print("\nF-Measure")

    # print("Dataset 1 x Dataset 2: {}".format(fm_d1_d2))
    # print("Dataset 1 x Dataset 3: {}".format(fm_d1_d3))
    # print("Dataset 2 x Dataset 3: {}".format(fm_d2_d3))
    

if __name__ == "__main__":
    main()