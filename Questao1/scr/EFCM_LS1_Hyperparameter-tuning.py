import sys,os,sys
sys.path.append('../')
import numpy as np
import pandas as pd
from copy import copy, deepcopy
from efcm.utils import generate_centers,generate_matrix_U
from efcm.optimizers import Efcmls1_Optimizer
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK,rand


if __name__ == '__main__':

    if sys.argv[2] == 'mr':
        dataset_names = ['dataset_01_raw_merge','dataset_02_raw_merge','dataset_03_raw_merge']
    if sys.argv[2] == 'tr':
        dataset_names =   ['dataset_01_raw_test','dataset_02_raw_test','dataset_03_raw_test']#

    if sys.argv[2] == 'ms':
        dataset_names =['dataset_01_scaled', 'dataset_02_scaled', 'dataset_03_scaled']
    if sys.argv[2] == 'trn':
        dataset_names = ['dataset_01_raw_test_notsort','dataset_02_raw_test_notsort','dataset_03_raw_test_notsort']
    
    try:
        data_idx = int(sys.argv[1])
    except:
        print('Dataset invalid')

    dataset = dataset_names[data_idx]
   

    data = pd.read_csv(os.path.join('../input/', '.'.join([dataset, 'csv'])))

    X_data = data.drop(columns='Target').values.copy()
    y_data = data.Target.values.copy()

    

    
    init='dirichlet'
    algorithm= rand.suggest#rand.suggest #tpe.suggest #rand.suggest #
    
    # random, dirichlet,laplace
    def objective(search_space):
        U_matrix_data = generate_matrix_U(n_samples=X_data.shape[0], n_cluster=7,init=init)
        params_dict_data = {'X':X_data, 'U':U_matrix_data,\
                           'G':None , 'n_cluster': 7,\
                           'T_u':None,'T_v':None,\
                           'tol_iter': 100}

        efcm_opt = Efcmls1_Optimizer()
        efcm_opt.parameters = params_dict_data
        
        tv = search_space['T_v']
        tu = (search_space['T_u'] ) /10
        eval_loss = efcm_opt.fit(T_u=tu,T_v=tv).eval()
        return {'loss':eval_loss , 'status': STATUS_OK}

    print('loading .. search_space ')
    search_space = {'T_u': hp.uniformint('T_u',5,30),'T_v': hp.uniformint('T_v', 10, 1000)}

    best_params = fmin(
      fn=objective,\
      space=search_space,\
      algo=algorithm,\
      max_evals=10)

    best_loss =  objective(best_params)['loss']
    best_T_u = best_params['T_u'] /10
    best_T_v = best_params['T_v']

    print('Best parameters T_u: {} T_v: {} Loss: {}'.format(best_T_u,best_T_v,best_loss))
    filename = '../run_best_params/'+dataset+'_bestparams_'+init+'_.txt'
    f = open(filename, "a")
    f.write("Dataset: {} \n Best_loss:{:.2f}\n Best_T_u: {}\n Best_T_v: {}\n".format(dataset, best_loss,best_T_u, best_T_v))
    f.close()

