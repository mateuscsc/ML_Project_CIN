
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
# Custom module
from ._base import BaseCluster
############################# 
import numpy as np
import math
from copy import copy,deepcopy
import pandas as pd
from time import sleep

class EFCM_LS1(BaseCluster):
    
    def __init__(self, n_cluster, m='auto', theta_u=3, theta_v=3,
                 epsilon=1e-5, tol_iter=150, verbose=0, init='laplace'):
        
        super(EFCM_LS1, self).__init__(n_cluster, m)
        assert theta_u > 0, "theta_u must be larger than 0.01"
        assert theta_v > 0, "theta_v must be larger than 0"

        self.theta_u = theta_u
        self.theta_v = theta_v
        self.epsilon = epsilon

        self.tol_iter = tol_iter
        self.n_dim = None
        self.verbose = verbose
        self.fitted = False
        self.init_method = init
        # 
        self.U,self.prev_U, self.V, self.center_ = None, None, None,None
        

    def __configure(self):
        self.U,self.prev_U, self.V, self.center_ = None, None, None,None

    def reset(self):
        self.__configure()

    def get_params(self, deep=True):
        return {
            'n_cluster': self.n_cluster,
            'theta_u': self.theta_u,
            'theta_v': self.theta_v,
            'max_iter': self.tol_iter,
            'epsilon': self.epsilon,
            'verbose': self.verbose,
        }

    def set_params(self, **params):
        for p, v in params.items():
            setattr(self, p, v)
        return self

    def fit(self, X, y=None):
        """

        :param X: shape: [n_samples, n_features]
        :param y:
        :return:
        """
        self.reset()
        self._data = X.copy() #:: TODO 

        if y is not None:
            y = np.array(y, dtype=np.float64)

        self.n_dim = self._data.shape[1]
        self.n_samples = self._data.shape[0]
        self.center_ = self._data[np.random.choice(np.arange(self.n_samples), replace=False, size=self.n_cluster), :].copy()

        if self.init_method =='dirichlet':

            self.U = np.random.dirichlet(np.ones(self.n_cluster),size=self.n_samples)
            self.U = np.fmax(self.U, 0.000000001)
            self.U = np.fmin(self.U, 1.00)
        elif self.init_method == 'random':
            self.U = np.random.randint(self.n_samples, size=(self.n_samples, self.n_cluster))
            self.U = Normalizer(norm='l1').fit_transform(self.U)
            self.U = np.fmax(self.U, 0.000000001)
            self.U = np.fmin(self.U, 1.00)

        elif self.init_method == 'laplace' :

            for _ in range(self.n_cluster):

                try:
                    self.U = np.column_stack((self.U, abs(np.random.laplace(0, 1, size=(self.n_samples, 1))) ))
                except:
                    self.U = abs(np.random.laplace(0, 1, size=(self.n_samples, 1)))

            self.U = Normalizer(norm='l1').fit_transform(self.U)
            self.U = np.fmax(self.U, 0.000000001)
            self.U = np.fmin(self.U, 1.00)

        else:
            raise ValueError('init method only supports [random, dirichlet,laplace]')
            
        _loss = []
        for i in range(self.tol_iter):
            
            self.center_ = self.update_g
            self.V = self.update_v

            if i > 0 and  np.linalg.norm(self.U - self.prev_U,ord=1) < self.epsilon:
                break

            else:
                self.prev_U = self.U.copy()
                prev_C = self.center_.copy()
                _loss.append(self.overall_loss)
                self.U = self.update_u

        self._loss = _loss
        self.fitted = True
        return self

    @property
    def update_g(self):
        #
        _X = self._data.copy()
        n_split_u = self.n_cluster
        u_i = np.hsplit(self.U, n_split_u)
        center = [[] for _ in range(n_split_u)]
        k=0
        m = _X.shape[1]
        for u_ik in u_i:
            j = 0
            b = []
            x_i = np.hsplit(_X, m)
            while j < m:
                _y = u_ik * x_i[j]
                _y = np.sort(_y,axis=0)
                u_ik = np.sort(u_ik,axis=0)
                s = - np.linalg.norm(u_ik,ord=1) 
                _r = 0
                
                while s<0:
                    _r += 1
                    s += 2* abs(u_ik[_r])
                j+=1
                
                try:
                    if s == 0 and (s+ 2* abs(u_ik[_r+1]) ==0 ):
                        a_r = (_y[_r]/u_ik[_r]) + (_y[_r+1]/u_ik[_r+1]) #
                        a_r = a_r/2
                        b.append(a_r)
                    else:
                        b.append(_y[_r]/u_ik[_r])
                except:
                    b.append(_y[_r]/u_ik[_r])
            
            center[k].extend(np.array(b).ravel())
            k+=1
        
        return np.asarray(center)

    @property
    def update_v(self):
        #
        _X = self._data.copy()
        u = self.U.copy()
        _G = self.center_.copy()

        n_split_u = u.shape[1]
        u_i = np.hsplit(u, n_split_u)
        m = n_split_u
        V = []
        g_i = np.vsplit(_G, m)
        k = 0
        
        for u_ik in u_i:
            num = np.sum(u_ik*np.absolute((_X-g_i[k])),axis=0)
            num = np.fmax(num, np.finfo(np.float64).eps)
            num = np.fmin(num, np.finfo(np.float64).max)
            num = np.exp(- num / self.theta_v) 
            num = np.fmax(num, np.finfo(np.float64).eps)
            num = np.fmin(num, np.finfo(np.float64).max)
            v_kj = num/ num.sum()
            v_kj = np.fmax(v_kj, np.finfo(np.float64).eps)
            v_kj = np.fmin(v_kj, np.finfo(np.float64).max)
            V.append(v_kj)
            k +=1

        return np.asarray(V)

    @property
    def update_u(self):
        #
        _X = self._data.copy()
        _G = self.center_.copy()
        n_split_g = _G.shape[0]
        g_i = np.vsplit(_G, n_split_g)
        m = n_split_g
        U = []
        for g_ik in g_i:

            dist_num = np.linalg.norm(np.expand_dims(_X, 2)-np.transpose(g_ik), ord=1, axis=1, keepdims=False)
            dist_num = - dist_num / self.theta_u
            dist_num = np.exp(dist_num)
            dist_num = np.fmax(dist_num, np.finfo(np.float64).eps)  # avoid underflow
            dist_num = np.fmin(dist_num, np.finfo(np.float64).max)

            dist_den = np.linalg.norm(np.expand_dims(_X, 2)-np.transpose(_G), ord=1, axis=1, keepdims=False)
            dist_den = - dist_den / self.theta_u
            dist_den = np.exp(dist_den)
            dist_den = np.fmax(dist_den, np.finfo(np.float64).eps)  # avoid underflow
            dist_den = np.fmin(dist_den, np.finfo(np.float64).max)

            u_k = dist_num/np.sum(dist_den,axis=1, keepdims=True)
            u_k = np.fmax(u_k, np.finfo(np.float64).eps)  # avoid underflow
            u_k = np.fmin(u_k, np.finfo(np.float64).max)
            U.append(u_k.flatten())

        U_arr = np.asarray(U)

        return U_arr.T

    @property    
    def overall_loss(self):
        #
        _X = self._data.copy()
        _G = self.center_.copy()
        _V = self.V.copy()
        _U = self.prev_U.copy()

        n_cluster = _G.shape[0]
        g_i = np.vsplit(_G, n_cluster)
        v_i = np.vsplit(_V, n_cluster)
        #####
        u_i = np.hsplit(_U, n_cluster)
        arr,arr_u,arr_v = None,None,None
        k = 0

        while k < n_cluster:
            
            v_dif = np.sum(v_i[k]*np.absolute(_X-g_i[k]),axis=1, keepdims=True)
            E_u_diff = u_i[k]*np.log(u_i[k])
            E_u_diff = np.fmax(E_u_diff, np.finfo(np.float64).eps)  # avoid underflow
            E_u_diff = np.fmin(E_u_diff, np.finfo(np.float64).max)
            E_v_diff = v_i[k]*np.log(v_i[k])
            E_v_diff = np.fmax(E_v_diff, np.finfo(np.float64).eps)  # avoid underflow
            E_v_diff = np.fmin(E_v_diff, np.finfo(np.float64).max)
            try:
                arr = np.column_stack((arr,v_dif))
            except:
                arr = v_dif.copy()
            try:
                arr_v = np.vstack((arr_v,E_v_diff))
            except:
                arr_v= E_v_diff.copy()
            try:
                arr_u = np.column_stack((arr_u,E_u_diff))
            except:
                arr_u= E_u_diff.copy()    
                
            k+=1
        d1 = _U * v_dif
        ################## comp U
        # 
        E_u = self.theta_u * np.sum(arr_u,axis=0,keepdims=True).sum()
        ################## comp V
        # 
        E_v = self.theta_v * np.sum(arr_v,axis=1,keepdims=True).sum()

        return np.sum(d1,axis=0,keepdims=True).sum() + E_u + E_v

    @property
    def loss(self):
        #TODO  check fit
        return (np.array(self._loss))[-1]
        
    @property
    def loss_history(self):
        #TODO  check fit
        return np.array(self._loss)

    @property
    def labels_(self):

        if self.fitted:
            return np.argmax(self.U,axis=1)
        else:
            raise ValueError('NotFitted')

    @property
    def fuzzy_matrix(self):

        if self.fitted:
            return self.U
        else:
            raise ValueError('NotFitted')

    @property
    def fuzzy_weight_matrix(self):

        if self.fitted:
            return self.V
        else:
            raise ValueError('NotFitted')
    
    def __str__(self):
        return "EFCM_LS1"
            
            
            