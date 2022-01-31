

# Custom module
#
############################# 
import numpy as np

# singleton class 
class Efcmls1_Optimizer(object):
    #__OPT_PARAMS = ("X","U","G","V","n_cluster","T_u","T_v","tol_iter")
    def __init__(self,kwargs=None):

        self.X = None
        self.U = None
        self.G = None
        self.n_cluster = None
        self.T_u = None
        self.T_v = None
        self.tol_iter = 1
        self.__OPT_PARAMS = ("X","U","G","V","n_cluster","T_u","T_v","tol_iter")
        #####
        if kwargs != None:
            self.parameters = kwargs            
    # operate on class-level data
    # @classmethod
    # def getobfparams(cls):
    #     return cls.__OPT_PARAMS
        
    @property    
    def parameters(self):
    # todo dict
        return self.__dict__
    @parameters.setter
    def parameters(self, params):
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        # todo check conistency params. avoid update params (e.g _fr1)
        _keys = [1  for i in params.keys()if i not in self.__OPT_PARAMS ]
         
        if sum(_keys) > 0:
            raise ValueError(f"__OPT_PARAMS invalid. Check params keys values. They must be equal:{Efcmls1_Optimizer.getobfparams()}")
        
        return self.__dict__.update(**params)
    # singleton method    
    # def __new__(cls):
    #     if not hasattr(cls, 'instance'):
    #         cls.instance = super(Efcmls1_Optimizer, cls).__new__(cls)
    #     return cls.instance        
        
    def fit(self,params=None,**kwargs):
        # todo exceptions
        # todo dict version <**
        if kwargs:
            self.parameters = kwargs
            return self
        elif params:
            self.parameters = params
            return self
        else:
            raise ValueError("Efcmls1_Optimizer is None. Check params keys values")
        assert self.T_u > 0, "theta_u must be larger than 0.01"
        assert self.T_v > 0, "theta_v must be larger than 0"



    @property
    def update_v(self):
        #
        _X = self.X.copy()
        u = self.U.copy()
        _G = self.G.copy()

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
            num = np.exp(- num / self.T_v) 
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
        _X = self.X.copy()
        _G = self.G.copy()
        n_split_g = _G.shape[0]
        if self.n_cluster == None:
            self.n_cluster = n_split_g

        g_i = np.vsplit(_G, n_split_g)
        m = n_split_g
        U = []
        for g_ik in g_i:

            dist_num = np.linalg.norm(np.expand_dims(_X, 2)-np.transpose(g_ik), ord=1, axis=1, keepdims=False)
            dist_num = - dist_num / self.T_u
            dist_num = np.exp(dist_num)
            dist_num = np.fmax(dist_num, np.finfo(np.float64).eps)  # avoid underflow
            dist_num = np.fmin(dist_num, np.finfo(np.float64).max)

            dist_den = np.linalg.norm(np.expand_dims(_X, 2)-np.transpose(_G), ord=1, axis=1, keepdims=False)
            dist_den = - dist_den / self.T_u
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
    def update_g(self):
        #
        _X = self.X.copy()
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
    def overall_loss(self):
        #
        _X = self.X.copy()
        _G = self.G.copy()
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
        E_u = self.T_u * np.sum(arr_u,axis=0,keepdims=True).sum()
        ################## comp V
        # 
        E_v = self.T_v * np.sum(arr_v,axis=1,keepdims=True).sum()

        return np.sum(d1,axis=0,keepdims=True).sum() + E_u + E_v

    def eval(self):
        # todo getter 
        # todo @property version
        epsilon=1e-10
        _loss = []
        self.prev_U = None
        for i in range(self.tol_iter):

            self.G = self.update_g
            self.V = self.update_v
            if i > 0 and  np.linalg.norm(self.U - self.prev_U,ord=1) < epsilon:
                break
            else:
                self.prev_U = self.U.copy()
                _loss.append(self.overall_loss)
                self.U = self.update_u
                
        try:
            return _loss[-1]
        except:
            print('Invalid loss')

    def __str__(self):
        return "Efcmls1_Optimizer"
            
            