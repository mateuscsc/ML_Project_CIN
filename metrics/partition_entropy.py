import numpy as np

"""
Baseado na página 55 da dissertação de Sara Inés Rizo Rodríguez
"""

def PE(U):
  with np.errstate(divide='ignore'):
    log_U = np.log(U)

  log_U[np.isinf(log_U)] = 1
  n,_ = U.shape
  
  return -(1/n) * np.sum(np.multiply(U, log_U))

