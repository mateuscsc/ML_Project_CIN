import numpy as np

"""
Baseado na página 55 da dissertação de Sara Inés Rizo Rodríguez
"""

def MPC(U):
  n, C = U.shape
  vpc = (1/n) * np.sum(np.power(U, 2))

  return 1 - (C/C-1) * (1 - vpc)