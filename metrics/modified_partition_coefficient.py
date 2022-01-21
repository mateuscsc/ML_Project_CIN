import numpy as np

"""
Baseado na página 55 da dissertação de Sara Inés Rizo Rodríguez

The range of MPC is the unit interval [0, 1], where MPC=0 corresponds to maximum 
fuzziness and MPC=1 to a hard partition.
"""

def MPC(U):
  n, C = U.shape
  vpc = (1/n) * np.sum(np.power(U, 2))

  return 1 - (C/C-1) * (1 - vpc)