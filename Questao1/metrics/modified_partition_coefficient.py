import numpy as np

"""
Baseado na página 55 da dissertação de Sara Inés Rizo Rodríguez

The range of MPC is the unit interval [0, 1], where MPC=0 corresponds to maximum 
fuzziness and MPC=1 to a hard partition.
"""

def MPC(U):
  n, K = U.shape
  return (K/(K-1)) * (np.sum(np.power(U, 2)) / n) - (1/(K-1))