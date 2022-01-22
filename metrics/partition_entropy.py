import numpy as np

"""
Baseado na página 55 da dissertação de Sara Inés Rizo Rodríguez

The PE index is a scalar measure of the amount of fuzziness in a given U.
The PE index values range in [0, log(C)], the closer the value of PE to 0,
the crisper the clustering is. The index value close to the upper bound 
indicates the absence of any clustering structure in the datasets or inability
of the algorithm to extract it.
"""

def PE(U):
  with np.errstate(divide='ignore'):
    log_U = np.log(U)

  log_U[np.isinf(log_U)] = 1
  n,_ = U.shape
  
  return -(1/n) * np.sum(np.multiply(U, log_U))

