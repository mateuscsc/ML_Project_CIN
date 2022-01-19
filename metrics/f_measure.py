import numpy as np

"""
Baseado na página 59 da dissertação de Sara Inés Rizo Rodríguez
"""

def FMeasure(P, Q):
  N = len(P)
  unique, counts = np.unique(P, return_counts=True)
  b = dict(zip(unique, counts))

  return (1/N) * np.sum([b[i] * np.max([fm(P==i, Q==k) for k in b.keys()]) for i in b.keys()]) 

def fm(Pi, Pk):
  q = precision(Pi, Pk) + recall(Pi, Pk)
  
  if (q == 0):
    return 0

  return 2 * ((precision(Pi, Pk) * recall(Pi, Pk)) / q)

def precision(Pi, Pk):
  q = true_positives(Pi, Pk) + false_positives(Pi, Pk)
  
  if (q == 0):
    return 0

  return true_positives(Pi, Pk) / q

def recall(Pi, Pk):
  q = true_positives(Pi, Pk) + false_negatives(Pi, Pk)
  
  if (q == 0):
    return 0

  return true_positives(Pi, Pk) / q

def true_positives(Pi, Pk):
  return np.sum([x and y for x,y in zip(Pi, Pk)])

def false_positives(Pi, Pk):
  return np.sum([not x and y for x,y in zip(Pi, Pk)])
  
def false_negatives(Pi, Pk):
  return np.sum([x and not y for x,y in zip(Pi, Pk)])