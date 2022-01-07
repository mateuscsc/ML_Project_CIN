import numpy as np

def generate_u_matrix (n, m):
    matrix = np.random.rand(n, m)
    u = matrix/matrix.sum(axis=1)[:,None]

    return u

def efcm_ls1 (d, c, tu, tv, t=100, epsilon=10e-10):
    """
    EFCM-LS1 Algorithm

    Input
    _____
    d : array
        the dataset array
    c : int
        the number of clusters
    tu : float
        the membership degree of objects
    tv : int
        the relevance weight of the variables
    t : int
        the maximum number of iterations (default value: 100)
    epsilon: float
        the stop condition parameter (default value: 1e-5)
    """

    # Initialization
    count = 0
    p = d.shape[0]
    u = generate_u_matrix(p,c)

    while max() < epsilon or count >= t:
        count += 1

        # Step 1: Representation

        # Step 2: Weighting

        # Step 3: Assignment

# For test only
efcm_ls1(np.random.randint(0, 255, (100,5)), 7, 0, 0)