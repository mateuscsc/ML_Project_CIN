import numpy as np
import pandas as pd

def generate_u_matrix (n, m):
    matrix = np.random.rand(n, m)
    u = matrix/matrix.sum(axis=1)[:,None]

    return u

def efcm_ls1 (d, c, tu, tv, t=150, epsilon=10e-10):
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
        the maximum number of iterations (default value: 150)
    epsilon: float
        the stop condition parameter (default value: 1e-10)
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


def main():
    df = pd.read_table('segmentation.data', skiprows=2, index_col=False, sep=",")

    dataset1 = df.iloc[:,3:8]       # Shape
    dataset2 = df.iloc[:,9:18]      # RGB
    dataset3 = df.iloc[:,3:18]      # Shape + RGB

    results_dataset1 = []
    results_dataset2 = []
    results_dataset3 = []

    for i in range(150):
        results_dataset1.append(efcm_ls1(dataset1, 7, 0, 0, 150, 10e-10))

    for i in range(150):
        results_dataset2.append(efcm_ls1(dataset2, 7, 0, 0, 150, 10e-10))

    for i in range(150):
        results_dataset3.append(efcm_ls1(dataset3, 7, 0, 0, 150, 10e-10))

if __name__ == "__main__":
    main()