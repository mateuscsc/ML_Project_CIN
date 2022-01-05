
def efcm_ls1 (d, c, tu, tv, t=100, epsilon=1e-5):
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
        the maximum number of iterations (default value: 1e-5)
    epsilon: float
        the stop condition parameter (default value: 1e-5)
    """

    #Initialization
    count = 0

    while max(abs()) < epsilon or count >= t:
        count += 1

        # Step 1: Representation

        # Step 2: Weighting

        # Step 3: Assignment
