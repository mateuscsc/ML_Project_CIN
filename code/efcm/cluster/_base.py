import numpy as np



class BaseCluster:
    def __init__(self, n_cluster, m):
        self.n_cluster = n_cluster
        self.m = m

    def _euclidean_distance(self, X, Y=None): # TODO cityblock distance
        """
        return the element-wise euclidean distance between X and Y
        :param X: [n_samples_X, n_features]
        :param Y: if None, return the element-wise distance between X and X, else [n_samples_Y, n_features]
        :return: [n_samples_X, n_samples_Y]
        """
        if Y is None:
            Y = X.copy()

        return np.linalg.norm(np.expand_dims(X, 2)-np.expand_dims(np.transpose(Y), 0), ord=2, axis=1, keepdims=False)

    def city_block_distance(self,X,Y=None):

        if Y is None:
            Y = X.copy()

        return np.linalg.norm(np.expand_dims(X, 2) - np.expand_dims(np.transpose(Y), 0), ord=1, axis=1, keepdims=False)

    def predict(self, X, y=None):
        """
        predict membership grad using fuzzy rules
        :param X: [n_samples, n_features]
        :param y: None
        :return: Mem [n_samples, n_clusters]
        """
        raise NotImplementedError('Function predict is not implemented yet.')

    def fit(self, X, y=None):
        raise NotImplementedError('Function fit is not implemented yet.')