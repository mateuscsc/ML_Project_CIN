import numpy as np
from copy import copy
from sklearn.preprocessing import Normalizer

def generate_centers(X_data=None,n_cluster=None,init='kmean'):

	assert type(X_data) != None, "X_data TypeError: 'NoneType' "
	assert type(n_cluster) != None, "n_cluster TypeError: 'NoneType' "


	if init == 'random':
		center_ = X_data[np.random.choice(np.arange(X_data.shape[0]), replace=False, size=n_cluster), :].copy()
		return center_
	if init == 'kmean':
		from sklearn.cluster import KMeans
		km = KMeans(n_clusters=n_cluster)
		km.fit(X_data)
		center_ = copy(km.cluster_centers_)
		return center_

def generate_matrix_U(n_samples=None,n_cluster=None,init='laplace'):

	assert type(n_samples) != None, "n_samples TypeError: 'NoneType' "
	assert type(n_cluster) != None, "n_cluster TypeError: 'NoneType' "

	if init == 'random':
		U_matrix = np.random.randint(n_samples, size=(n_samples, n_cluster))
		U_matrix = Normalizer(norm='l1').fit_transform(U_matrix)
		U_matrix = np.fmax(U_matrix, 0.000000001)
		U_matrix = np.fmin(U_matrix, 1.00)

	elif init == 'dirichlet':
		U_matrix = np.random.dirichlet(np.ones(n_cluster),size=n_samples)
		U_matrix = np.fmax(U_matrix, 0.000000001)
		U_matrix = np.fmin(U_matrix, 1.00)

	elif init == 'laplace':
		for _ in range(n_cluster):
			try:
				U_matrix = np.column_stack((U_matrix, abs(np.random.laplace(0, 1, size=(n_samples, 1))) ))
			except:
				U_matrix = abs(np.random.laplace(0, 1, size=(n_samples, 1)))
		U_matrix = Normalizer(norm='l1').fit_transform(U_matrix)
		U_matrix = np.fmax(U_matrix, 0.000000001)
		U_matrix = np.fmin(U_matrix, 1.00)
	else:
		raise ValueError('init method only supports [random, dirichlet,laplace]')

	return U_matrix