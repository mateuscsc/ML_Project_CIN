import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import sys

def main():
    df = pd.read_table('segmentation.data', skiprows=3, index_col=False, sep=",")

    dataset1 = df.iloc[:,3:9].to_numpy()        # Shape
    dataset2 = df.iloc[:,9:19].to_numpy()       # RGB
    dataset3 = df.iloc[:,3:19].to_numpy()       # Shape + RGB

    best_dataset1 = { 'objective': sys.maxsize, 'kmeans': None, 'i': 0 }
    best_dataset2 = { 'objective': sys.maxsize, 'kmeans': None, 'i': 0 }
    best_dataset3 = { 'objective': sys.maxsize, 'kmeans': None, 'i': 0 }

    for i in range(50):
        km = KMeans(n_clusters=7, max_iter=150).fit(dataset1)

        if (km.inertia_ < best_dataset1['objective']):
            best_dataset1['objective'] = km.inertia_
            best_dataset1['kmeans'] = km
            best_dataset1['i'] = i

    print(best_dataset1['kmeans'].cluster_centers_.shape)

    '''
    for i in range(50):
        km = KMeans(n_clusters=7, max_iter=150).fit(dataset2)
        print("Labels: {} \nClusters: {}".format(km.labels_, km.cluster_centers_))

    for i in range(50):
        km = KMeans(n_clusters=7, max_iter=150).fit(dataset3)
        print("Labels: {} \nClusters: {}".format(km.labels_, km.cluster_centers_))
    '''

if __name__ == "__main__":
    main()