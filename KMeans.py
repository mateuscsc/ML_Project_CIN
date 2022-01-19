import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

def main():
    df = pd.read_table('segmentation.data', skiprows=3, index_col=False, sep=",")

    dataset1 = df.iloc[:,3:9].to_numpy()        # Shape
    dataset2 = df.iloc[:,9:19].to_numpy()       # RGB
    dataset3 = df.iloc[:,3:19].to_numpy()       # Shape + RGB

    results_dataset1 = []
    results_dataset2 = []
    results_dataset3 = []

    for i in range(50):
        km = KMeans(n_clusters=7, max_iter=150).fit(dataset1)
        print("Labels: {} \nClusters: {}".format(km.labels_, km.cluster_centers_))
        print()

    for i in range(50):
        km = KMeans(n_clusters=7, max_iter=150).fit(dataset2)
        print("Labels: {} \nClusters: {}".format(km.labels_, km.cluster_centers_))

    for i in range(50):
        km = KMeans(n_clusters=7, max_iter=150).fit(dataset3)
        print("Labels: {} \nClusters: {}".format(km.labels_, km.cluster_centers_))

if __name__ == "__main__":
    main()