import numpy as np
import pandas as pd
from fcmeans import FCM
from matplotlib import pyplot as plt

def main():
    df = pd.read_table('segmentation.data', skiprows=3, index_col=False, sep=",")

    dataset1 = df.iloc[:,3:9].to_numpy()        # Shape
    dataset2 = df.iloc[:,9:19].to_numpy()       # RGB
    dataset3 = df.iloc[:,3:19].to_numpy()       # Shape + RGB

    results_dataset1 = []
    results_dataset2 = []
    results_dataset3 = []

    for i in range(150):
        fcm = FCM(n_clusters=7)
        fcm.fit(dataset1)
        fcm_centers = fcm.centers

    for i in range(150):
        fcm = FCM(n_clusters=7)
        fcm.fit(dataset2)

    for i in range(150):
        fcm = FCM(n_clusters=7)
        fcm.fit(dataset3)

if __name__ == "__main__":
    main()