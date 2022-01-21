import sys
import pandas as pd
import skfuzzy as fuzz
from sklearn.metrics import adjusted_rand_score
from metrics.f_measure import FMeasure
from metrics.partition_entropy import PE
from metrics.modified_partition_coefficient import MPC

def main():
    df = pd.read_table('segmentation.data', skiprows=3, index_col=False, sep=",")

    dataset1 = df.iloc[:,3:9].to_numpy()        # Shape
    dataset2 = df.iloc[:,9:19].to_numpy()       # RGB
    dataset3 = df.iloc[:,3:19].to_numpy()       # Shape + RGB

    best_dataset1 = { 'objective': sys.maxsize, 'crisp': None, 'fuzzy': None, 'mpc': None, 'pe': None, 'f-measure': None, 'rand': None }
    best_dataset2 = { 'objective': sys.maxsize, 'crisp': None, 'fuzzy': None, 'mpc': None, 'pe': None, 'f-measure': None, 'rand': None }
    best_dataset3 = { 'objective': sys.maxsize, 'crisp': None, 'fuzzy': None, 'mpc': None, 'pe': None, 'f-measure': None, 'rand': None }

    for i in range(50):
        _, u, _, _, jm, _, _ = fuzz.cmeans(dataset1.T, 7, 2, 0.005, 150)

        if (jm[-1] < best_dataset1['objective']):
            best_dataset1['fuzzy'] = u.T                    # Shape to U=(N, C)
            best_dataset1['crisp'] = u.T.argmax(axis=1)     # Get labels
            best_dataset1['objective'] = jm[-1]             # Get objective function value

    for i in range(50):
        _, u, _, _, jm, _, _ = fuzz.cmeans(dataset2.T, 7, 2, 0.005, 150)

        if (jm[-1] < best_dataset2['objective']):
            best_dataset2['fuzzy'] = u.T                    # Shape to U=(N, C)
            best_dataset2['crisp'] = u.T.argmax(axis=1)     # Get labels
            best_dataset2['objective'] = jm[-1]             # Get objective function value

    for i in range(50):
        _, u, _, _, jm, _, _ = fuzz.cmeans(dataset3.T, 7, 2, 0.005, 150)

        if (jm[-1] < best_dataset3['objective']):
            best_dataset3['fuzzy'] = u.T                    # Shape to U=(N, C)
            best_dataset3['crisp'] = u.T.argmax(axis=1)     # Get labels
            best_dataset3['objective'] = jm[-1]             # Get objective function value
    
    # Modified Partition Coefficient
    best_dataset1['mpc'] = MPC(best_dataset1['fuzzy'])
    best_dataset2['mpc'] = MPC(best_dataset2['fuzzy'])
    best_dataset3['mpc'] = MPC(best_dataset3['fuzzy'])

    # Partition Entropy
    best_dataset1['pe'] = PE(best_dataset1['fuzzy'])
    best_dataset2['pe'] = PE(best_dataset2['fuzzy'])
    best_dataset3['pe'] = PE(best_dataset3['fuzzy'])

    # Rand
    best_dataset1['rand'] = adjusted_rand_score(best_dataset1['crisp'], )
    best_dataset2['rand'] = adjusted_rand_score(best_dataset2['crisp'], )
    best_dataset3['rand'] = adjusted_rand_score(best_dataset3['crisp'], )

    # F-Measure
    best_dataset1['f-measure'] = FMeasure(best_dataset1['crisp'], )
    best_dataset2['f-measure'] = FMeasure(best_dataset2['crisp'], )
    best_dataset3['f-measure'] = FMeasure(best_dataset3['crisp'], )

    # Print results
    print("Dataset 1")
    print("MPC: {}\tPE: {}".format(best_dataset1['mpc'], best_dataset1['pe']))
    print("Dataset 2")
    print("MPC: {}\tPE: {}".format(best_dataset2['mpc'], best_dataset2['pe']))
    print("Dataset 3")
    print("MPC: {}\tPE: {}".format(best_dataset3['mpc'], best_dataset3['pe']))

if __name__ == "__main__":
    main()