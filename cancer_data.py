import mglearn
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
import numpy as np
import pandas as pd

cancer = load_breast_cancer()
print("cancer.keys(): \n{}".format(cancer.keys()))

print("Data shape: {}".format(cancer.data.shape))


print("Number of samples per class:\n{}".format(
    {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}
))

# print("Traits:\n{}".format(cancer.feature_names))

from sklearn.datasets import fetch_california_housing

california = fetch_california_housing()


# data_url = "http://lib.stat.cmu.edu/datasets/boston"
# raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
# data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
# target = raw_df.values[1::2, 2]

print("Data shape: {}".format(california.data.shape))



