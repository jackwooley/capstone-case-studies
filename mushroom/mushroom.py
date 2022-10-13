# UCI MUSHROOM DATASET
# K-NEAREST NEIGHBOR

import numpy as np
import pandas as pd

labels = ["edible", "cap-shape", "cap-surface", "cap-color", "bruises", "odor", "gill-attachment",
          "gill-spacing", "gill-size", "gill-color", "stalk-shape", "stalk-root", "stalk-surface-above-ring",
          "stalk-surface-below-ring", "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color",
          "ring-number", "ring-type", "spore-print-color", "population", "habitat"]
df = pd.read_csv("agaricus-lepiota.data", names=labels)
df.replace('?', np.NaN, inplace=True)

# EDA
# (8124, 23)
print(df.shape)
print(df.head(5))
# Response variable is "edible": either 'e' for "edible", or 'p' for "poisonous"
count = df["edible"].value_counts()
total = count["p"] + count["e"]
print((count["p"] / total) * 100, "% of the dataset is poisonous")
print((count["e"] / total) * 100, "% of the dataset is edible")
# 48% are poisonous
# 52% are edible
# Good split
na_vec = df.isnull().sum()
print(na_vec)
# All missing values (2480) are found in "stalk-root" column
# Exploring "stalk-root" column
count = df["stalk-root"].value_counts()
print(count)
# Drop?
df = df.drop(["stalk-root"], axis=1)
print(df.shape)
