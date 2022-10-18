# UCI MUSHROOM DATASET
# K-NEAREST NEIGHBOR

import numpy as np
import pandas as pd
import shutup

# shutup.please()


def eda():
    df = read_in_mush()
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

    # How to deal with nominal features
    # If {A, B, C} then A = (1, 0, 0), B = (0, 1, 0), C = (0, 0, 1)
    # Will drastically blow up dimensions
    dim = 0
    for x in df.columns:
        y = len(df[x].value_counts())
        print(x, y)
        dim += y
    print("ABOVE WILL ADD ", dim, " DIMENSIONS")


def blow_up(df):
    empty = [0] * df.shape[0]
    orig_columns = []
    for x in df.columns:
        counter = 0
        for y in range(len(df[x].value_counts())):
            df[x + str(counter)] = empty
            for z in df.iterrows():
                if z[1][x] == df[x].value_counts().index[y]:
                    df[x + str(counter)][z[0]] = 1
            counter += 1
        df = df.drop([x], axis=1)
    print(df.shape)


def data_to_dummy(df):
    for x in df.columns:
        df1 = pd.get_dummies(df[x])
        df = pd.concat([df, df1], axis=1).reindex(df.index)
        df.drop(x, axis=1, inplace=True)
    return df


def get_data():
    df = read_in_mush()
    df = df.drop(["stalk-root"], axis=1)

    return df


def read_in_mush():
    labels = ["edible", "cap-shape", "cap-surface", "cap-color", "bruises", "odor", "gill-attachment",
              "gill-spacing", "gill-size", "gill-color", "stalk-shape", "stalk-root", "stalk-surface-above-ring",
              "stalk-surface-below-ring", "stalk-color-above-ring", "stalk-color-below-ring", "veil-type",
              "veil-color",
              "ring-number", "ring-type", "spore-print-color", "population", "habitat"]
    df = pd.read_csv("agaricus-lepiota.data", names=labels)
    df.replace('?', np.NaN, inplace=True)
    return df


if __name__ == "__main__":
    eda()
    data = get_data()
    # blow_up(data)
    data_to_dummy(data)