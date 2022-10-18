import pandas as pd
from sklearn import preprocessing


def get_data():
    labels = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "eval"]
    df = pd.read_csv("car.data", names=labels)
    # print(df.head(5))
    # na_vec = df.isnull().sum()
    # print(na_vec)
    # No missing data
    return df


def to_ordinal(df):
    enc = preprocessing.OrdinalEncoder()
    enc.fit(df)
    df = enc.transform(df)
    return df


if __name__ == "__main__":
    data = get_data()
    data = to_ordinal(data)
