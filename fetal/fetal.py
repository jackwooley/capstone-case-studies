import pandas as pd

df = pd.read_csv("fetal_health.csv")
print(df.head(5))
# (2126, 22)
print(df.shape)
na_vec = df.isnull().sum()
# NO MISSING VALUES
print(na_vec)
# MOSTLY NORMAL
print(df["fetal_health"].value_counts())
