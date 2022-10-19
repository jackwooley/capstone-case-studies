from matplotlib import pyplot as plt
import seaborn as sns
import mushroom
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


def main():
    df = mushroom.get_data()
    # show data
    # figure, axes = plt.subplots(nrows=6, ncols=4, figsize =(15,20))
    # row = 0
    # col = 0
    # for feature in df.columns:
    #     df[feature].value_counts().plot(ax=axes[row, col], kind='bar', xlabel=feature, rot=0)
    #     col = col+1 if col < 3 else 0
    #     row = row + 1 if col == 0 else row
    # plt.tight_layout()
    # plt.savefig('count_chart.png')
    # # plt.show()
    # # show again based on correlation
    # target = 'edible'
    # y = df[target]
    # X = df.drop([target], axis=1)

    # df1 = df[df[target] == 'e']
    # df2 = df[df[target] != 'e']
    # figure, axes = plt.subplots(nrows=len(df.columns)-1, ncols=2, figsize=(15,40))
    # row = 0
    # for feature in df.columns:
    #     if feature == target:
    #         continue
    #     df1[feature].value_counts().plot(ax=axes[row, 0], kind='bar', xlabel=feature+" edible", rot=0)
    #     df2[feature].value_counts().plot(ax=axes[row, 1], kind='bar', xlabel=feature+" poison", rot=0)
    #     row += 1
    # plt.tight_layout()
    # plt.savefig('distribution.png')

    # one hot encoding
    dum_data = mushroom.data_to_dummy(df)
    # Target Encoding



    return



if __name__ == "__main__":
    main()
