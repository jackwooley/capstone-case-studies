from matplotlib import pyplot as plt
import seaborn as sns
import car
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


def main():
    df = car.get_data()
    # # show data
    # figure, axes = plt.subplots(nrows=4, ncols=2, figsize =(15,20))
    # row = 0
    # col = 0
    # for feature in df.columns:
    #     df[feature].value_counts().plot(ax=axes[row, col], kind='bar', xlabel=feature, rot=0)
    #     col = col+1 if col < 1 else 0
    #     row = row + 1 if col == 0 else row
    # plt.tight_layout()
    # plt.savefig('feature_count.png')
    # plt.show()
    # show again based on correlation
    target = 'eval'

    df1 = df[df[target] == 'unacc']
    df2 = df[df[target] == 'acc']
    df3 = df[df[target] == 'good']
    df4 = df[df[target] == 'vgood']

    figure, axes = plt.subplots(nrows=len(df.columns)-1, ncols=4, figsize=(15,40))
    row = 0
    for feature in df.columns:
        if feature == target:
            continue
        df1[feature].value_counts().plot(ax=axes[row, 0], kind='bar', xlabel=feature+" unacc", rot=0)
        df2[feature].value_counts().plot(ax=axes[row, 1], kind='bar', xlabel=feature+" acc", rot=0)
        df3[feature].value_counts().plot(ax=axes[row, 2], kind='bar', xlabel=feature+" good", rot=0)
        df4[feature].value_counts().plot(ax=axes[row, 3], kind='bar', xlabel=feature+" v-good", rot=0)
        row += 1
    plt.tight_layout()
    plt.savefig('distribution.png')

    # one hot encoding
    # dum_data = car.data_to_dummy(df)
    # Target Encoding



    return



if __name__ == "__main__":
    main()