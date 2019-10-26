import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from IPython.display import display
from utils import *

#get_ipython().magic('matplotlib inline')

if __name__ == '__main__':

    feature_file = 'file_audio_features.csv'
    df = pd.read_csv(feature_file)
    # df = df[df['label'].isin([3, 4, 5, 8])]
    # df['label'] = df['label'].map({3: 0, 4: 1, 5: 2, 8: 3})

    df = df[df['label'].isin([1, 2, 3, 4, 5, 6, 7, 8])]
    df['label'] = df['label'].map({1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7})

    df.to_csv('no_sample_df.csv', index=False)

    df = up_sample(df,0)
    df = up_sample(df,1)
    df = up_sample(df,2)
    df = up_sample(df,3)
    df = up_sample(df,4)
    df = up_sample(df,5)
    df = up_sample(df,6)
    df = up_sample(df,7)
    df.to_csv('modified_df.csv', index=False)

    scalar = MinMaxScaler()
    df[df.columns[3:]] = scalar.fit_transform(df[df.columns[3:]])

    x_train, x_test = train_test_split(df, test_size=0.20)
    x_train.to_csv('audio_train.csv', index=False)
    x_test.to_csv('audio_test.csv', index=False)