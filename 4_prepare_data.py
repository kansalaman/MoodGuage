import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from IPython.display import display
#from utils import *

#get_ipython().magic('matplotlib inline')

if __name__ == '__main__':

    feature_file = 'file_audio_features.csv'
    df = pd.read_csv(feature_file)
    df = df[df['label'].isin([1, 2, 3, 4, 5, 6, 7, 8])]

    df.to_csv('no_sample_df.csv')

    #df = up_sample(df,5)
    df.to_csv('modified_df.csv')

    scalar = MinMaxScaler()
    df[df.columns[2:]] = scalar.fit_transform(df[df.columns[2:]])

    x_train, x_test = train_test_split(df, test_size=0.30)
    x_train.to_csv('audio_train.csv', index=False)
    x_test.to_csv('audio_test.csv', index=False)