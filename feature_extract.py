import numpy as np
import pickle
import pandas as pd
import librosa
from tqdm import tqdm
audio_vecs = pickle.load(open('file_vector_pickle.pkl', 'rb'))
file_label_df = pd.read_csv('filename_to_emotion.csv')
feature_dataframe_columns = ['file_name', 'label', 'sig_mean', 'sig_std',
                             'rmse_mean', 'rmse_std', 'silence', 'harmonic', 'auto_corr_max', 'auto_corr_std']
feature_dataframe = pd.DataFrame(columns=feature_dataframe_columns)

for file_name, vector in tqdm(audio_vecs.items()):
    features = [file_name,int(file_name.split('-')[2])]
    features.append(np.mean(abs(vector)))
    features.append(np.std(vector))
    rmse = librosa.feature.rms(vector+0.0001)[0]
    features.append(np.mean(rmse))
    features.append(np.std(rmse))
    silence = 0
    rmse_mean = np.mean(rmse)

    for val in rmse:
        if(val <= 0.4*rmse_mean):
            silence += 1
    silence = silence/float(len(rmse))
    features.append(silence)
    vector_harmonic = librosa.effects.hpss(vector)[0]
    features.append(np.mean(vector)*1000)

    clip_bound = 0.45*np.mean(abs(vector))
    center_clipped = []
    for val in vector:
        if val >= clip_bound:
            center_clipped.append(val-clip_bound)
        elif val <= -clip_bound:
            center_clipped.append(val+clip_bound)
        else:
            center_clipped.append(0)
    auto_corrs = librosa.core.autocorrelate(np.array(center_clipped))
    features.append(1000*np.max(auto_corrs)/len(auto_corrs))
    features.append(np.std(auto_corrs))
    feature_dataframe = feature_dataframe.append(pd.DataFrame(
        features, index=feature_dataframe_columns).transpose(), ignore_index=True)

feature_dataframe.to_csv('file_audio_features.csv')
