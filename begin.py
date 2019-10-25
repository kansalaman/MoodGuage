#%%
import os
import re
import pandas as pd
import librosa
import pickle
from tqdm import tqdm

base_directory = "/home/aman/MoodGuage/Audio_Speech_Actors_01-24/"

audio_vec=dict()
file_name_list=[]
emotion_list=[]
df_ravdess=pd.DataFrame(columns=['file_name','emotion'])
sr=44100

for subdir in os.listdir(base_directory):
    if(subdir=='.gitignore'):
        continue
    current_directory=base_directory+subdir+'/'
    for file in tqdm(os.listdir(current_directory)):
        if(file=='.gitignore'):
            continue
        file_name,extension=file.split('.')
        file_name_list.append(file_name)
        emotion_list.append(int(file_name.split('-')[2]))
        # audio_vec[file_name]=librosa.load(current_directory+file,sr=sr)[0]

# fhand=open('file_vector_pickle.pkl','wb')
# pickle.dump(audio_vec,fhand)
# fhand.close()
df_ravdess['file_name']=file_name_list
df_ravdess['emotion']=emotion_list
df_ravdess.to_csv("filename_to_emotion.csv")




# %%
# IN THIS PART WE ARE EXTRACTING FEATURES FROM THE AUDIO
import numpy as np
audio_vecs=pickle.load(open('file_vector_pickle.pkl','rb'))
file_label_df=pd.read_csv('filename_to_emotion.csv')
feature_dataframe_columns=['file_name','label','sig_mean','sig_std','rmse_mean','rmse_std','silence','harmonic','auto_corr_max','auto_corr_std']
feature_dataframe=pd.DataFrame(columns=feature_dataframe_columns)

for file_name,vector in tqdm(audio_vecs.values()):
    features=[file_name,feature_dataframe[file_name]]
    features.append(np.mean(abs(vector)))
    features.append(np.std(vector))
    rmse=librosa.feature.rmse(vector+0.0001)[0]
    features.append(np.mean(rmse))
    features.append(np.std(rmse))
    silence=0
    rmse_mean=np.mean(rmse)

    for val in rmse:
        if(val<=0.4*rmse_mean):
            silence+=1
    silence=silence/float(len(rmse))
    features.append(silence)
    vector_harmonic=librosa.effects.hpss(vector)[0]
    features.append(np.mean(vector)*1000)

    clip_bound=0.45*np.mean(abs(vector))
    center_clipped=[]
    for val in vector:
        if val>=clip_bound:
            center_clipped.append(val-clip_bound)
        elif val<=-clip_bound:
            center_clipped.append(val+clip_bound)
        else:
            center_clipped.append(0)
    auto_corrs=librosa.core.autocorrelate(np.array(center_clipped))
    features.append(1000*np.max(auto_corrs)/len(auto_corrs))
    features.append(np.std(auto_corrs))
    feature_dataframe.append(np.array(features))

feature_dataframe.to_csv('file_audio_features.csv')



