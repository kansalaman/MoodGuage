#%%
import os
import re
import pandas as pd
import librosa
import pickle
from tqdm import tqdm
from config import model_config

base_directory = "/home/aman/MoodGuage/Audio_Speech_Actors_01-24/"

audio_vec=dict()
file_name_list=[]
emotion_list=[]
df_ravdess=pd.DataFrame(columns=['file_name','emotion'])
sr=model_config['sr']

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




