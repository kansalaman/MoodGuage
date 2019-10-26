from config import model_config
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def load_data(batched=True,test=False):
    bs=model_config['batch_size']
    if(test):
        df=pd.read_csv('audio_test.csv')
    else:
        df=pd.read_csv('audio_train.csv')
    
    data=(np.array(df[df.columns[3:]]),np.array(df[df.columns[2]]))
    if test or not batched:
        return [torch.FloatTensor(data[0]),torch.LongTensor(data[1])]
    
    data=list(zip(data[0],data[1]))
    n_iters=len(data)//bs
    batches=[]
    for i in range(1,n_iters+1):
        input_batch=[]
        output_batch=[]
        for e in data[bs*(i-1):bs*i]:
            input_batch.append(e[0])
            output_batch.append(e[1])
        batches.append([torch.FloatTensor(input_batch),torch.LongTensor(output_batch)])
    
    return batches

def evaluate(targets,predictions):
    performance={
        'acc':accuracy_score(targets,predictions),
        'f1':f1_score(targets,predictions,average='macro'),
        'precision':precision_score(targets,predictions,average='macro'),
        'recall': recall_score(targets,predictions,average='macro')}
    return performance

