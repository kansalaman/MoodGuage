from torch.utils.data import dataset,dataloader
import pickle
import librosa
from config import model_config
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
import torch
import numpy as np
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

class soundDataset(dataset.Dataset):
    def __init__(self):
        # audio_vecs = pickle.load(open('file_vector_pickle.pkl', 'rb'))
        self.emotion_list=pickle.load(open('emos.pkl','rb'))
        self.audio_vec_list = pickle.load(open('audio_feats.pkl', 'rb'))
        # self.emotion_list=[]
        # self.audio_vec_list=[]
        # for file,audio_vec in tqdm(audio_vecs.items()):
        #     self.emotion_list.append(int(file.split('-')[2])-1)
        #     self.audio_vec_list.append(librosa.feature.mfcc(audio_vec,sr=model_config['sr']).T)
        #     #we can specify hoplength here if needed
        # fhand=open('audio_feats.pkl','wb')
        # pickle.dump(self.audio_vec_list,fhand)
        # fhand.close()
        # fhand=open('emos.pkl','wb')
        # pickle.dump(self.emotion_list, fhand)
        # fhand.close()
    
    def __getitem__(self, index):
        return torch.from_numpy(self.audio_vec_list[index]),self.emotion_list[index]
    
    def __len__(self):
        return len(self.audio_vec_list)

class LSTM_mfcc(nn.Module):
    def __init__(self):
        super(LSTM_mfcc,self).__init__()
        self.n_layers=2
        self.input_dim=20
        self.hidden_dim=40
        self.output_dim=8
        self.dropout=0.2
        self.rnn=nn.LSTM(self.input_dim,self.hidden_dim,bias=True,num_layers=2,dropout=self.dropout,bidirectional=True)
        self.out=nn.Linear(self.hidden_dim*2,self.output_dim)
        self.softmax=F.softmax
    
    def forward(self,input_seq):
        rnn_output,(hidden,_)=self.rnn(input_seq)
        #NO support for bidirectionals
        x = torch.cat((rnn_output[-1][:, :40], rnn_output[0][:, 40:]))
        x=self.out(x.view(1,self.hidden_dim*2))
        class_scores=F.softmax(x,dim=1)
        return class_scores


# class LSTM_mfcc(nn.Module):
#     def __init__(self):
#         super(LSTM_mfcc, self).__init__()
#         self.n_layers = 2
#         self.input_dim = 20
#         self.hidden_dim = 40
#         self.output_dim = 8
#         self.dropout = 0.2
#         self.rnn = nn.LSTM(self.input_dim, self.hidden_dim, bias=True,
#                            num_layers=2, dropout=self.dropout, bidirectional=False)
#         self.out = nn.Linear(self.hidden_dim, self.output_dim)
#         self.softmax = F.softmax

#     def forward(self, input_seq):
#         rnn_output, (hidden, _) = self.rnn(input_seq)
#         #NO support for bidirectionals
#         # print(hidden.shape)

#         class_scores = F.softmax(self.out(rnn_output[-1]), dim=1)
#         return class_scores


device = 'cpu'
model=LSTM_mfcc()
dset=soundDataset()
indices=np.array(list(range(len(dset))))
np.random.seed(42)
np.random.shuffle(indices)
dsplit=int(np.floor(len(dset)*0.8))
train_indices,test_indices=indices[:dsplit],indices[dsplit:]
train_sampler=SubsetRandomSampler(train_indices)
test_sampler=SubsetRandomSampler(test_indices)
train_loader=dataloader.DataLoader(dset,batch_size=1,sampler=train_sampler)
test_loader = dataloader.DataLoader(
    dset, batch_size=1, sampler=test_sampler)

num_epochs=10
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.00001)

for epoch in range(num_epochs):
    model.train()
    i=0
    for data,target in train_loader:
        print('train iteration',i)
        i+=1
        inputs=data[0].unsqueeze(1)
        targets=target
        inputs = inputs.to(device)
        targets = targets.to(device)
        model.zero_grad()
        optimizer.zero_grad()
        # print(inputs.shape)
        predictions=model(inputs)
        predictions=predictions.to(device)
        loss=criterion(predictions,targets)
        loss.backward()
        optimizer.step()

        if(i%10==0):
             with torch.no_grad():
                correct = 0
                j = 0
                for data, target in test_loader:
                    # print('test iteration', i)
                    j += 1
                    inputs = data[0].unsqueeze(1)
                    targets = target
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    predictions = model(inputs)
                    if(np.argmax(predictions, axis=1)[0].item() == targets[0].item()):
                        correct += 1
                print('Accuracy is {:.2f}'.format(correct/len(test_indices)))
    
    with torch.no_grad():
        correct=0
        i=0
        for data, target in test_loader:
            # print('test iteration',i)
            i+=1
            inputs = data[0].unsqueeze(1)
            targets = target
            inputs = inputs.to(device)
            targets = targets.to(device)
            predictions = model(inputs)
            if(np.argmax(predictions, axis=1)[0].item() == targets[0].item()):
                correct+=1
        print('Accuracy is {:.2f}'.format(correct/len(test_indices)))


