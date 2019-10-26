import torch
import torch.nn as nn
from config import model_config
import torch.nn.functional as F
from torch import optim
from lstm_utils import load_data, evaluate
import numpy as np
import pickle

class LSTMClassifier(nn.Module):

    def __init__(self):
        super(LSTMClassifier,self).__init__()
        #HYPERPARAMETER
        self.n_layers=model_config['n_layers']
        #INPUT DIMENSION OBTAINED BY MFCC FEATURES
        self.input_dim=model_config['input_dim']
        self.hidden_dim=model_config['hidden_dim']
        self.output_dim=model_config['output_dim']
        self.bidirectional=model_config['bidirectional']
        self.dropout=model_config['dropout'] if self.n_layers > 1 else 0

        self.rnn=nn.LSTM(self.input_dim, self.hidden_dim, bias=True,
        num_layers=self.n_layers,dropout=self.dropout, bidirectional=self.bidirectional)

        self.out=nn.Linear(self.hidden_dim,self.output_dim)
        self.softmax=F.softmax

    def forward(self, input_seq):
        rnn_output, (hidden,_) = self.rnn(input_seq)
        if(self.bidirectional):
            rnn_output=rnn_output[:,:,:self.hidden_dim]+\
                rnn_output[:,:,self.hidden_dim:]
        class_scores=F.softmax(self.out(rnn_output[0]),dim=1)
        return class_scores
    

if __name__=='__main__':
    emotion_dict={}

    device='cuda:{}'.format(model_config['gpu']) if torch.cuda.is_available() else 'cpu'


    model=LSTMClassifier()
    model=model.to(device)
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(),lr=model_config['learning_rate'])

    train_batches=load_data()
    test_pairs=load_data(test=True)

    best_acc=0
    for epoch in range(model_config['n_epochs']):
        losses=[]
        for batch in train_batches:
            inputs=batch[0].unsqueeze(0)
            targets=batch[1]
            inputs=inputs.to(device)
            targets=targets.to(device)
            model.zero_grad()
            optimizer.zero_grad()
            predictions=model(inputs)
            predictions=predictions.to(device)
            
            loss=criterion(predictions,targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        with torch.no_grad():
            inputs=test_pairs[0].unsqueeze(0)
            targets=test_pairs[1]
            inputs=inputs.to(device)
            targets=targets.to(device)
            predictions=torch.argmax(model(inputs),dim=1)
            predictions=predictions.to(device)
            targets=np.array(targets.cpu())
            predictions=np.array(predictions.cpu())
            performance=evaluate(targets,predictions)

            if(performance['acc']>best_acc):
                best_acc=performance['acc']
                print(performance)
                torch.save({
                    'model':model.state_dict(),
                    'optimizer':optimizer.state_dict()
                },'my_best_model.pth')

                with open('my_best_performance.pkl'.format(model_config['model_code']),'wb') as f:
                    pickle.dump(performance,f)




