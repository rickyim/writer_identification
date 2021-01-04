import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
import sys



class_10 = []
class_107 = []
interval = 20
batch_size = 7
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, target_dim, num_layers):
        super(LSTMClassifier, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.target_dim = target_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(self.input_dim, self.output_dim, self.num_layers, batch_first=True)
        self.out2score = nn.Linear(self.num_layers * self.output_dim, self.target_dim)

    def forward(self, input):
        _, lstm_out = self.lstm(input)
        #hidden_state
        lstm_out = lstm_out[0].squeeze()

        #print(lstm_out.size())
        if len(lstm_out.size())>=3:
            lstm_out = lstm_out.permute(1, 0, 2).contiguous()
        lstm_out = lstm_out.view(-1, self.output_dim*self.num_layers)
        score = self.out2score(lstm_out)
        
        try:
            outclass = F.log_softmax(score, dim = 1)
        except Exception:
            print('the shape of score is', score.shape)
        return outclass

def rnn_test(filedata, num_class):
    input_dim = 3
    if(num_class==107):
        seq_len = 20
        num_layers = 4
        output_dim = 400
        class_list = class_107
        params ='./bs-60_4_400_19'
    else:
        num_layers = 1
        seq_len = 20
        output_dim = 70
        class_list = class_10
        params = './70_32'
    #define a net
    net = LSTMClassifier(input_dim, output_dim, num_class, num_layers)
    #if load_dict:
    state_dict = torch.load(params)
    net.load_state_dict(state_dict)

    #using Gpus
    if torch.cuda.is_available():
        #print("cuda is available...")
        net=nn.DataParallel(net, [0,])    
        net.cuda()
    #else:
        #print('cuda disabled')
    #print('start testing')



    person_rec = np.array([])
    for character_id in range(len(filedata)):
        charac_rec = np.array([])
        #extract sequence from a character
        character = filedata[character_id]
        datapoints = np.zeros((1, 3))
        for stroke_id in range(len(character)):       
            stroke = character[stroke_id]
            stroke_len = len(stroke)
            for point_id in range(stroke_len):
                datapoints = np.append(datapoints, np.expand_dims(np.append(stroke[point_id], 1), axis = 0), axis = 0)
            datapoints[-1, -1] = 0

        x_min = np.min(datapoints[1:, 0])
        x_max = np.max(datapoints[1:, 0])
        y_min = np.min(datapoints[1:, 1])
        y_max = np.max(datapoints[1:, 1])
        if (x_max-x_min)>1e-2:
            datapoints[1:, 0] = (datapoints[1:, 0] - x_min)/(x_max - x_min)
        if (y_max-y_min)>1e-2:
            datapoints[1:, 1] = (datapoints[1:, 1] - y_min)/(y_max - y_min)

        points_seq = np.array([np.append((datapoints[i+1]-datapoints[i])[0:2], datapoints[i, 2]) for i in range(1, len(datapoints)-1)])
        points_len = len(points_seq)
        #print('point length = ', points_len)       

        while(points_len>0):
            # a sequence batch
            if(points_len <= seq_len):
                save_seq = np.zeros((seq_len,3))
                save_seq[-points_len:] = np.array(points_seq[-points_len:])
            else:
                save_seq = np.array(points_seq[-points_len:-points_len+seq_len])
            input_batch = np.expand_dims(save_seq, axis = 0)
            points_len -= seq_len
            #print('length=', points_len)

            while(points_len>0 and len(input_batch)<batch_size):
                if(points_len <= seq_len):
                    save_seq = np.zeros((seq_len,3))
                    save_seq[-points_len:] = np.array(points_seq[-points_len:])
                else:
                    save_seq = np.array(points_seq[-points_len:-points_len+seq_len])
                save_seq = np.expand_dims(save_seq, axis = 0)
                input_batch = np.concatenate((input_batch, save_seq), axis = 0)
                points_len -= seq_len
                #print('length=', points_len)

        
            with torch.no_grad():
                test_input = torch.as_tensor(input_batch).type(torch.FloatTensor).cuda()
                #print(test_input.size())
                outscores = net(test_input)
                if len(list(outscores.data.size()))>1:
                    _, predicted = torch.max(outscores.data, 1)
                else:
                    #print(outscores.data.cpu().numpy())
                    _, predicted = torch.max(outscores.data, 0)
                predicted = predicted.cpu().numpy()
                charac_rec = np.append(charac_rec, predicted)
                charac_rec = charac_rec.astype('int64')
        if len(charac_rec)<=0:
            continue
        pred = np.bincount(charac_rec).argmax()
        person_rec = np.append(person_rec, pred)
        #print(total, count)
    #print(person_rec)
    person_choice = np.bincount(person_rec.astype('int64')).argmax()
    #print(person_choice)
    return class_list[person_choice]
        
