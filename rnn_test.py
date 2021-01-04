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



class_10 = [2018310897, 2018310939, 2018310892, 2018310874, 2018211051, 2018310898, 2018211702, 2017310350, 2018211054, 2016310874]
class_107 = [2018310885, 2018211080, 2018211057, 2018210817, 2018270032, 2018310915, 2018210809, 2017210966, 2018211079, 2018270031, 2018211073, 2018310927, 2018211277, 2018310926, 2018310946, 2018211060, 2017213726, 2018310875, 2018310769, 2018310894, 2018310919, 2018310932, 2018310910, 2018310922, 2018311127, 2018310897, 2018280076, 2018211059, 2018211067, 2018310943, 2018310881, 2018310939, 2018211058, 2018211081, 2017310851, 2018310876, 2018310904, 2018211053, 2018310692, 2018211068, 2018310887, 2018211061, 2018312481, 2018214043, 2015011455, 2018211077, 2018211074, 2018312459, 2018310892, 2018211062, 2018310909, 2018310882, 2017213725, 2018310874, 2018310888, 2018214042, 2018311146, 2018310936, 2017312279, 2018211038, 2018211051, 2018211048, 2015011548, 2018211039, 2017211061, 2018310898, 2018310942, 2018211047, 2018310948, 2018211167, 2018312484, 2017312287, 2017310881, 2018310883, 2018211208, 2018310895, 2018310908, 2018211702, 2018312470, 2017310350, 2018214113, 2015011414, 2018310906, 2018310916, 2018211054, 2018310911, 2018214052, 2018210850, 2018310929, 2018211270, 2018211114, 2018310907, 2018211063, 2018310755, 2018310884, 2018280357, 2018310900, 2018310934, 2018312476, 2018210461, 2018310933, 2018211069, 2017310472, 2018211064, 2016310874, 2018310921, 2015011431]

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
        
