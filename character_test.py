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


dict_save_path = './params/'
accuracy_save_path = './acc'
seq_length = 40
input_dim = 3
output_dim = int(sys.argv[1])
target_dim = 107
load_dict = True
save_acc_hist = False
dict_num = int(sys.argv[2])
batch_size=  7
num_layers = 4
class WordDataSet(Dataset):
    def __init__(self, tr_dir, person, character, is_train=True):
        self.tr_dir = tr_dir
        self.is_train = is_train
        self.person = person
        self.character = character
        self.length = len(glob.glob(os.path.join(self.tr_dir, str(seq_length)+'_'+str(person)+'-'+str(character)+'-*.npy')))

    def __len__(self):
        #print(self.length)
        return self.length

    def __getitem__(self, idx):
        data = np.load(os.path.join(self.tr_dir, str(seq_length)+'_'+str(person)+'-'+str(character)+'-'+str(idx)+'.npy')).item()
        input = data['data'].astype(float)
        #print(input.shape)
        label = np.squeeze(data['gt'])
        if self.is_train:
        #random rotate
            rotrange = np.pi*0/180
            degree = 2*rotrange*np.random.rand()-rotrange
            for points in range(len(input)):
                x = input[points][0]
                y = input[points][1]
                #input_length = np.sqrt(x**2 + y**2)
                input[points][0] = (x*np.cos(degree)-y*np.sin(degree))
                input[points][1] = (x*np.sin(degree)+y*np.cos(degree))
        return {'input': torch.as_tensor(input), 'label':torch.as_tensor(label, dtype=torch.long)}

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

#define a net
net = LSTMClassifier(input_dim, output_dim, target_dim, num_layers)
loss_function = nn.NLLLoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)
#if load_dict:
state_dict = torch.load(os.path.join(dict_save_path, 'bs-60_'+str(num_layers)+'_'+str(output_dim)+'_'+str(dict_num)))
for key, value in state_dict.items():
    print(key)
net.load_state_dict(state_dict)

#using Gpus
if torch.cuda.is_available():
    
    print("cuda is available...")
    #if torch.cuda.device_count() > 1:
        #print('using %d Gpus'%torch.cuda.device_count([0,]))
    net=nn.DataParallel(net, [0,])
    
    net.cuda()
else:
    print('cuda disabled')


print('start testing')

person_right = 0
person_total = 0
character_right = 0
character_total = 0
acc_rec = np.array([])
for person in range( 107):
    total = 0
    count = 0
    person_rec = np.array([])
    for character in range(100):
        charac_rec = np.array([])
        with torch.no_grad():
            test_dataSet = WordDataSet('./character_test', person, character, is_train=False)
            if test_dataSet.__len__() <= 0:
                continue
            test_dataLoader = DataLoader(test_dataSet, batch_size = batch_size, shuffle=True, num_workers=0)
            for i_batch, sample in enumerate(test_dataLoader):
                test_input, test_label = sample['input'].type(torch.FloatTensor), sample['label'].type(torch.long)
                test_input = test_input.cuda()
                outscores = net(test_input)
                if len(list(outscores.data.size()))>1:
                    _, predicted = torch.max(outscores.data, 1)
                else:
                    #print(outscores.data.cpu().numpy())
                    _, predicted = torch.max(outscores.data, 0)
                predicted = predicted.cpu().numpy()
                charac_rec = np.append(charac_rec, predicted)
                charac_rec = charac_rec.astype('int64')
        pred = np.bincount(charac_rec).argmax()
        person_rec = np.append(person_rec, pred)
        #print(pred)
        if pred == person:
            count= count + 1
            character_right = character_right + 1
        total = total + 1
        character_total = character_total + 1
        #print(total, count)
    #print(person_rec)
    person_choice = np.bincount(person_rec.astype('int64')).argmax()
    print('character accuracy = {}'.format(count/total))
    acc_rec = np.append(acc_rec, count/total)
    print('person choice = {}'.format(person_choice))
    if person_choice == person:
        person_right = person_right + 1
    person_total = person_total  +1
print('total accuracy: ', person_right/person_total)
print('total character accuracy: ', character_right/character_total)
np.save('test_acc.npy', acc_rec)

    
        
