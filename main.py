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
load_dict = False
save_dict = True
save_acc_hist = True
dict_num = 19
batch_size=  60
max_epoch = 20
clip_coef = 0.25
num_layers = int(sys.argv[2])
class WordDataSet(Dataset):
    def __init__(self, tr_dir, is_train=True):
        self.tr_dir = tr_dir
        self.length = len(glob.glob(os.path.join(tr_dir, str(seq_length)+'_'+'*')))
        self.length = self.length- self.length % batch_size
        self.length = self.length-self.length%batch_size
        self.is_train = is_train

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data = np.load(os.path.join(self.tr_dir,str(seq_length)+'_'+str(idx)+'.npy')).item()
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
        self.out2score = nn.Linear(self.num_layers*self.output_dim, self.target_dim)

    def forward(self, input):
        _, lstm_out = self.lstm(input)
        #hidden_state
        lstm_out = lstm_out[0].squeeze()
        
        #print(lstm_out.size())
        if self.num_layers>1:
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
if load_dict:
    net.load_state_dict(torch.load(os.path.join(dict_save_path, 'bs-60_'+str(num_layers)+'_'+str(output_dim)+'_'+str(dict_num))))

loss_function = nn.NLLLoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)
#using Gpus
if torch.cuda.is_available():
    
    print("cuda is available...")
    if torch.cuda.device_count() > 1:
        print('using %d Gpus'%torch.cuda.device_count())
        net=nn.DataParallel(net, [0,1,2,3])
    
    net.cuda()
else:
    print('cuda disabled')


dataSet = WordDataSet('./train')
dataLoader = DataLoader(dataSet, batch_size = batch_size, shuffle=True, num_workers=0)

test_dataSet = WordDataSet('./test')
test_dataLoader = DataLoader(dataSet, batch_size = batch_size, shuffle=True, num_workers=0)
print('start training...')

accuracy_rec = np.array([])
loss_rec_train = np.array([])
loss_rec_test = np.array([])


for name, param in net.named_parameters():
    print(name, param.data.size())



for epoch in range(max_epoch):
    print('epoch {}'.format(epoch))
    for i_batch, sample in enumerate(dataLoader):
        input, label = sample['input'].type(torch.FloatTensor), sample['label'].type(torch.long)
        #input to cuda
        #input = input.permute(1, 0, 2)
        input = input.cuda()
        label = label.cuda()
        optimizer.zero_grad()
        outscores = net(input)
        #print(outscores.data.size())
        loss_nlloss = loss_function(outscores, label)
        loss_nlloss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), clip_coef)
       
        optimizer.step()
        loss_rec_train = np.append(loss_rec_train, loss_nlloss.item())
    
        if i_batch%100 == 0:
            ###print gradient###
            params = net.parameters()
            params = list(filter(lambda p: p.grad is not None, params))
            total_norm = 0
            for p in params:
                param_norm = p.grad.data.norm(2)
                total_norm = total_norm + param_norm.item()**2
            total_norm = total_norm**(1/2)
            print('current norm is {}'.format(total_norm))

            print('iter {}, training loss {}'.format(i_batch, loss_nlloss))
            total = 0
            correct = 0
            loss_test = 0
            with torch.no_grad():
                for test_batch, test_sample in enumerate(test_dataLoader):
                    test_input, test_label = test_sample['input'].type(torch.FloatTensor), test_sample['label'].type(torch.long)
                    test_input = test_input.cuda()
                    test_label = test_label.cuda()
                   
                    test_outscores = net(test_input)
                    #compute output loss
                    test_loss = loss_function(test_outscores, test_label)
                        
                    _, predicted = torch.max(test_outscores.data, 1)
                    # total number of samples
                    total += test_label.size(0)
                    correct += (predicted==test_label).sum().item()
                    #print('prediction:')
                    #print(predicted)
                    #print(test_label)
                    loss_test = loss_test + test_loss.item()
                    if test_batch > 50:
                        break
            acc = correct/total*100
            loss_rec_test = np.append(loss_rec_test, loss_test/total*batch_size)
            accuracy_rec = np.append(accuracy_rec, acc)
            print('testing accuracy is {} %%'.format(acc))
            net_str = 'bs-'+str(batch_size)+'_'+str(num_layers)+'_'+str(output_dim)
            if save_dict:
                torch.save(net.module.state_dict(), os.path.join(dict_save_path, net_str+'_'+str(epoch)))
            
            if save_acc_hist:
                np.save(os.path.join(accuracy_save_path, net_str+'_accuracy_rec'), accuracy_rec)
                np.save(os.path.join(accuracy_save_path, net_str+'_loss_rec_train'), loss_rec_train)
                np.save(os.path.join(accuracy_save_path, net_str+'_loss_rec_test'), loss_rec_test)
