import sigpy.plot as pl
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import scipy.io
from torch.autograd import Variable
# % matplotlib notebook

import UFNet

def read_flipangles(flip_file):
    f = open(flip_file)
    flips =  [float(a) for a in f.readlines()]
    f.close()
    return np.array(flips)
flips = read_flipangles('/mikRAID/jtamir/projects/MRF_direct_contrast_synthesis/data/DictionaryAndSequenceInfo/flipangles.txt')
N_flip = len(flips)

mrf_dict = scipy.io.loadmat('/mikRAID/jtamir/projects/MRF_direct_contrast_synthesis/data/DictionaryAndSequenceInfo/fp_dictionary.mat')
# print(MRF_dic.keys())
fp_dict = mrf_dict['fp_dict']
t1_list = mrf_dict['t1_list']
t2_list = mrf_dict['t2_list']
N_dict = t1_list.shape[0]

fp_dic = np.hstack(list(fp_dict[0][0])).reshape((N_flip, 2, N_dict)).transpose((0, 2, 1))
fp_dic = np.abs(fp_dic[:,:,0] + 1j * fp_dic[:,:,1])
fp_train = fp_dic.transpose(1,0)[:,None,:]
# fp_dic = fp_dic[:,:,0] + 1j * fp_dic[:,:,1]
# print(fp_dic.shape)

# fp_train = fp_dic.transpose((1,2,0))

p = np.random.rand(22031,256)
p_norm = np.linalg.norm(p,axis=1)
p_normal = p/p_norm[None,:].T
B_tensor = torch.tensor(p_normal)

# Here we train the network
n_dictionary = 22031
n_dimension = 256
tau = 0.07
def train_net(net,
              epochs=5,
              batch_size=1,
              lr=0.001):
    optimizer = torch.optim.SGD(net.parameters(), lr,
                                momentum=0.9,
                                weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    loss_all = list([])
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        print("Starting Epoch: %d" %(epoch+1))
        for index in range(n_dictionary):
            input_dic = fp_train_cuda[index,:,:].unsqueeze(0)
            output_dic = net(input_dic)
            
            
            ## error: Trying to backward through the graph a second time, but the buffers have already been freed. Specify retain_graph=True when calling backward the first time.
#             print(output_dic)
#             B_tensor_cuda.requires_grad = False
            output_dic1 = torch.mm(output_dic,B_tensor_cuda.detach().float().transpose(0,1))/tau
#             print(output_dic.requires_grad)
#             print(output_dic.shape)
            loss = criterion(output_dic1,torch.LongTensor([index]).cuda())
            if index % 1000 == 0:
                print(loss.item())
                loss_all.append(loss.item())
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            B_tensor_cuda[index,:] = output_dic
        print('Epoch finished ! Loss: {}'.format(epoch_loss / 22031))
        torch.save(net.state_dict(),
                           '/home/kewang/checkpoints_unf_3/' + 'CP{}.pth'.format(epoch + 1))
        np.save("loss_all.npy",np.array(loss_all))
        np.save("Bank.npy",B_tensor_cuda.detach().cpu().numpy())
        print('Checkpoint {} saved !'.format(epoch + 1))
        

        
net_uf = UFNet.PixelNet(1)
net_ufcuda = net_uf.cuda()
fp_train_cuda = torch.tensor(fp_train).cuda()
B_tensor_cuda = B_tensor.cuda()
train_net(net_ufcuda,1000,1,0.02)