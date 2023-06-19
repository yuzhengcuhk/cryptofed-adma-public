import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import socket
import hashlib
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from GAN import TrainGan

img_size = 28               #MNIST数据集的缘故
batch_size = 64

dataloader_Alice = torch.utils.data.DataLoader(
    datasets.MNIST(
        "./alice",
        train=True,
        download=False,
        transform=transforms.Compose(
            [transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=batch_size,
    shuffle=True,
)

model_alice = TrainGan()
model_alice.traingan("./alice", "alice", 1)   #训练各自的模型,public data of james
model_alice_gen = model_alice.generator
model_alice_dis = model_alice.discriminator

#gen_state = {'state':model_alice_gen.state_dict()}
#dis_state = {'state':model_alice_dis.state_dict()}
#if not os.path.isdir('Alice_Para'):
    #os.mkdir('Alice_Para')
torch.save(model_alice_gen.state_dict(), './alicegenweight.pth')
torch.save(model_alice_dis.state_dict(), './alicedisweight.pth')

with open('alicegenpara.txt', 'w') as alicetxt:
    for param_tensor in model_alice_gen.state_dict():
        #print(model_client_gen.state_dict()[param_tensor].size())
        array_param = (model_alice_gen.state_dict()[param_tensor]).numpy()
        dim_array = len(array_param.shape)
        if dim_array == 0:
            alicetxt.write(str(0) + " ")
            alicetxt.write("\n")
        if dim_array == 1:
            for i in range(array_param.shape[0]):
                alicetxt.write(str(array_param[i]) + " ")
            alicetxt.write("\n")
        if dim_array == 2:
            for i in range(array_param.shape[0]):
                for j in range(array_param.shape[1]):
                    alicetxt.write(str(array_param[i][j]) + " ")
                alicetxt.write("\n")
alicetxt.close()
