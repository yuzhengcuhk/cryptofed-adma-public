from typing import Any, Union, Optional
import pickle
import torch
import os
import subprocess
import random
import hashlib
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


import socket
from GAN import TrainGan

img_size = 28               #MNIST数据集的缘故
batch_size = 64

dataloader_Bob = torch.utils.data.DataLoader(
    datasets.MNIST(
        "./bob",
        train=True,
        download=False,
        transform=transforms.Compose(
            [transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=batch_size,
    shuffle=False,
)


model_bob = TrainGan()
model_bob.traingan("./bob", "bob", 1)   #训练各自的模型,public data of james
model_bob_gen = model_bob.generator
model_bob_dis = model_bob.discriminator

#gen_state = {'state':model_bob_gen.state_dict()}
#dis_state = {'state':model_bob_dis.state_dict()}
#if not os.path.isdir('Bob_Para'):
    #os.mkdir('Bob_Para')
torch.save(model_bob_gen.state_dict(), './bobgenweight.pth')
torch.save(model_bob_dis.state_dict(), './bobdisweight.pth')

bobpara = 0
bobpara_rand = 0
#训练结束之后，保存generator的参数文件成txt格式
with open('bobgenpara.txt', 'w') as bobtxt:
    with open('bobgenpara_rand.txt', 'w') as bobrandtxt:
        for param_tensor in model_bob_gen.state_dict():
            #print(model_client_gen.state_dict()[param_tensor].size())
            array_param = (model_bob_gen.state_dict()[param_tensor]).numpy()
            dim_array = len(array_param.shape)
            if dim_array == 0:
                bobpara = random.random()
                bobpara_rand = -bobpara
                bobtxt.write(str(bobpara) + " ")
                bobrandtxt.write(str(bobpara_rand) + " ")
                bobtxt.write("\n")
                bobrandtxt.write("\n")
            if dim_array == 1:
                for i in range(array_param.shape[0]):
                    bobpara_rand = random.random()
                    bobpara = array_param[i] - bobpara_rand
                    bobtxt.write(str(bobpara) + " ")
                    bobrandtxt.write(str(bobpara_rand) + " ")
                bobtxt.write("\n")
                bobrandtxt.write("\n")
            if dim_array == 2:
                for i in range(array_param.shape[0]):
                    for j in range(array_param.shape[1]):
                        bobpara_rand = random.random()
                        bobpara = array_param[i][j] - bobpara_rand
                        bobtxt.write(str(bobpara) + " ")
                        bobrandtxt.write(str(bobpara_rand) + " ")
                    bobtxt.write("\n")
                    bobrandtxt.write("\n")
bobtxt.close()
bobrandtxt.close()

#subprocess.call([os.getcwd()+'/bob_puben'])
