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

model_centerT = TrainGan()
model_centerT_A = model_centerT.generator
model_centerT_B = model_centerT.generator
model_centerT_gen = model_centerT.generator

model_centerT_A.load_state_dict(torch.load('./newAgendw.pth'))
model_centerT_B.load_state_dict(torch.load('./newBgendw.pth'))

for p_ten in model_centerT_gen.state_dict():
    tmp_tensor = torch.add(model_centerT_A.state_dict()[p_ten], model_centerT_B.state_dict()[p_ten])/2
    model_centerT_gen.state_dict()[p_ten] = tmp_tensor

torch.save(model_centerT_gen.state_dict(), './mergegendweight.pth')