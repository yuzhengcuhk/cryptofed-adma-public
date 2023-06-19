import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import socket
import hashlib
import os
import subprocess
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from GAN import TrainGan

model_centerTGDU = TrainGan()
model_centerTGDU_gen = model_centerTGDU.generator
model_centerTGDU_dis = model_centerTGDU.discriminator

model_centerTGDU_gen.load_state_dict(torch.load('./mergegendweight.pth'))
model_centerTGDU_dis.load_state_dict(torch.load('./centerdisweight.pth'))

model_centerTGDU.traingan("./center", "center", "center.txt", 3)

torch.save(model_centerTGDU_gen.state_dict(), './newcentergen.pth')
torch.save(model_centerTGDU_dis.state_dict(), './centerdisweight.pth')