#Alice Train Gan with Discriminator keep unchanged
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

model_ATGDU = TrainGan()
model_ATGDU_gen = model_ATGDU.generator
model_ATGDU_dis = model_ATGDU.discriminator

model_ATGDU_gen.load_state_dict(torch.load('./mergegendweight.pth'))
model_ATGDU_dis.load_state_dict(torch.load('./alicedisweight.pth'))

model_ATGDU.traingan("./alice", "alice", "alice2.txt", 0)

torch.save(model_ATGDU_gen.state_dict(), './newAgendw.pth')