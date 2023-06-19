#Bob Train Gan with Discriminator keep unchanged
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



'''
cs_gan = TrainGan()
cs_gen = cs_gan.generator
cs_dis = cs_gan.discriminator
cs_gan.traingan("./CenterWorker/fashion", "./CenterWorker/center",  "csloss.txt")
torch.save(cs_gen.state_dict(), './CSGen.pth')
torch.save(cs_dis.state_dict(), './CSDis.pth')'''


bob_gan = TrainGan()
bob_gen = bob_gan.generator
bob_dis = bob_gan.discriminator
bob_gan.trainprvgan("./Bob/fashion", "./Bob/bob", "bobloss.txt", 1)
torch.save(bob_gen.state_dict(), './BobGen.pth')
torch.save(bob_dis.state_dict(), './BobDis.pth')

alice_gan = TrainGan()
alice_gen = alice_gan.generator
alice_dis = alice_gan.discriminator
alice_gan.trainprvgan("./Alice/fashion", "./Alice/alice", "aliceloss.txt", 2)

torch.save(alice_gen.state_dict(), './AliceGen.pth')
torch.save(alice_dis.state_dict(), './AliceDis.pth')

GAN_stage3 = TrainGan()
GAN_stage3_gen = GAN_stage3.generator
GAN_stage3_dis = GAN_stage3.discriminator

GAN_stage3.gentrainmultidis("./CenterWorker/fashion", "./CenterWorker/images")

torch.save(GAN_stage3_gen.state_dict(), './NewCeneGen.pth')
torch.save(GAN_stage3_dis.state_dict(), './NewCeneDis.pth')