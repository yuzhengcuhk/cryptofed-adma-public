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

from PreGAN import TrainGan


# bob_gan = TrainGan()
# bob_gen = bob_gan.generator
# bob_dis = bob_gan.discriminator
# bob_gan.traingan("./Bob/bob", "./Bob/images", "", 5)
# torch.save(bob_gen.state_dict(), './sgdBobGen.pth')
# torch.save(bob_dis.state_dict(), './sgdBobDis.pth')

alice_gan = TrainGan()
alice_gen = alice_gan.generator
alice_dis = alice_gan.discriminator
alice_gan.traingan("./Bob/bob", "./Alice/images", "", 6)

torch.save(alice_gen.state_dict(), './sgdAliceGen.pth')
torch.save(alice_dis.state_dict(), './sgdAliceDis.pth')

cs_gan = TrainGan()
cs_gen = cs_gan.generator
cs_dis = cs_gan.discriminator
cs_gan.gentrainmultidis("./Bob/bob", "./CenterWorker/images")
torch.save(cs_gen.state_dict(), './CSGen.pth')
torch.save(cs_dis.state_dict(), './CSDis.pth')