import torch
import subprocess
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from GAN import TrainGan

GAN_stage3 = TrainGan()
GAN_stage3_gen = GAN_stage3.generator
GAN_stage3_dis = GAN_stage3.discriminator

GAN_stage3.gentrainmultidis("./CenterWorker/fashion", "./CenterWorker/center")

torch.save(GAN_stage3_gen.state_dict(), './NewCeneGen.pth')
torch.save(GAN_stage3_dis.state_dict(), './NewCeneDis.pth')
'''
bob_gan = TrainGan()
bob_gen = bob_gan.generator
bob_dis = bob_gan.discriminator
bob_gan.trainprvgan("./Bob/bob", "./Bob/bob", "bobloss.txt", 3)
torch.save(bob_gen.state_dict(), './BobGen.pth')
torch.save(bob_dis.state_dict(), './BobDis.pth')

alice_gan = TrainGan()
alice_gen = alice_gan.generator
alice_dis = alice_gan.discriminator
alice_gan.trainprvgan("./Alice/alice", "./Alice/alice", "aliceloss.txt", 4)

torch.save(alice_gen.state_dict(), './AliceGen.pth')
torch.save(alice_dis.state_dict(), './AliceDis.pth')'''