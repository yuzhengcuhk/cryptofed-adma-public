# import argparse
# import os
# from abc import ABC
# from typing import List, Any
#
# import numpy as np
# import math
#
# import torchvision.transforms as transforms
# from torchvision.utils import save_image
#
# from torch.utils.data import DataLoader
# from torchvision import datasets
# from torch.autograd import Variable
#
# import torch.nn as nn
# import torch.nn.functional as F
# import torch
# import random
#
# from dp_sgd import DP_SGD
# from dp_adam import DP_Adam
#
# os.makedirs("images", exist_ok=True)
#
# parser = argparse.ArgumentParser()
# parser.add_argument("--n_epochs", type=int, default=250, help="number of epochs of training")
# parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
# parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
# parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
# parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
# parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
# parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
# parser.add_argument("--img_size", type=int, default=224, help="size of each image dimension")#mnist img_size 28
# parser.add_argument("--channels", type=int, default=1, help="number of image channels")
# parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
# opt = parser.parse_args()
# print(opt)
#
# img_shape = (opt.channels, opt.img_size, opt.img_size)
# clip_value = 2
#
#
#
# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()
#
#         def block(in_feat, out_feat, normalize=True):
#             layers = [nn.Linear(in_feat, out_feat)]  # 输入样本大小和输出样本大小，对输入样本进行线性变换
#             if normalize:
#                 layers.append(nn.BatchNorm1d(out_feat, 0.8))
#             layers.append(nn.LeakyReLU(0.2, inplace=True))
#             return layers
#
#         self.model = nn.Sequential(  # 构建这个网络的模型,最后线性变换的时候把输出重新定位回图像的大小
#             *block(opt.latent_dim, 128, normalize=False),
#             *block(128, 256),
#             *block(256, 512),
#             *block(512, 1024),
#             nn.Linear(1024, int(np.prod(img_shape))),
#             nn.Tanh()
#         )
#
#     def forward(self, z):
#         img = self.model(z)
#         img = img.view(img.size(0), *img_shape)  # 转换成1维的数据了
#         return img
#
#
# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#
#         self.model = nn.Sequential(  # 先把图像转成512*512的
#             nn.Linear(int(np.prod(img_shape)), 512),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(512, 256),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(256, 1),
#             nn.Sigmoid(),
#         )
#
#     def forward(self, img):
#         img_flat = img.view(img.size(0), -1)
#         validity = self.model(img_flat)
#
#         return validity
#
#
# class TrainGan(nn.Module):
#     def __init__(self):
#         super(TrainGan, self).__init__()
#         self.generator = Generator()
#         self.discriminator = Discriminator()
#
#     def forward(self, img):
#         pass
#
#     def traingan(self, datasetpath, workerpath, lossfilepath):
#         # Loss function
#         adversarial_loss = torch.nn.BCELoss()
#
#         os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
#         # Initialize generator and discriminator
#         cuda = True if torch.cuda.is_available() else False
#
#         if cuda:
#             print("cuda!")
#             self.generator.cuda()
#             self.discriminator.cuda()
#             adversarial_loss.cuda()
#
#         # Configure data loader
#         dataloader = torch.utils.data.DataLoader(
#             datasets.FashionMNIST(
#             #datasets.MNIST(
#                 datasetpath,
#                 train=True,
#                 download=False,
#                 transform=transforms.Compose(
#                     [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
#                 ),
#             ),
#             batch_size=opt.batch_size,
#             shuffle=True,
#         )
#
#         # Optimizer
#         optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
#         optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
#         #optimizer_D = torch.optim.SGD(self.discriminator.parameters(), lr=opt.lr)
#         #optimizer_G = torch.optim.SGD(self.generator.parameters(), lr=opt.lr)
#
#         Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
#
#         # ----------
#         #  Training
#         # ----------
#         #print(len(dataloader))
#         lossfile = open(lossfilepath, 'w+')
#         for epoch in range(opt.n_epochs):
#             for i, (imgs, _) in enumerate(dataloader):     #mnist用于mnist和fashionmnist
#             #for idx, (data, target) in enumerate(dataloader):
#
#                 if i == 312:
#                     break
#                 record_len = 100
#
#                 # Adversarial ground truths
#                 valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
#                 fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)
#                 # Configure input
#                 real_imgs = Variable(imgs.type(Tensor))
#
#                 # -----------------
#                 #  Train Generator
#                 # -----------------
#
#                 optimizer_G.zero_grad()
#
#                 # Sample noise as generator input
#                 z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
#
#                 # Generate a batch of images
#                 gen_imgs = self.generator(z)
#                 d_resu = self.discriminator(gen_imgs)
#
#                 g_loss = adversarial_loss(d_resu, valid)
#                 g_loss.backward()
#
#                 optimizer_G.step()
#
#                 # ---------------------
#                 #  Train Discriminator
#                 # ---------------------
#
#                 optimizer_D.zero_grad()
#
#                 # Measure discriminator's ability to classify real from generated samples
#                 real_loss = adversarial_loss(self.discriminator(real_imgs), valid)
#                 fake_loss = adversarial_loss(self.discriminator(gen_imgs.detach()), fake)
#                 d_loss = (real_loss + fake_loss) / 2
#
#                 d_loss.backward()
#                 optimizer_D.step()
#
#
#                 datalen = 312
#                 if i % record_len == 0:
#                     print(
#                         "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
#                         % (epoch, opt.n_epochs, i, datalen, d_loss.item(), g_loss.item()), file=lossfile
#                     )
#                 print(
#                     "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
#                     % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
#                 )
#                 batches_done = epoch * datalen + i
#                 if batches_done % opt.sample_interval == 0:
#                     worker_image_path = workerpath + "/"
#                     image_num = "%d.png" % batches_done
#                     worker_image_path += image_num
#                     save_image(gen_imgs.data[:25], worker_image_path, nrow=5, normalize=True)
#         lossfile.close()
#
#     def trainprvgan(self, datasetpath, workerpath, lossfilepath, flag):
#         # Loss function
#         adversarial_loss = torch.nn.BCELoss()
#
#         os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
#         # Initialize generator and discriminator
#         cuda = True if torch.cuda.is_available() else False
#
#         if cuda:
#             self.generator.cuda()
#             self.discriminator.cuda()
#             adversarial_loss.cuda()
#
#         # Configure data loader
#         dataloader = torch.utils.data.DataLoader(
#             datasets.FashionMNIST(
#             #datasets.MNIST(
#                 datasetpath,
#                 train=True,
#                 download=False,
#                 transform=transforms.Compose(
#                     [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
#                 ),
#             ),
#             batch_size=opt.batch_size,
#             shuffle=True,
#         )
#
#         # Optimizer
#         optimizer_G = DP_Adam(self.generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
#         optimizer_D = DP_Adam(self.discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
#         #optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
#         #optimizer_D = DP_SGD(self.discriminator.parameters(), lr=opt.lr)
#         #optimizer_G = DP_SGD(self.generator.parameters(), lr=opt.lr)
#
#         Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
#
#         # ----------
#         #  Training
#         # ----------
#         lossfile = open(lossfilepath, 'w+')
#         for epoch in range(opt.n_epochs):
#             for i, (imgs, _) in enumerate(dataloader):
#
#                 if flag == 1:
#                     if i == 625:
#                         break
#                 if flag == 2:
#                     if (i >= 312 and i < 625):
#                         continue
#                 record_len = 100
#
#                 # Adversarial ground truths
#                 valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
#                 fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)
#
#                 # Configure input
#                 real_imgs = Variable(imgs.type(Tensor))
#
#                 # -----------------
#                 #  Train Generator
#                 # -----------------
#
#                 optimizer_G.zero_grad()
#
#                 # Sample noise as generator input
#                 z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
#
#                 # Generate a batch of images
#                 gen_imgs = self.generator(z)
#                 d_resu = self.discriminator(gen_imgs)
#
#                 g_loss = adversarial_loss(d_resu, valid)
#                 g_loss.backward()
#                 if flag == 1:
#                     if i == 624:
#                         nn.utils.clip_grad_norm_(self.generator.parameters(), clip_value)
#                 if flag == 2:
#                     if i == 937:
#                         nn.utils.clip_grad_norm_(self.generator.parameters(), clip_value)
#                 optimizer_G.dpadam_step(flag, i)
#
#                 # ---------------------
#                 #  Train Discriminator
#                 # ---------------------
#
#                 optimizer_D.zero_grad()
#
#                 # Measure discriminator's ability to classify real from generated samples
#                 real_loss = adversarial_loss(self.discriminator(real_imgs), valid)
#                 fake_loss = adversarial_loss(self.discriminator(gen_imgs.detach()), fake)
#                 d_loss = (real_loss + fake_loss) / 2
#
#                 d_loss.backward()
#                 if flag == 1:
#                     if i == 624:
#                         #pass
#                         nn.utils.clip_grad_norm_(self.discriminator.parameters(), clip_value)
#                 if flag == 2:
#                     if i == 937:
#                         nn.utils.clip_grad_norm_(self.discriminator.parameters(), clip_value)
#                 optimizer_D.dpadam_step(flag, i)
#
#                 datalen = 625
#                 if i % record_len == 0:
#                     print(
#                         "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
#                         % (epoch, opt.n_epochs, i, datalen, d_loss.item(), g_loss.item()), file=lossfile
#                     )
#                 print(
#                     "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
#                     % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
#                 )
#                 batches_done = epoch * datalen + i
#                 if batches_done % opt.sample_interval == 0:
#                     worker_image_path = workerpath + "/"
#                     image_num = "%d.png" % batches_done
#                     worker_image_path += image_num
#                     save_image(gen_imgs.data[:25], worker_image_path, nrow=5, normalize=True)
#         lossfile.close()
#
#     def gentrainmultidis(self, datasetpath, workerpath):
#         # Loss function
#         adversarial_loss = torch.nn.BCELoss()
#
#         os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
#         # Initialize generator and discriminator
#         cuda = True if torch.cuda.is_available() else False
#
#         bob_gen = Generator()
#         alice_gen = Generator()
#         bob_dis = Discriminator()
#         alice_dis = Discriminator()
#         bob_dis.load_state_dict(torch.load('./BobDis.pth'))
#         alice_dis.load_state_dict(torch.load('./AliceDis.pth'))
#         alice_gen.load_state_dict(torch.load('./AliceGen.pth'))
#         bob_gen.load_state_dict(torch.load('./BobGen.pth'))
#
#
#
#         self.generator.load_state_dict(torch.load('./BobGen.pth'))
#         self.discriminator.load_state_dict(torch.load('./BobDis.pth'))
#
#         for param_tensor in bob_gen.state_dict():
#             array_param = (bob_gen.state_dict()[param_tensor]).numpy()
#             dim_array = len(array_param.shape)
#             if dim_array == 0:
#                 pass
#             else:
#                 self.generator.state_dict()[param_tensor] = (torch.add(bob_gen.state_dict()[param_tensor], alice_gen.state_dict()[param_tensor])) / 2
#                 continue
#
#         if cuda:
#             self.generator.cuda()
#             self.discriminator.cuda()
#             bob_dis.cuda()
#             alice_dis.cuda()
#             adversarial_loss.cuda()
#
#         # Configure data loader
#         #print()
#         dataloader = torch.utils.data.DataLoader(
#             datasets.FashionMNIST(
#             #datasets.MNIST(
#                 datasetpath,
#                 train=True,
#                 download=False,
#                 transform=transforms.Compose(
#                     [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
#                 ),
#             ),
#             batch_size=opt.batch_size,
#             shuffle=True,
#         )
#
#         # Optimizers
#         optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
#         #optimizer_D = DP_Adam(self.discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
#         optimizer_BOb_D = DP_Adam(bob_dis.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
#         optimizer_Alice_D = DP_Adam(alice_dis.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
#         #optimizer_G = torch.optim.SGD(self.generator.parameters(), lr=opt.lr)
#         #optimizer_BOb_D = DP_SGD(bob_dis.parameters(), lr=opt.lr)
#         #optimizer_Alice_D = DP_SGD(alice_dis.parameters(), lr=opt.lr)
#
#         Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
#
#         # ----------
#         #  Training
#         # ----------
#         centerfile = open("csfile.txt", 'w+')
#         for epoch in range(opt.n_epochs):
#             for i, (imgs, _) in enumerate(dataloader):
#
#                 if i == 312:    #center
#                     break
#                 record_len = 100
#
#                 # Adversarial ground truths
#                 valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
#                 fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)
#
#                 # Configure input
#                 real_imgs = Variable(imgs.type(Tensor))
#
#                 # -----------------
#                 #  Train Generator
#                 # -----------------
#
#                 optimizer_G.zero_grad()
#
#                 # Sample noise as generator input
#                 z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
#                 #print(imgs.shape[0])
#
#                 # Generate a batch of images
#                 gen_imgs = self.generator(z)
#                 bob_out = bob_dis(gen_imgs)
#                 #print(bob_out.shape[1])
#                 alice_out = alice_dis(gen_imgs)
#                 d_d = torch.zeros(alice_out.shape[0], 1)
#                 for k in range(alice_out.shape[0]):
#                     if bob_out[k][0] > alice_out[k][0]:
#                         d_d[k][0] = alice_out[k][0]
#                     else:
#                         d_d[k][0] = bob_out[k][0]
#                 d_resu = d_d.cuda()
#                 d_out = Variable(torch.tensor(d_resu, dtype=torch.float32), requires_grad=True).cuda()
#                 # Loss measures generator's ability to fool the discriminator
#
#                 g_loss = adversarial_loss(d_out, valid)
#                 #print("here")
#                 g_loss.backward()
#                 #print("heree")
#                 optimizer_G.step()  # 如何在这里加上differential privacy呢？
#
#                 # ---------------------
#                 #  Train Discriminator
#                 # ---------------------
#
#                 #optimizer_D.zero_grad()
#                 optimizer_BOb_D.zero_grad()
#                 optimizer_Alice_D.zero_grad()
#
#                 # Measure discriminator's ability to classify real from generated samples
#                 Bob_real_loss = adversarial_loss(self.discriminator(real_imgs), valid)
#                 Bob_fake_loss = adversarial_loss(self.discriminator(gen_imgs.detach()), fake)
#                 Bob_d_loss = (Bob_real_loss + Bob_fake_loss) / 2
#                 Bob_d_loss.backward()
#                 #optimizer_D.dpadam_step()
#                 if i == 311:
#                     nn.utils.clip_grad_norm_(bob_dis.parameters(), clip_value)
#                 optimizer_BOb_D.dpadam_step(3, i)
#
#                 Alice_real_loss = adversarial_loss(alice_dis(real_imgs), valid)
#                 Alice_fake_loss = adversarial_loss(alice_dis(gen_imgs.detach()), fake)
#                 Alice_d_loss = (Alice_real_loss + Alice_fake_loss) / 2
#                 Alice_d_loss.backward()
#                 if i == 311:
#                     nn.utils.clip_grad_norm_(alice_dis.parameters(), clip_value)
#                 optimizer_Alice_D.dpadam_step(3, i)
#                 #batches_done = 0
#                 if i % record_len == 0:
#                     print(
#                         "[Epoch %d/%d] [Batch %d/%d] [BOB_D loss: %f] [Alice_D loss: %f] [G loss: %f]"
#                         % (epoch, opt.n_epochs, i, 312, Bob_d_loss.item(), Alice_d_loss.item(), g_loss.item()), file=centerfile
#                     )
#                 print(
#                     "[Epoch %d/%d] [Batch %d/%d] [BOB_D loss: %f] [Alice_D loss: %f] [G loss: %f]"
#                     % (epoch, opt.n_epochs, i, len(dataloader), Bob_d_loss.item(), Alice_d_loss.item(), g_loss.item())
#                 )
#                 batches_done = epoch * 312 + i
#                 if batches_done % opt.sample_interval == 0:
#                     worker_image_path = workerpath + "/"
#                     image_num = "%d.png" % batches_done
#                     worker_image_path += image_num
#                     save_image(gen_imgs.data[:25], worker_image_path, nrow=5, normalize=True)
#         centerfile.close()