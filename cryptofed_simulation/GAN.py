import argparse
import os
from abc import ABC
from typing import List, Any

import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
import random

from dp_sgd import DP_SGD

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]  # 输入样本大小和输出样本大小，对输入样本进行线性变换
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(  # 构建这个网络的模型,最后线性变换的时候把输出重新定位回图像的大小
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)  # 转换成1维的数据了
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(  # 先把图像转成512*512的
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity


class TrainGan(nn.Module):
    def __init__(self):
        super(TrainGan, self).__init__()
        self.generator = Generator()
        self.discriminator = Discriminator()

    def forward(self, img):
        pass

    def traingan(self, datasetpath, workerpath):
        # Loss function
        adversarial_loss = torch.nn.BCELoss()

        # Initialize generator and discriminator

        if cuda:
            self.generator.cuda()
            self.discriminator.cuda()
            adversarial_loss.cuda()

        # Configure data loader
        #print()
        dataloader = torch.utils.data.DataLoader(
            datasets.MNIST(
                datasetpath,
                train=True,
                download=False,
                transform=transforms.Compose(
                    [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
                ),
            ),
            batch_size=opt.batch_size,
            shuffle=True,
        )

        clip_value = 1.5
        noise = random.gauss(0, 6)

        # Optimizers
        #optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        optimizer_D = DP_SGD(self.generator.parameters(), lr=opt.lr)
        optimizer_G = DP_SGD(self.discriminator.parameters(), lr=opt.lr)

        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

        # ----------
        #  Training
        # ----------

        for epoch in range(opt.n_epochs):
            for i, (imgs, _) in enumerate(dataloader):

                # Adversarial ground truths
                valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
                fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

                # Configure input
                real_imgs = Variable(imgs.type(Tensor))

                # -----------------
                #  Train Generator
                # -----------------

                optimizer_G.zero_grad()

                # Sample noise as generator input
                z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

                # Generate a batch of images
                gen_imgs = self.generator(z)

                # Loss measures generator's ability to fool the discriminator
                g_loss = adversarial_loss(self.discriminator(gen_imgs), valid)

                g_loss.backward()
                # differential privacy                              暂时先这么写着？
                #torch.nn.utils.clip_grad_norm(generator.parameters(), 1e-4)

                optimizer_G.dpsgd_step(clip_value, noise)  # 如何在这里加上differential privacy呢？

                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_D.zero_grad()

                # Measure discriminator's ability to classify real from generated samples
                real_loss = adversarial_loss(self.discriminator(real_imgs), valid)
                fake_loss = adversarial_loss(self.discriminator(gen_imgs.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2

                d_loss.backward()
                optimizer_D.dpsgd_step(clip_value, noise)

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
                )

                batches_done = epoch * len(dataloader) + i
                if batches_done % opt.sample_interval == 0:
                    worker_image_path = workerpath + "/"
                    print(worker_image_path)
                    image_num = "%d.png" % batches_done
                    worker_image_path += image_num
                    print(worker_image_path)
                    save_image(gen_imgs.data[:25], worker_image_path, nrow=5, normalize=True)
