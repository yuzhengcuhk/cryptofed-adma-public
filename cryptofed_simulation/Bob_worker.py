from typing import Any, Union, Optional
import pickle
import torch
import os
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
        "../data/minist_bob",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=batch_size,
    shuffle=True,
)


model_bob = TrainGan()
model_bob.traingan("../data/mnist_bob", "bob")   #训练各自的模型,public data of james
model_bob_gen = model_bob.generator

state = {'state':model_bob_gen.state_dict()}
if not os.path.isdir('Bob_Para'):
    os.mkdir('Bob_Para')
torch.save(state, './Bob_Para/bobweight.pth')

bob_worker = socket.socket()
bob_worker.bind(('localhost', 8080))
bob_worker.listen()
while True:
    conn, addr = bob_worker.accept()
    print('等待指令：')
    while True:
        data = conn.recv(1024)
        if not data:
            print('客户端断开')
            break

        #第一次接收的是命令，包括get和文件名，用filename接收文件名
        cmd, filename = data.decode().split()

        #从接收到的文件名判断是不是一个文件
        if os.path.isfile(filename):
            f = open(filename, 'rb')  #如果是，读模式打开这个文件
            m = hashlib.md5()         #生成md5对象
            file_size = os.stat(filename).st_size  #将文件大小赋值给file_size
            conn.send(str(file_size).encode())     #发送文件大小
            conn.recv(1024)           #接收确认信息

            for line in f:            #开始发文件
                m.update(line)        #发一行更新一下md5值
                conn.send(line)       #一行一行发送

            #print('file md5:', m.hexdigest())  打印整个文件的md5
            f.close()
            conn.send(m.hexdigest().encode())   #send md5
        print('send done')
    break
bob_worker.close()





