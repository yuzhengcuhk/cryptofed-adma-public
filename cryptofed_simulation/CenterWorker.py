import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import socket
import hashlib
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from GAN import TrainGan




img_size = 28               #MNIST数据集的缘故
batch_size = 64


dataloader_center = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../data/minist_center",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=batch_size,
    shuffle=True,
)


model_center = TrainGan()
model_center.traingan("../data/minist_center", "center")   #训练各自的模型,public data of james

center_worker = socket.socket()
host = '127.0.0.1'
port = 8080
center_worker.connect((host, port))

while True:
    cmd = input('what do u want?:').strip()  #获得命令和想要下载的文件
    if cmd.startswith('get'):               #如果字符串中有'get'开始执行下一步骤
        center_worker.send(cmd.encode())           #发送指令
        response = center_worker.recv(1024)        #接收文件大小
        print('file size', response.decode())
        center_worker.send(b'111')                 #发送确认信息
        file_size = int(response.decode())  #转整形，方便判断大小
        new_file_size = file_size           #赋值方便最后打印大小
        filename = 'new' + cmd.split()[1]           #将命令分割获得文件名
        f = open(filename, 'wb')            #写模式创建文件
        m = hashlib.md5()                   #生成md5对象
        while new_file_size > 0:
            data = center_worker.recv(1024)
            new_file_size -= len(data)      #收多少减多少
            m.update(data)                  #同步服务器的md5
            f.write(data)
        else:
            new_file_md5 = m.hexdigest()    #得到下载完的文件的md5
            print('file recv done', file_size)
            f.close()

        server_file_md5 = center_worker.recv(1024) #接收服务器端的md5
        print('server file md5:', server_file_md5)
        print('recv file md5:', new_file_md5)
    else:
        continue
center_worker.close()







