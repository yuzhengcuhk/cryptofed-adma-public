import numpy as np
import torch
from GAN import TrainGan
'''
如果使用同态加密，decrypt_sum.txt存储另外两位worker加和后的权重
如果不使用同态加密，ABsum.txt和bobgenpara.txt为存储两个worker的权重
center都不知道他们的具体参数，只有加和之后才知道  
所以我觉得好像没有必要用同态加密了。。。。
'''

model_trained_center = TrainGan()
model_center_trained_gen = model_trained_center.generator
model_center_trained_gen.load_state_dict(torch.load('./centergenweight.pth'))
'''from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in model_center_trained_gen.items():
    name = k[8:]
    new_state_dict[name] = v
model_center_trained_gen.load_state_dict(new_state_dict)'''
model_center_trained_gen.eval()

#用了同态加密，读取decrypt_sum.txt
'''
subprocess.call([os.getcwd()+'/sede'])
with open('decrypt_sum.txt', 'r') as Alicefile:
    for param_tensor in model_center_trained_gen.state_dict():
        alicedata = Alicefile.readlines()
        array_param = (model_center_trained_gen.state_dict()[param_tensor]).numpy()
        dim_array = len(array_param.shape)
        if dim_array == 0:
            pass
        if dim_array == 1:
            for aline in alicedata:
                aodom = aline.split()
                anum = map(float, aodom)
                alst = (list(anum))
                adarray = alst.numpy()
                a_tensor = torch.from_numpy(adarray)
                a_tensor = (a_tensor * 2)/3
                model_center_trained_gen.state_dict()[param_tensor] = model_center_trained_gen.state_dict()[param_tensor]/3
                model_center_trained_gen.state_dict()[param_tensor] = torch.add(
                    model_center_trained_gen.state_dict()[param_tensor], a_tensor)
                continue
        if dim_array == 2:
            for i, aline in range(array_param.shape[0]), alicedata:
                aodom = aline.split()
                anum = map(float, aodom)
                alst = (list(anum))
                adarray = alst.numpy()
                a_tensor = torch.from_numpy(adarray)
                a_tensor = (a_tensor* 2) / 3
                model_center_trained_gen.state_dict()[param_tensor][i] = model_center_trained_gen.state_dict()[param_tensor][i]/3
                model_center_trained_gen.state_dict()[param_tensor][i] = torch.add(
                    model_center_trained_gen.state_dict()[param_tensor][i], a_tensor)
Alicefile.close()
torch.save(model_center_trained_gen.state_dict(), './mergegendweight.pth')
'''

#没用同态加密
with open('ABsum.txt', 'r') as Alicefile:
    with open('bobgenpara.txt', 'r') as Bobfile:
        for param_tensor in model_center_trained_gen.state_dict():
            array_param = (model_center_trained_gen.state_dict()[param_tensor]).numpy()
            dim_array = len(array_param.shape)
            alicedata = Alicefile.readline()
            bobdata = Bobfile.readline()
            if dim_array == 0:
                pass
            if dim_array == 1:
                aodom = alicedata.split()
                bodom = bobdata.split()
                anum = map(float, aodom)
                bnum = map(float, bodom)
                alst = (list(anum))
                blst = (list(bnum))
                adarray = np.array(alst)
                bdarray = np.array(blst)
                a_tensor = torch.from_numpy(adarray)
                b_tensor = torch.from_numpy(bdarray)
                b_tensor = (torch.add(a_tensor, b_tensor) * 2)/3
                model_center_trained_gen.state_dict()[param_tensor] = model_center_trained_gen.state_dict()[param_tensor]/3
                model_center_trained_gen.state_dict()[param_tensor] = torch.add(
                        model_center_trained_gen.state_dict()[param_tensor], b_tensor)
                continue
            if dim_array == 2:
                #alice_tensor = torch.FloatTensor(array_param.shape[0], array_param.shape[1]) #生成一个空的tensor
                #bob_tensor = torch.FloatTensor(array_param.shape[0], array_param.shape[1])
                #思路,利用reshape将它转成2维的
                for i in range(0, array_param.shape[0]):
                    aodom = alicedata.split()
                    bodom = bobdata.split()
                    anum = map(float, aodom)
                    bnum = map(float, bodom)
                    alst = (list(anum))
                    blst = (list(bnum))
                    adarray = np.array(alst)
                    bdarray = np.array(blst)
                    a_tensor = torch.from_numpy(adarray)
                    b_tensor = torch.from_numpy(bdarray)
                    b_tensor = (torch.add(a_tensor, b_tensor) * 2) / 3
                    model_center_trained_gen.state_dict()[param_tensor][i] = \
                    model_center_trained_gen.state_dict()[param_tensor][i] / 3
                    model_center_trained_gen.state_dict()[param_tensor][i] = torch.add(
                        model_center_trained_gen.state_dict()[param_tensor][i], b_tensor)
                    if i != array_param.shape[0]-1:
                        alicedata = Alicefile.readline()
                        bobdata = Bobfile.readline()
Alicefile.close()
Bobfile.close()
torch.save(model_center_trained_gen.state_dict(), './mergegendweight.pth')
