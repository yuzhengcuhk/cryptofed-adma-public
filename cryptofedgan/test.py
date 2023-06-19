import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
print(torch.__version__)
print('gpu', torch.cuda.is_available())