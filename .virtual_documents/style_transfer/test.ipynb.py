import torch
import numpy as np
import torch.nn as nn


matrix=torch.rand(1,512,512,512)



matrix2=torch.randint(1,10,(5,5))
matrix2=matrix2.view(1,1,5,5).type(torch.FloatTensor)


matrix2.shape


matrix2


net=nn.Upsample(scale_factor=2,mode='nearest')


output2=net(matrix2)
output2.shape






