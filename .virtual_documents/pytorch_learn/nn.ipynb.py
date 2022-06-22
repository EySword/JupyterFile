import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        #����������ṹ���������ݣ�1*32*32
        super(Net,self).__init__()
        #��һ�㣺�����
        self.conv1=nn.Conv2d(1,6,3) #����1�㣬���6�㣬33�ľ����
        #�ڶ��㣺�����
        self.conv2=nn.Conv2d(6,16,3)
        #�����㣺ȫ���Ӳ�
        self.fc1=nn.Linear(16*28*28,512) #16*28*28=12544
        #���Ĳ㣺ȫ���Ӳ�
        self.fc2=nn.Linear(512,64)
        #����㣺ȫ���Ӳ�
        self.fc3=nn.Linear(64,2)
        pass
    
    def forward(self,x):
        #�������ݵ�����
        x=self.conv1(x)
        x=F.relu(x)
        
        x=self.conv2(x)
        x=F.relu(x)
        
        x=x.view(-1,16*28*28)
        x=self.fc1(x)
        x=F.relu(x)
        
        x=self.fc2(x)
        x=F.relu(x)
        
        x=self.fc3(x)
        
        return x
    pass


net=Net()
print(net)


#�����������
input_data=torch.rand(1,1,32,32)
print(input_data)
print(input_data.shape)



#����������
out=net(input_data)
print(out)
print(out.size())


#���������ʵֵ
target=torch.rand(2)
target=target.view(1,-1)
print(target)
print(target.size())


criterion=nn.L1Loss() #������ʧ����
loss=criterion(out,target) #������ʧ
print(loss)


net.zero_grad() #���������ݶ�
loss.backward() #�Զ������ݶȣ����򴫵�


import torch.optim as optim
#�Ż�Ȩ��
optimizer=optim.SGD(net.parameters(),lr=0.01)
optimizer.step()


#�ٴ�����������
out=net(input_data)
print(out)
print(out.size())



