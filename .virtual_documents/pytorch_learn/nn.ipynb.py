import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        #定义神经网络结构，输入数据：1*32*32
        super(Net,self).__init__()
        #第一层：卷积层
        self.conv1=nn.Conv2d(1,6,3) #输入1层，输出6层，33的卷积核
        #第二层：卷积层
        self.conv2=nn.Conv2d(6,16,3)
        #第三层：全连接层
        self.fc1=nn.Linear(16*28*28,512) #16*28*28=12544
        #第四层：全连接层
        self.fc2=nn.Linear(512,64)
        #第五层：全连接层
        self.fc3=nn.Linear(64,2)
        pass
    
    def forward(self,x):
        #定义数据的流向
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


#生成随机数据
input_data=torch.rand(1,1,32,32)
print(input_data)
print(input_data.shape)



#运行神经网络
out=net(input_data)
print(out)
print(out.size())


#随机生成真实值
target=torch.rand(2)
target=target.view(1,-1)
print(target)
print(target.size())


criterion=nn.L1Loss() #定义损失函数
loss=criterion(out,target) #计算损失
print(loss)


net.zero_grad() #清零所有梯度
loss.backward() #自动计算梯度，反向传递


import torch.optim as optim
#优化权重
optimizer=optim.SGD(net.parameters(),lr=0.01)
optimizer.step()


#再次运行神经网络
out=net(input_data)
print(out)
print(out.size())



