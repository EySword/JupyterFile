import torch
import torchvision as tv
import torchvision.transforms as ttf

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic("matplotlib", " inline")


#归一化：
transform=ttf.Compose(
    [ttf.ToTensor(),
     ttf.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))#三个图层的平均值，标准差
#      ttf.Flip()#数据增强(翻转)
    ]
    )

#训练数据集
trainset=tv.datasets.CIFAR10(root='F:\dataset',train=True,download=True,transform=transform)

#加载方法
trainloader=torch.utils.data.DataLoader(trainset,batch_size=4,shuffle=True,num_workers=2)

#测试数据集
testset=tv.datasets.CIFAR10(root='F:\dataset',train=False,download=True,transform=transform)

testloader=torch.utils.data.DataLoader(testset,batch_size=4,shuffle=False,num_workers=2)


def imshow(img):
    #输入数据：torch.tensor [c,h,w]
    img=img/2+0.5
    nping=img.numpy()
    nping=np.transpose(nping, (1,2,0)) #[h,w,c]
    plt.imshow(nping)
    pass


dataiter=iter(trainloader) #随机加载一个train batch


images, labels=dataiter.next()


images.size()


imshow(tv.utils.make_grid(images)) #这个函数的作用是把若干幅图拼成一个图像
print(labels)


class Net(nn.Module):
    def __init__(self):
        #定义神经网络结构，输入数据：1*32*32
        super(Net,self).__init__()
        #第一层：卷积层
        self.conv1=nn.Conv2d(3,6,3) #输入3层，输出6层，33的卷积核
        #第二层：卷积层
        self.conv2=nn.Conv2d(6,16,3)
        #第三层：全连接层
        self.fc1=nn.Linear(16*28*28,512) #16*28*28=12544
        #第四层：全连接层
        self.fc2=nn.Linear(512,64)
        #第五层：全连接层
        self.fc3=nn.Linear(64,10)
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


#损失函数和更新规则
criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(net.parameters(),lr=0.001,momentum=0.9)


for epoch in range(2):
    for i,data in enumerate(trainloader):
        images,labels=data
        
        outputs=net(images)
        
        loss=criteron(outputs,labels)
        
        #更新权重第一步是清零梯度
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if(iget_ipython().run_line_magic("1000==0):", "")
            print('Eopch:(), Step:(), Loss:()'.format(epoch,i,loss.item()))
            pass
        pass
    pass


#模型测试
correct=0.0
total=0.0
with torch.no_grad():
    for data in testloader:
        images,labels =data
        
        outputs=net(images)
        _,predicted=torch.max(outputs.data,1)
        
        correct += (predicted==labels).sum()
        total+= labels.size(0)
        pass
    pass
print('准确率:',float(correct)/total)


torch.save(net.state_dict(),'F:\torch_model\pic_classifier.pt')


net2=Net()
net2.load_state_dict(torch.load('F:\torch_model\pic_classifier.pt'))
