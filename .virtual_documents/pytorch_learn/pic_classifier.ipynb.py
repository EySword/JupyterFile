import torch
import torchvision as tv
import torchvision.transforms as ttf

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic("matplotlib", " inline")


#��һ����
transform=ttf.Compose(
    [ttf.ToTensor(),
     ttf.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))#����ͼ���ƽ��ֵ����׼��
#      ttf.Flip()#������ǿ(��ת)
    ]
    )

#ѵ�����ݼ�
trainset=tv.datasets.CIFAR10(root='F:\dataset',train=True,download=True,transform=transform)

#���ط���
trainloader=torch.utils.data.DataLoader(trainset,batch_size=4,shuffle=True,num_workers=2)

#�������ݼ�
testset=tv.datasets.CIFAR10(root='F:\dataset',train=False,download=True,transform=transform)

testloader=torch.utils.data.DataLoader(testset,batch_size=4,shuffle=False,num_workers=2)


def imshow(img):
    #�������ݣ�torch.tensor [c,h,w]
    img=img/2+0.5
    nping=img.numpy()
    nping=np.transpose(nping, (1,2,0)) #[h,w,c]
    plt.imshow(nping)
    pass


dataiter=iter(trainloader) #�������һ��train batch


images, labels=dataiter.next()


images.size()


imshow(tv.utils.make_grid(images)) #��������������ǰ����ɷ�ͼƴ��һ��ͼ��
print(labels)


class Net(nn.Module):
    def __init__(self):
        #����������ṹ���������ݣ�1*32*32
        super(Net,self).__init__()
        #��һ�㣺�����
        self.conv1=nn.Conv2d(3,6,3) #����3�㣬���6�㣬33�ľ����
        #�ڶ��㣺�����
        self.conv2=nn.Conv2d(6,16,3)
        #�����㣺ȫ���Ӳ�
        self.fc1=nn.Linear(16*28*28,512) #16*28*28=12544
        #���Ĳ㣺ȫ���Ӳ�
        self.fc2=nn.Linear(512,64)
        #����㣺ȫ���Ӳ�
        self.fc3=nn.Linear(64,10)
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


#��ʧ�����͸��¹���
criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(net.parameters(),lr=0.001,momentum=0.9)


for epoch in range(2):
    for i,data in enumerate(trainloader):
        images,labels=data
        
        outputs=net(images)
        
        loss=criteron(outputs,labels)
        
        #����Ȩ�ص�һ���������ݶ�
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if(iget_ipython().run_line_magic("1000==0):", "")
            print('Eopch:(), Step:(), Loss:()'.format(epoch,i,loss.item()))
            pass
        pass
    pass


#ģ�Ͳ���
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
print('׼ȷ��:',float(correct)/total)


torch.save(net.state_dict(),'F:\torch_model\pic_classifier.pt')


net2=Net()
net2.load_state_dict(torch.load('F:\torch_model\pic_classifier.pt'))
