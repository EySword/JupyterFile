import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


import matplotlib.pyplot as plt
from torch.autograd import Variable


device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device


X = torch.linspace(1,-1,100)
X = X.view(-1,1)
Y = X.pow(2)+torch.rand(X.shape)*0.2


X = Variable(X).to(device)
Y = Variable(Y).to(device)


X_test = torch.linspace(1,-1,10)
X_test = X_test.view(-1,1)
Y_test = 3*pow(X_test,4)+1*pow(X_test,3)-pow(X_test,2)+1*X_test+torch.rand(X_test.shape)*0.01


plt.scatter(X.cpu(),Y.cpu(),s=1)


X=X.to(device).requires_grad_()
Y=Y.to(device).requires_grad_()
X_test=X_test.to(device)
Y_test=Y_test.to(device)


class Net(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(Net,self).__init__()
        self.in_dim,self.out_dim = input_dim, output_dim
        self.l1 = nn.Sequential(
            nn.Linear(self.in_dim, 1000),
            nn.ReLU())
        self.l2 = nn.Linear(1000,1)
    
    def forward(self,X):
        self.X = self.l1(X)
        self.X = self.l2(self.X)
        return self.X


net = Net(X.shape[1],Y.shape[1])
net.to(device)


optm = torch.optim.SGD(net.parameters(), lr=0.001)
loss = nn.MSELoss()


from tqdm import tqdm


train_loader = torch.utils.data.DataLoader(dataset=X,
                                           batch_size=10)
                                           


loss_list=[]


# net.train()
for epoc in tqdm(range(10000)):
    res = net(X)
    l = loss(res ,Y)

    net.zero_grad()
    l.backward()
    optm.step()
    if epocget_ipython().run_line_magic("100==0:", "")
        loss_list.append(l.item())


plt.figure()
plt.xscale('log')
plt.ylim((0,0.2))
plt.plot(loss_list)
plt.show


test=torch.linspace(-1,1,50).view(-1,1).to(device)
ite=iter(test)


ydata=[]
with torch.no_grad():
    for i,item in enumerate(ite):
        out=net(item)
        ydata.append(out.cpu())


plt.figure()
plt.scatter(test.cpu(),ydata,c='r')
plt.scatter(X.cpu(),Y.cpu())
plt.show()
