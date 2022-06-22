import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


device


x=torch.linspace(-1,1,100)
x=torch.unsqueeze(x,1)
y=x.pow(2)+torch.rand(x.size())*0.2
x,y = Variable(x).to(device), Variable(y).to(device)


plt.scatter(x.cpu().numpy(),y.cpu().numpy())


class Net(torch.nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden=torch.nn.Linear(n_feature,n_hidden)
        self.predict=torch.nn.Linear(n_hidden,n_output)
        pass
    def forward(self,x):
        x=self.hidden(x)
        x=F.relu(x)
        
        x=self.predict(x)
        return x
    pass


net=Net(1,1000,1)
net.to(device)


criterion=F.mse_loss
optimizer=torch.optim.SGD(net.parameters(),lr=0.001)


loss_list=torch.tensor([]).to(device)


import time


s=time.process_time()
list_cpu=[]
for i in range(10000):
    predictions=net(x)
    loss=criterion(predictions,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    list_cpu.append(loss.item())
#     torch.cat((loss_list,torch.tensor(loss.item())))
    
#     if i % 10 == 0:
#         plt.cla()  # Clear axis即清除当前图形中的当前活动轴。其他轴不受影响
#         plt.scatter(x.data.numpy(),y.data.numpy())
#         plt.plot(x.data.numpy(),predictions.data.numpy(),'r',lw=3)
#         plt.text(0.5,0,'Loss'+str(loss.item()))
#         plt.pause(0.1)
#         pass
#     pass
# plt.ioff()
# plt.show()
e=time.process_time()
print('time: ',e-s)


# loss_list.pop(0)
plt.figure()
plt.xscale('log')
plt.plot(list_cpu)
plt.show



