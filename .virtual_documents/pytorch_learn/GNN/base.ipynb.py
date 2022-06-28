import torch
import torch_geometric
from torch_geometric.data import Data


edge_index=torch.tensor([[0,1,1,2],[1,0,2,1]],dtype=torch.long)
x=torch.tensor([[-1],[0],[1]],dtype=torch.float)
data=Data(x=x,edge_index=edge_index)
print(data)


from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


data[0]


class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add') #step 5
        self.lin=torch.nn.Linear(in_channels, out_channels)
    
    def forward(self,x,edge_index):
        '''
        x:[N,inchannels]
        edge_index:[2,E]
        '''
        
        #Add self-loops to the adjacency matrix.
        edge_index, _ =add_self_loops(edge_index, num_nodes=x.size(0))
        print(f'self loop edge index: {edge_index}\n')
        
        #Linear transform node feature matrix.
        print(f'init x\'s shape: {x.shape}\n')
        x=self.lin(x)
        print(f'after linear x: {x}, shape: {x.shape}\n')
        
        
        #Compute normalization coefficients.
        row,col = edge_index
        print(f'edge index: first: {row}, second: {col}\n')
        deg = degree(col, x.size(0), dtype=x.dtype) #计算所有节点的入度
        #进行操作： D^(-1/2)AD^(-1/2)
        deg_inv_sqrt=deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')]=0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        print(f'norm D^(-1/2)AD^(-1/2): {norm}\n')
        
        #Normalize node features
        #Sum up neighboring node features
        return self.propagate(edge_index, x=x,norm=norm)
    
    def message(self, x_j, norm):
        '''
        x_j:[E, out_channels]
        '''
        return norm.view(-1,1) * x_j


conv = GCNConv(1,2)
ret = conv(x,edge_index)


ret



