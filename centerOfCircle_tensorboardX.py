import torch
import numpy as np
from torch import nn
from torch import optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter
#求三点的圆心
x_train = np.array([[0,2.2] ,[1,2.5] ,[2,2.2]]  ,dtype=np.float32 )
x_train = torch.from_numpy(x_train)



#print("x_train=",x_train)
c=np.array([[1],[1]]  ,dtype=np.float32 )
c = torch.from_numpy(c)
#print("c=",c)
#y_test=np.array([[1.2,2.2]] ,dtype=np.float32)
#y_test = torch.from_numpy(y_test)
#print("y_test=",y_test)
#print( "x_train-y_test =",x_train-y_test )
#print( "(x_train-y_test)*c =",torch.mm(x_train-y_test,c) )
#线性模型
class CenterOfCircle(nn.Module):
    def __init__(self):
        super(CenterOfCircle, self).__init__()
        self.linear = nn.Linear(6,2)  # 输入三点坐标3*2=6，输出一点坐标2*1=2

    def forward(self, x):
        out = self.linear(x.view(1,-1))
        centerOf = out.view( [1,2] )
        node =x.view([3,2])
        loss=torch.std(torch.sqrt(  torch.mm( (node-centerOf)*(node-centerOf) ,Variable(c)) ))#求输出一点到三点距离的均方差


        return out,loss

model = CenterOfCircle()
#优化函数
optimizer = optim.Adam(model.parameters(), lr=1e-4 , betas=(0.9, 0.99))  #试过SGD效果差，使用Adm

writer = SummaryWriter()
num_epochs = 20000
for epoch in range(num_epochs):
     inputs = Variable(x_train)
     # forward
     out,loss = model(inputs)  # 前向传播
     # backward
     optimizer.zero_grad() # 梯度归零
     loss.backward() # 反向传播
     optimizer.step() # 更新参数

     if (epoch + 1) % 20 == 0:
         print('Epoch[{}/{}], loss: {:.6f}'.format(epoch + 1, num_epochs, loss.data[0]))
         writer.add_scalar('loss', loss.data[0], epoch)
         for name, param in model.named_parameters():
             writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
model.eval()
predict = model(Variable(x_train))
print(predict[0].data)

writer.close()