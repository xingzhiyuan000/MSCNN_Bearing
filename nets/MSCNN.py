import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary

# 搭建神经网络

# 卷积-标准化-激活-最大池化
class conv1d_bn_relu_maxpool(nn.Module):
    def __init__(self, ch_in, ch_out, k, p, s):
        super(conv1d_bn_relu_maxpool, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.k = k #卷积核大小
        self.p = p #padding
        self.s = s #stride 步幅

        self.conv1=nn.Conv1d(in_channels=ch_in,out_channels=ch_out,kernel_size=k,stride=s,padding=p)
        self.bn1 = nn.BatchNorm1d(num_features=ch_out)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2,padding=0)


    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        return x


class MSCNN(nn.Module):
    def __init__(self):
        super(MSCNN, self).__init__()

        self.avg02 = nn.AvgPool1d(kernel_size=2)
        self.avg03 = nn.AvgPool1d(kernel_size=3)

        self.conv_bn_relu_maxpool1 = conv1d_bn_relu_maxpool(ch_in=1, ch_out=16, k=100, p=50, s=1)
        self.conv_bn_relu_maxpool2 = conv1d_bn_relu_maxpool(ch_in=16, ch_out=32, k=100, p=50, s=1)

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(in_features=29344, out_features=1024, bias=True)
        self.fc2 = nn.Linear(in_features=1024, out_features=13, bias=True)


    def forward(self, x):
        x01=x
        x02 = self.avg02(x)
        x03 = self.avg03(x)

        x1=self.conv_bn_relu_maxpool1(x01)
        x1 = self.conv_bn_relu_maxpool2(x1)
        x1 = self.flatten(x1)

        x2 = self.conv_bn_relu_maxpool1(x02)
        x2 = self.conv_bn_relu_maxpool2(x2)
        x2 = self.flatten(x2)

        x3 = self.conv_bn_relu_maxpool1(x03)
        x3 = self.conv_bn_relu_maxpool2(x3)
        x3 = self.flatten(x3)

        output = torch.cat((x1, x2, x3), -1)

        output = self.dropout(output)
        output = self.fc1(output)
        output = self.fc2(output)

        return output


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Model=MSCNN() #实例化网络模型
    wang_DS_RGB=Model.to(device) #将模型转移到cuda上
    input=torch.ones((64,1,2000)) #生成一个batchsize为64的，通道数为1，宽度为2048的信号
    input=input.to(device) #将数据转移到cuda上
    output=Model(input) #将输入喂入网络中进行处理
    print(output.shape)
    summary(Model,input_size=(1,2000)) #输入一个通道为1的宽度为2048，并展示出网络模型结构和参数
