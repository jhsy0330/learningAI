## 损失函数

#### 平方损失函数

$$
loss = \frac{1}{2}W^2
$$



## softmax

## MLP多层感知机

## 模型选择

K-折交叉验证

过拟合 和 欠拟合

VC 维

#### 数据复杂度

* 样本个数
* 样本元素个数
* 时间空间结构
* 多样性

### 过拟合处理: 权重衰退

$$
W^* = argmin{l(W,b)}+\frac{\lamda}{2}\abs{x}^2
$$

### 暂退法DropOut 

$$
\begin{split}\begin{aligned}
h' =
\begin{cases}
    0 & \text{ 概率为 } p \\
    \frac{h}{1-p} & \text{ 其他情况}
\end{cases}
\end{aligned}\end{split}
$$



### 权重初始化

* Xavier初始化

## 深度学习计算

### Pytorch中块和层的概念

* 自定义块
* 继承module父类

### 参数管理

### 访问参数

```python
import torch
from torch import nn

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
net(X)

print(net[2].state_dict())
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)
```

### 嵌套块访问参数

### 初始化网络参数

#### 初始化函数

```python
net.apply(my_init)
def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)
```

#### 直接设置

```python
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
net[0].weight.data[0]
```

#### 参数绑定

```python
# 我们需要给共享层一个名称，以便可以引用它的参数
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))
net(X)
# 检查参数是否相同
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# 确保它们实际上是同一个对象，而不只是有相同的值
print(net[2].weight.data[0] == net[4].weight.data[0])
```

### 自定义层

```python
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)
```

### 读写文件(加载和保存Tensor)

#### 加载和保存变量

```python
import torch
from torch import nn
from torch.nn import functional as F
import os

os.makedirs('./temp', exist_ok=True)
SavePath = './temp/tensor'
# 可以保存Tensor list dict
x = [torch.arange(4),torch.tensor([-1,-2,-3,-4])]
torch.save(x, SavePath)
```

#### 加载和保存模型参数

保存参数

```python
torch.save(net.state_dict(), SavePath)
```

加载参数

```python
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
clone.eval()
```

### GPU

```python
import torch
from torch import nn

torch.device('cpu'), torch.device('cuda'), torch.device('cuda:1')
# 查询可用数量
torch.cuda.device_count()
```

检测gpu

```python
def try_gpu(i=0):  #@save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():  #@save
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]"""
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

try_gpu(), try_gpu(10), try_all_gpus()
```

查询tensor所在设备

```python
x = torch.tensor([1, 2, 3])
x.device
```

存储在GPU上

```python
X = torch.ones(2, 3, device=try_gpu())
X
```

## 卷积神经网络

### 卷积层

### 卷积核

### 填充

### 步幅

### 通道

### 多输入多输出

