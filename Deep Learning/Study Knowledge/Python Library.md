### Torch 库

```python
import torch
from torch.utils import data
from torch import nn

# 定义
torch.tensor([2.0, 1, 4], [3 , 2, 1], [1, 2, 3]) # 返回张量（浮点数）
torch.zeros((2, 3, 4)) # 返回全 0 张量
torch.zeros_like(X) # 返回和 X 相同形状得全 0 张量
torch.ones((2, 3, 4)) # 返回全 1 张量
torch.arange(12) # 返回出 0 到 12 的所有整数（左闭右开区间）形成的一个一维 tensor
torch.normal(mean, std, size=(3, 4)) # 返回形状为 size 的张量，数值服从 N(mean, std^2)
x.clone() # 返回新张量，分配新内存，深拷贝
x.reshape(3, 4) # 返回将 x 改变形状后的新张量，改变形状但不改变元素数量和值

# 计算
torch.matmul(x, y) # 返回 x 与 y 的矩阵乘法结果
x.shape # 返回张量 x 的形状
x.numel # numel 返回 x 张量中的元素总数

# 微分
x = torch.tensor([1.0, 2, 3, 4], requires_grad=True) # 会同时创建一个 x.grad 变量记录梯度
y = 2 * x + 1 # 隐式构造 x 的计算图
y.backward() # 反向传递计算 y 内所跟踪的变量的梯度，并加到相关的 .grad 中
z = y.detach() * 2 # z = y * 2，但 z 不参与梯度计算（当作一个常量）

# 数据预处理
data.TensorDataset(features, labels) # 返回 features 与 labels 一一对应的 dataset 格式
data.DataLoader(dataset, batch_size, shuffle=is_train) # 返回将 dataset 以 batch_size 为单位分开的迭代器， shuffle 参数表示需要打乱顺序

# 神经网络层
nn.Sequential(model1， model2, …) # 返回一个序列容器，将多个层连接起来
nn.Flatten() # 返回一个平铺层，可将输入映射成一维张量输出
nn.Linear(2, 1) # 返回一个输入维度为 2，输出维度为 1 的线性回归层
model.weight.data.normal_(0, 0.01) # 使用正态分布替换模型 model 中的 w 参数
model.bias.data.fill_(0) # 使用 0 替换模型 model 中的 b 参数

# 损失函数
nn.MSELoss() # 返回一个 MSE 损失函数（平方范数）
nn.CrossEntropyLoss() # 返回一个交叉熵损失函数

# 优化器
torch.optim.SGD(params, lr=0.03) # 返回一个 SGD 优化器（随机梯度下降法）
optimizer.zero_grad() # 清空优化器 optimizer 的参数的梯度
optimizer.step() # 根据梯度更新参数
```



### Pandas 库

```python
import pandas as pd
import torch

data = pd.read_csv(data_file) # data 的格式是 DataFrame
inputs = data.iloc[:, 0:2] # iloc 用于取部分区域
meanVal = inputs.mean(numeric_only=True) # mean 用于取平均值（新版 pandas 若存在非数值需要指定参数 numeric_only=True，否则会报错）
inputs = inputs.fillna(meanVal) # fillna 用于填充缺失值
inputs = pd.get_dummies(DataFrame, dummy_na=True) # get_dummies 用于将离散值列向量拆分成布尔矩阵，dummy_na=True 参数会将 NaN 也视为一个类别
X = torch.tensor(DataFrame.values) # torch.tensor 用于将数值条目转换成张量，DataFrame.values 会将 DataFrame 转换成ndarray的多维数组格式
```

