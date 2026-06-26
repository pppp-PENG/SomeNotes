### 1. 求导相关

##### 求导计算

机器学习中，求导结果默认使用分子布局

- 标量对标量求导：结果为标量

- ==标量对列向量 (n, 1) 求导：结果为行向量 (1, n)==

$$
\frac{\partial}{\partial \vec x} a = \vec 0^{T} \qquad \qquad \qquad
\frac{\partial}{\partial \vec x} au = a \frac{\partial u}{\partial \vec x} \qquad \qquad \qquad
\frac{\partial}{\partial \vec x} \sum(\vec x) = \vec 1^{T} \\
\frac{\partial}{\partial \vec x} \lVert \vec x \rVert^2_2 = 2 \vec x^{T} \qquad \qquad \qquad \qquad
\frac{\partial}{\partial \vec x} <\vec u, \vec v> = u^T \frac{\partial \vec v}{\partial \vec x} + v^T \frac{\partial \vec u}{\partial \vec x}
$$

- 列向量 $(n, 1)$ 对列向量 $(k, 1)$ 求导：结果为矩阵 $(n, k)$

##### Pytorch 自动求导

使用链式法则反向传递计算

1. 声明 `x` 变量时使用 `requires_grad` 参数对 x 变量进行跟踪，并创建一个变量 `x.grad` 用于存储 `x` 的梯度
2. 每次有 `x` 参与的运算都会记录相关的公式，形成一个关于该变量的有向无环的计算图
3. 当使用 `y.backward()` 方法时，将根据计算图从 `y` 变量开始反向传递计算导数到 `x` 变量，再将结果相乘得出 `x` 变量的梯度（链式法则）
4. 将计算的梯度加到 `x.grad` 上（不断使用 `.backward()` 方法便会不断累积）

### 2. 深度学习基础

##### 误差

- 训练误差：模型在训练数据上的误差
- 泛化误差：模型在新数据上的误差

##### 数据集

- 验证数据集：用于评估模型好坏的数据集（用于调超参数）
- 测试数据集：只使用一次的数据集

##### K-折交叉验证

当没有足够多的数据集时使用，常将 $K$ 设置为 5 或 10

1. 将数据集分为 $K$ 块
2. `for i in range(1, K + 1)` 使用第 i 块数据集作为验证数据集，其余作为训练数据集
3. 报告 $K$ 个验证数据集误差的平均

##### VC 维

对于一个分类模型，VC 等于不论给任何标号该模型都能完美分类的数据集大小

例如：$N$ 维输入的感知机的 VC 维是 $N + 1$

##### 正则化

- 权重衰退法（Weight Decay）【优化器使用】

为防止过拟合，通过 $L_2$ 正则项使得模型参数不会过大，从而控制模型复杂度
$$
计算梯度时，增加正则项 \ \frac{\lambda}{2} \lVert \omega \lVert^2_2 \\
有 \ grad = \frac{\partial}{\partial \omega}(l(\omega, b) + \frac{\lambda}{2} \lVert \omega \lVert^2_2) = \frac{\partial l(\omega, b)}{\partial \omega} + \lambda \ \omega \\
故参数更新为 \ \omega_{t+1} = \omega_t - \eta \ grad = (1 - \eta \ \lambda)\omega_t - \eta \frac{\partial l(\omega, b)}{\partial \omega}
$$
因此该方法等价于让 $\omega$ 每次更新时先缩小数值再向负梯度方向前进，超参数为 $\eta$

- 丢弃法（Dropout）【丢弃层】

$$
\text{for each } x_i \ \text{in } \vec x: \quad x_i' = \begin{cases} 0 & \text{with probablity } p \\ \frac{x_i}{1 - p} & \text{otherwise} \end{cases} \\
且有: \quad E(x_i') = x_i
$$

##### 数值稳定性

$$
由链式法则可知，对于 \ d \ 层神经网络的第 \ t \ 层的参数 \ \omega_t \ 求梯度，有: \quad \frac{\partial l}{\partial \omega_t} = \frac{\partial l}{\partial h_d} \frac{\partial h_d}{\partial h_{d-1}}…\frac{\partial h_t}{\partial \omega_t}
$$

越接近输入层的参数的梯度需要==越多的矩阵乘法==计算，此时便会发生梯度爆炸或梯度消失

- 梯度爆炸：当矩阵乘法中的每一项都大于 0 的时候，靠近底部输入层的参数的梯度便会十分大，导致参数每次更新步幅过大
- 梯度消失：当激活函数如 sigmoid 的导数容易接近 0 时，矩阵乘法的每一项都可能小于1，使得最后求得梯度小于浮点数下限导致梯度变成 0，参数无法更新

提高数值稳定性的方法：

- 将矩阵乘法改为加法（如 ResNet、LSTM）
- 归一化（如梯度归一化、梯度裁剪）
- 设置合理的权重初始值和激活函数

##### 数据增强

通过修改已有数据集，生成新数据集，使数据集具有多样性

##### 微调（Fine-Tuning）

<img src="C:\Users\asus\AppData\Roaming\Typora\typora-user-images\image-20250425133712215.png" alt="image-20250425133712215" style="zoom:50%;" />

### 3. 神经网络层

##### 线性回归（Linear Regression）

用于连续值的预测问题，给定特征预测标签的值

$$
\text{batch\_size} = m \quad \text{features.shape} = n \\
\quad \hat y_{m \times 1} = X_{m \times n} \ \omega_{n \times 1} + \begin{bmatrix} b \\ \vdots \\ b \end{bmatrix} \\
$$
输入集 $X_{m \times n}$ 中的 $x_{ij}$ 表示第 $i$ 个样本的第 $j$ 个特征值；输出集 $\hat y_i$ 表示第 $i$ 个样本的预测值

##### Softmax 逻辑回归（Softmax Logical Regression）

用于分类问题，给定特征预测每个类别的概率

$$
\text{batch\_size} = m \quad \text{num\_inputs} = n \quad \text{num\_outputs} = K \\
\hat o_{m \times K} = X_{m \times n} \ \omega_{n \times K} + \begin{bmatrix} b_{1 \times K} \\ \vdots \\ b_{1 \times K} \end{bmatrix} \\
\hat y_{m \times K} = softmax(\hat o_{m \times K}) \\
其中，\hat y_{ij} = \frac{e^{o_{ij}}}{\sum_{k=1}^K{e^{o_{ik}}}}
$$
输入数据集 $X_{i \times j}$ 中表示第 $i$ 个样本向量的第 $j$ 个特征的值；输出集 $\hat y_{i \times j}$ 表示第 $i$ 个样本属于第 $j$ 类的预测概率

##### 多层感知机（MLP, Multilayer Perceptron）

相比 SVM 更易扩展
$$
\text{batch\_size} = m \quad \text{num\_inputs} = n \quad \text{num\_outputs} = K \\
\text{num\_hiddens} = L \quad \text{num\_dims\_hidden\_i} = l_i \\
h_{1 } = \sigma(X_{m \times n} \ \omega_{n \times l_1} + \begin{bmatrix} b_{1 \times l_1} \\ \vdots \\ b_{1 \times l_1} \end{bmatrix}) \\
h_{i + 1} = \sigma(h_i \ \omega_{l_i \times l_{i+1}} + \begin{bmatrix} b_{1 \times l_i} \\ \vdots \\ b_{1 \times l_i} \end{bmatrix}) \\
\hat o = \sigma(h_m \ \omega_{l_m \times K} + \begin{bmatrix} b_{1 \times K} \\ \vdots \\ b_{1 \times K} \end{bmatrix}) \\
$$
输入 $\vec x$ 表示一个样本向量，有 $n$ 个特征值；$\vec h_i$ 表示第 $i$ 个隐藏层；$\sigma(x)$ 为激活函数（用于引入非线性性，截断本层与后层的线性相关），主要有三种：
$$
Sigmoid(x) = \frac{1}{1 + e^{-x}} \\
tanh(x) = \frac{1 - e^{-2x}}{1 + e^{-2x}} \\
ReLU(x) = max(x, 0)
$$
$ReLU$ 激活函数少了指数运算，减少计算量

##### 卷积层（Convolutional Layer）

【可改变输出的通道数，影响输出的宽高，线性】

特点：

- 局部特征提取：以卷积核为视野进行局部特征提取
- 提高计算效率：通过共享参数减少参数量，提高计算效率
- 保留空间信息

二维多输入多输出通道卷积层
$$
\text{kernal\_size} = K \quad \text{input\_channel} = c_i \quad \text{output\_channel} = c_o \\
\text{input\_height} = n_h \quad \text{input\_width} = n_w \quad \text{output\_height} = m_h \quad \text{output\_width} = m_w \\
out_{c_i \times m_h \times m_w} = Conv2d(X_{c_o \times n_h \times n_w}) \\
其中: \quad o_{m, i, j} = \sum_{n}^{c_i} f(\omega_{{(m, n,)}_{k \times k}},  X_{{(n,)}_{n_h \times n_w}}) \\
f(\omega, X) \ 表示使用卷积核 \ \omega \ 对 \ X \ 进行操作
$$

##### 池化层（Pooling Layer）

【不改变输出的通道数，但可影响输出的宽高，非线性】

缓解卷积层对于位置的敏感性（将卷积层输出的边缘锐度模糊化）
$$
\text{input\_channel} = \text{output\_channel} = c \\
\text{input\_height} = n_h \quad \text{input\_width} = n_w \quad \text{output\_height} = m_h \quad \text{output\_width} = m_w \\
out_{c \times m_h \times m_w} = pool2d(X_{c \times n_h \times n_w}) \\
其中: \quad o_{m, i, j} = \sum_{n}^{c_i} f(k,  X_{{(n,)}_{n_h \times n_w}}) \\
f(k, X) \ 表示使用形状为 \ k \times k \ 的窗口对 \ X \ 进行最大/平均池化操作
$$

##### 批量归一化层（BN, Batch Normalization）

【不改变输出的通道数和宽高，线性】

- 对于卷积层，作用在通道维上（对每一个通道进行批量归一）
- 对于全连接层，作用在特征维上（对每一个特征进行批量归一）

$$
\text{batch\_size} = m \\
\mu_x = \frac{1}{m} \sum_i^m x_i \quad \sigma_x = \frac{1}{m} \sum_i^m (x_i - \mu_x)^2 + \epsilon \\
x' = \gamma \frac{x - \mu_x}{\sigma_x} + \beta
$$



解决深层的神经网络中顶层和底层参数在反向传播中的梯度差距过大的问题

### 4. 损失函数

##### 回归常用

- 绝对误差损失（L1 Loss, Mean Absolute Error Loss）

$$
MAELoss(\hat y_i, y_i) = |\hat y_i - y_i|
$$

- 均方误差损失（L2 Loss, Mean Square Error Loss）

$$
MSELoss(\hat y_i, y_i) = \frac{1}{2} (\hat y_i - y_i) ^ 2
$$

- 平滑误差损失（Huber Loss）

$$
HuberLoss(\hat y_i, y_i) = \begin{cases}
	|\hat y_i - y_i| & \text{if } |\hat y_i - y_i| > 1 \\
	(\hat y_i - y_i) ^ 2 & \text{otherwise}
\end{cases}
$$

##### 分类常用

- 交叉熵损失（Cross Entropy Loss）：衡量两个概率分布的区别

$$
CrossEntropyLoss = \sum_i -p_i log(q_i) \\
假设该真实数据属于第 \ i \ 类，有 \ (p_i = 1) \and (\forall j \ne i, p_j = 0) \\
故:CrossEntropyLoss = -log(q_i)
$$

### 5. 优化算法

##### 小批量随机梯度下降法（SGD, Stochastic Gradient Descent）

$$
记第 \ t \ 个批量的 \ Loss \ 为 \ l(\omega_t) \\
则参数更新规则为: \\
g_t = \frac{1}{b} \triangledown l(\omega_t) \\
\omega_t = \omega_{t-1} - \mu g_t
$$

##### 冲量法

$$
记第 \ t \ 个批量的 \ Loss \ 为 \ l(\omega_t) \\
则参数更新规则为: \\
g_t = \frac{1}{b} \triangledown l(\omega_t) \\
v_t = g_t + \beta v_{t-1} \\
\omega_t = \omega_{t-1} - \mu v_t
$$

##### Adam

$$
记第 \ t \ 个批量的 \ Loss \ 为 \ l(\omega_t) \\
则参数更新规则为: \\
g_t = \frac{1}{b} \triangledown l(\omega_t) \\
v_t = (1 - \beta_1)g_t + \beta_1 v_{t-1} \qquad \hat v_t = \frac{v_t}{1 - \beta_1^t} \\ s_t = (1 - \beta_1)g_t + \beta_1 s_{t-1} \qquad \hat s_t = \frac{s_t}{1 - \beta_1^t} \\
g_t' = \frac{\hat v_t}{\sqrt{\hat s_t} + \epsilon} \\
\omega_t = \omega_{t-1} - \mu g_t'
$$

### 6. 卷积神经网络（Convolutional Neural Networks, CNN）

##### LeNet

​	输入：$X$（$1 @ 28 \times 28$）

1. 卷积层：$kernal\_size=5, stride=1, padding=2 \ (6 @ 28 \times 28)$，sigmoid 激活

    平均池化层：$kernal\_size=2, stride=2, padding=0 \ (6 @ 14 \times 14)$

2. 卷积层：$kernal\_size=5, stride=1, padding=0 \ (16 @ 10 \times 10)$，sigmoid 激活

    平均池化层：$kernal\_size=2, stride=2, padding=0 \ (16 @ 5 \times 5)$

3. 全连接层：$\omega_{400 \times 120}$，sigmoid 激活（$120$）

4. 全连接层：$\omega_{120 \times 84}$，sigmoid 激活（$84$）

5. softmax 层：$\omega_{84 \times 10}$，softmax 回归预测（$10$）

​	输出：$Y$（$10$）

##### AlexNet

​	输入：$X$（$3 @ 224 \times 224$）

1. 卷积层：$kernal\_size=11, stride=4, padding=1 \ (96 @ 54 \times 54)$，ReLU 激活

    最大池化层：$kernal\_size=3, stride=2, padding=0 \ (96 @ 26 \times 26)$

2. 卷积层：$kernal\_size=5, stride=1, padding=2 \ (256 @ 26 \times 26)$，ReLU 激活

    最大池化层：$kernal\_size=3, stride=2, padding=0 \ (256 @ 12 \times 12)$

3. 卷积层：$kernal\_size=3, stride=1, padding=1 \ (384 @ 12 \times 12)$，ReLU 激活

4. 卷积层：$kernal\_size=3, stride=1, padding=1 \ (384 @ 12 \times 12)$，ReLU 激活

5. 卷积层：$kernal\_size=3, stride=1, padding=1 \ (256 @ 12 \times 12)$，ReLU 激活

    最大池化层：$kernal\_size=3, stride=2, padding=0 \ (256 @ 5 \times 5)$

6. 全连接层：$\omega_{6400 \times 4096}$，ReLU 激活（$4096$），dropout（p=0.5） 丢弃

7. 全连接层：$\omega_{4096 \times 4096}$，ReLU 激活（$4096$），dropout（p=0.5） 丢弃

8. softmax 层：$\omega_{4096 \times 1000}$，softmax 回归预测（$1000$）

​	输出：$Y$（$1000$）

##### VGG 块

【可改变输出的通道数，输出的宽高减半，非线性】

<img src="C:\Users\asus\AppData\Roaming\Typora\typora-user-images\image-20250423121547129.png" alt="image-20250423121547129" style="zoom: 33%;" />

1. 多个卷积层（$kernal\_size=3, stride=1, padding=1$），ReLU 激活【第一个卷积层可控制输出通道数，输出的宽和高不变】

2. 一个池化层（$kernal\_size=2, stride=2, padding=0$），ReLU 激活【输出的宽和高减半】

##### NiN（Network in Network） 块

【可改变输出的通道数，输出的宽高不变，非线性】

使用 $1 \times 1$ 的卷积核充当全连接层，以减少参数量级，避免过拟合

<img src="C:\Users\asus\AppData\Roaming\Typora\typora-user-images\image-20250423121605719.png" alt="image-20250423121605719" style="zoom:50%;" />

1. 一个卷积层，ReLU 激活【可控制输出的通道数】
2. 两个卷积层（$kernal\_size=1, stride=1, padding=0$），分别接 ReLU 激活【输出的宽和高不变】

##### NiN 架构

==使用 $1 \times 1$ 的卷积核、使用全局平均池化==

1. 交替使用 NiN 块和最大化池化层（$kernal\_size=3, stride=2, padding=0$）【每次池化高宽约减半】
2. 最后一个 NiN 块并控制输出通道数为结果个数
3. 全局平均池化得到输出【高宽都化为 1，展平后变成长为通道数的向量输出】

##### Inception 块

对输入同时用不同的层处理，最后合并（各处理结果高宽相同，在通道维度上合并），各层输出的高宽不变，但通道数会变

<img src="C:\Users\asus\AppData\Roaming\Typora\typora-user-images\image-20250423121649809.png" alt="image-20250423121649809" style="zoom:50%;" />

##### GoogLeNet

1. 卷积 + 最大池化
2. 卷积 * 2 + 最大池化
3. Inception 块 * 2 + 最大池化
4. Inception 块 * 5 + 最大池化
5. Inception 块 * 2 + 全局平均池化
6. 全连接

##### 残差块

<img src="C:\Users\asus\AppData\Roaming\Typora\typora-user-images\image-20250423114444327.png" alt="image-20250423114444327" style="zoom:50%;" />

##### ResNet

1. 一系列残差块
2. 全局平均池化层
3. 全连接层

### 7. 计算机视觉（Computer Vision, CV）

##### 物体检测

- 数据集：一般每个样本有 6 个特征：所属图片 id、四元组（标记真实边缘框位置及形状）、标签
- 锚框：以每个像素为中心生成不同大小不同形状的锚框
- 交并比（IoU, Intersection over union）：用于找出与目标边缘框重合度最高的锚框

##### RoI（兴趣区域）池化层

<img src="C:\Users\asus\AppData\Roaming\Typora\typora-user-images\image-20250426203750357.png" alt="image-20250426203750357" style="zoom:50%;" />

- 给定一个锚框，均匀分割成 $n \times m$ 块，输出每个块的最大值
- 无论锚框多大，输出高宽总是为 $m \times n$

##### R-CNN

<img src="C:\Users\asus\AppData\Roaming\Typora\typora-user-images\image-20250426203533598.png" alt="image-20250426203533598" style="zoom:50%;" />

- 使用启发式搜索算法选择锚框
- 使用卷积神经网络来对目标分类

##### Fast R-CNN

<img src="C:\Users\asus\AppData\Roaming\Typora\typora-user-images\image-20250426204931196.png" alt="image-20250426204931196" style="zoom:50%;" />

- 使用 CNN 对图片抽取特征
- 使用 RoI 池化层对每个锚框生成固定长度特征

##### Faster R-CNN

<img src="C:\Users\asus\AppData\Roaming\Typora\typora-user-images\image-20250426205002429.png" alt="image-20250426205002429" style="zoom:50%;" />

- 使用一个区域提议网络来替代启发式搜索来获得更好的锚框

##### SSD

<img src="C:\Users\asus\AppData\Roaming\Typora\typora-user-images\image-20250426205204908.png" alt="image-20250426205204908" style="zoom:50%;" />

- 一个基础网络来抽取特征，然后用多个卷积层块来减半高宽
- 在每段生成锚框
- 对每个锚框预测类别和边缘框

##### YOLO

- 将图片均匀分成 $S \times S$ 个锚框
- 每个锚框预测 $B$ 个边缘框

##### 语义分割

- 转置卷积（Transposed convolution）：用于增大高宽

$$
Y[i:i+h, j:j+w] += X[i, j] \cdot K
$$
其中，$Y$ 为输出矩阵，$X$ 为输入矩阵，$K$ 为卷积核

- FCN（全连接卷积神经网络）：使用去除了最后两层的 CNN 抽取特征（全局池化层和全连接层）

<img src="C:\Users\asus\AppData\Roaming\Typora\typora-user-images\image-20250426213725340.png" alt="image-20250426213725340" style="zoom:50%;" />

##### 样式迁移

<img src="C:\Users\asus\AppData\Roaming\Typora\typora-user-images\image-20250427125937269.png" alt="image-20250427125937269" style="zoom:50%;" />

- 使用 CNN 的某一层输出表示该图片的样式，某一层输出表示该图片的内容
- 内容损失：预测内容抽取结果张量与原图内容抽取结果张量的均方差
- 样式损失：预测样式抽取结果在各通道的分布与原样式抽取结果在相对应各通道的差异

### 8. 循环神经网络（Recurrent Neural Networks, RNN）

##### 马尔可夫假设（Markov）

假设当前数据只和过去 $\tau$ 个数据点相关
$$
p(x_t \vert x_1,...,x_{t-1}) = p(x_t \vert x_{t-\tau},...,x_{t-1}) = p(x_t \vert f(x_{t-\tau},...,x_{t-1}))
$$

##### 潜变量自回归模型

引入潜变量 $h_t$ 表示过去信息 $f(x_1,...,x_{t-1})$
$$
p(x_t \vert x_1,...,x_{t-1}) = p(x_t \vert f(x_1,...,x_{t-1})) = p(x_t \vert h_t)
$$

##### 文本预处理

1. 将文本分词
2. 统计词频并按高到低排序然后编号（编号数越大词频越小）

##### 语言模型

用联合概率分布预测下一个词出现的概率，一般都基于马尔可夫假设

统计词频时也统计连续 $\tau$ 个词的不同序列出现的频次

##### 循环神经网络（Recurrent Neural Networks, RNN）

$$
h_t = \tanh(h_{t-1} W_{h\times h} + x_t W_{x\times h} + b) \\
Y_t = h_t W_{h\times o} + b
$$

损失函数使用困惑度（perplexity）：平均交叉熵做自然指数
$$
\pi = \frac{1}{n}\sum_{i=1}^n - \text{log} p(x_t \vert x_{t-1},...) \\
perplexity = e^\pi
$$


使用梯度裁剪预防梯度爆炸：若梯度长度超过 $\theta$ 则拖影回长度$\theta$
$$
\bf g \leftarrow \min(1, \frac{\theta}{\lVert \bf g\rVert}) \bf g
$$

##### 门控循环单元（Gate Recurrent Unit, GRU）

$$
R_t = \sigma(X_tW_{x\times r} + H_{t-1}W_{h\times r} + b) \\
Z_t = \sigma(X_tW_{x\times z} + H_{t-1}W_{h\times z} + b) \\
\tilde H_t = \tanh(X_tW_{x\times h} + (R_t \odot H_{t-1})W_{h\times h} + b) \\
H_t = Z_t \odot H_{t-1} + (1 - Z_t) \odot \tilde H_t \\
Y_t = h_t W_{h\times o} + b
$$

其中，$\sigma$ 表示 $\text{sigmoid}$ 激活函数，$R_t$ 表示遗忘门（Reset gate），$Z_t$ 表示更新门（Update gate），$\tilde H_t$ 表示候选隐状态，$H_t$ 表示隐状态

<img src="C:\Users\asus\AppData\Roaming\Typora\typora-user-images\image-20250428211219375.png" alt="image-20250428211219375" style="zoom:50%;" />

##### 长短期记忆网络（Long Short-Term Memory, LSTM）

$$
I_t = \sigma(X_tW_{x\times i} + H_{t-1}W_{h\times i} + b) \\
F_t = \sigma(X_tW_{x\times f} + H_{t-1}W_{h\times f} + b) \\
O_t = \sigma(X_tW_{x\times o} + H_{t-1}W_{h\times o} + b) \\
\tilde C_t = \tanh(X_tW_{x\times c} + H_{t-1}W_{h\times c} + b) \\
C_t = F_t \odot C_{t-1} + I_t \odot \tilde C_t \\
H_t = O_t \odot \tanh(C_t)
$$

其中，$\sigma$ 表示 $\text{sigmoid}$ 激活函数，$I_t$ 表示输入门（Input gate），$F_t$ 表示遗忘门（Forget gate），$O_t$ 表示输出门（Output gate），$\tilde C_t$ 表示候选记忆元，$C_t$ 表示记忆元，$H_t$ 表示隐状态

<img src="C:\Users\asus\AppData\Roaming\Typora\typora-user-images\image-20250429203536607.png" alt="image-20250429203536607" style="zoom:50%;" />

##### 深度循环神经网络

$$
H_t^1 = f_1(H_{t-1}^1, X_t) \\
... \\
H_t^i = f_i(H_{t-1}^i, H_i^{i-1}) \\
... \\
O_t = g(H_t^L)
$$

其中，$H_t^i$ 表示 $t$ 时刻第 $i$ 层的隐藏层，$O_t$ 表示 $t$ 时刻的输出

<img src="C:\Users\asus\AppData\Roaming\Typora\typora-user-images\image-20250429213747597.png" alt="image-20250429213747597" style="zoom:50%;" />



##### 双向循环神经网络

$$
\overrightarrow H_t = \phi(X_t W_{x\times h}^{(f)} + \vec H_{t-1} W_{h\times h}^{(f)} + b^{(f)}) \\
\overleftarrow H_t = \phi(X_t W_{x\times h}^{(b)} + \vec H_{t-1} W_{h\times h}^{(b)} + b^{(b)}) \\
H_t = [\overrightarrow H_t, \overleftarrow H_t] \\
\vec O_t = H_t W_{h\times q} + b
$$

<img src="C:\Users\asus\AppData\Roaming\Typora\typora-user-images\image-20250430134656357.png" alt="image-20250430134656357" style="zoom:50%;" />

##### Word2vec

##### Seq2Seq (Sequence to Sequence)

- Encoder
    - Embedding 将输入特征维度从 one-hot 编码转成可学习的语义空间
    - 使用无全连接层的循环神经网络
    - 保留网络的输出和 Encoder 最后隐藏状态
- Decoder
    - Embedding 将输入特征维度从 one-hot 编码转成可学习的语义空间
    - 循环神经网络初始隐藏状态为 Encoder 最后隐藏状态，且==输入特征维度扩张为（Encoder 输出 + Decoder 输入）==
    - 使用完整的循环神经网络，全连接层的输出特征维度对齐输入的特征维度
- 损失函数
    - 使用交叉熵损失计算损失函数（看成 vocab 个类的分类问题）
    - 为了统一输入维度，需对短序列进行填充（Padding），因此==使用 mask 将不相关的填充项清 0==，避免进入反向传播


<img src="C:\Users\asus\AppData\Roaming\Typora\typora-user-images\image-20250506164749284.png" alt="image-20250506164749284" style="zoom:50%;" />

##### 束搜索（Beam Search）

- 保存最好的 $k$ 个候选
- 在每个时刻，对每个候选新加一项（$n$ 种可能），在 $kn$ 个选项中选出最好的 $k$ 个
- 时间复杂度：$O(knT)$

<img src="C:\Users\asus\AppData\Roaming\Typora\typora-user-images\image-20250505153908387.png" alt="image-20250505153908387" style="zoom:50%;" />

### 9. 注意力机制（Attention Mechanism）

##### 注意力机制（Attention Mechanism）

用 Query 加权 Key-Value 对，以获得合适的 Value 值
$$
O = Q \cdot K^T \cdot V
$$

##### 自注意力（Self Attention）

<img src="C:\Users\asus\AppData\Roaming\Typora\typora-user-images\image-20250512200138673.png" alt="image-20250512200138673" style="zoom:50%;" />

##### 多头注意力（Multi-head Attention）

<img src="C:\Users\asus\AppData\Roaming\Typora\typora-user-images\image-20250512201105302.png" alt="image-20250512201105302" style="zoom: 67%;" />

##### Transformer

- Position Encoding：位置编码，对输入的值加上位置信息值
- Position Wise FFN：基于位置的前馈网络，使用单隐藏层的 MLP
- Add & Norm：矩阵相加，使用 Layer Normalization

<img src="D:\codes\DeepLearning\handWriteModels\Transformer\imgs\transformer stucture.png" alt="image-20250509205236370" style="zoom:50%;" />

