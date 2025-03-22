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
