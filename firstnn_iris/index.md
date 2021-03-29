# TensorFlow2.1入门学习笔记(5)——构建第一个神经网络，鸢尾花分类问题（附源码）

 根据前面的基础知识，可以开始第一个神经网络的搭建，主要学习的资料西安科技大学：[神经网络与深度学习——TensorFlow2.0实战](https://www.icourse163.org/learn/XUST-1206363802#/learn/announce)，北京大学：[人工智能实践Tensorflow笔记](https://www.icourse163.org/learn/PKU-1002536002#/learn/announce)
<!-- more -->

### 1.问题背景

#### 问题描述

![](https://img-blog.csdnimg.cn/20200512181537793.png " ")

人们通过经验总结出的规律：通过测量鸢尾花的花萼长、花萼宽、花瓣长、花瓣宽，可以得出鸢尾花的类别。（如：花萼长>花萼宽 且 花瓣长/花瓣宽>2 则为杂色鸢尾花）。大量依靠人工分类工作量巨大，不同的人员分类，标准，准确率都会有所差距。可以借助深度学习来学习其中的特征并对新数据进行预测。
#### 流程设计
  - 大量的[花萼长、花萼宽、花瓣长、花瓣宽（输入特征），对应的类别（标签）]数据对构成数据集
  - 把数据集喂入搭建好的神经网络结构
  - 网络优化参数得到模型
  - 模型读入新输入特征，输出识别结果


![](https://img-blog.csdnimg.cn/20200512181625192.png " ")

#### 模型设计
###### 搭建网络模型

![](https://img-blog.csdnimg.cn/20200512181759189.png " ")

###### 转换为数学模型
所有输入特征x与相应特征权重w相乘加上偏置项b输出结果y。
x：一行四列矩阵，对应四个特征
w：四行三列矩阵
b：3个偏置项
y：一行三列矩阵，对应三种类别的可信度

![](https://img-blog.csdnimg.cn/20200512181930231.png " ")

###### 搭建网络
每个神经元$y_0,y_1,y_2与输入节点x_0,x_1,x_2,x_3$都有联系，称为全连接神经网络权重w与偏置项b会随机初始化一组参数

![](https://img-blog.csdnimg.cn/20200512185443354.png " ")

###### 前向传播
神经网络执行y = x * w + b的过程称为前向传播

![](https://img-blog.csdnimg.cn/20200512185915493.png " ")

###### 损失函数

损失函数：预测值(y)与标准答案($y_i$)的差距，可以定量判断w，b的优劣，当损失函数输出最小时会出现最优解。（有多种损失函数，这里用均方误差）

  - 均方误差：$MSE(y,y_i)=\frac{\sum_{k=0}^n(y-y_i)^2}{n}$


![](https://img-blog.csdnimg.cn/20200512215140250.png " ")

###### 梯度下降
目的：找到一组参数w和b，使得损失函数最小。
梯度：函数对个参数求偏导后的向量，梯度下降的方向是函数减小的方向。
<font color=red>梯度下降：</font>延损失函数梯度下降的方向，寻找损失函数的最小值，得到最优参数。
<font color=red>学习率(learning rate, lr)：</font>当学习率设置过小时，收敛过程将变得十分缓慢。当学习率设置过大时，梯度可能会在最小值附近震荡，甚至无法收敛

![](https://img-blog.csdnimg.cn/2020051221584380.png " ")

###### 反向传播
$w_{t+1}=w_t-lr*\frac{\partial loss}{\partial w_t}$
从前向后，逐层求损失函数对每层神经元参数的偏导数，迭代更新所有参数。

![](https://img-blog.csdnimg.cn/20200512221012828.png " ")

### 2.数据读入
#### 数据集介绍：
该数据集已集成在sklearn包中，可直接调入使用，数据共有150组，每组包括花萼长、花萼宽、花瓣长、花瓣宽共四个输入特征。同时给出了这一组特征的的对应鸢尾花类别。类别包括Setosa Iris（狗尾草鸢尾），Versicolour Iris（杂色鸢尾），Viginaica Iris（弗吉尼亚鸢尾）三类，分别用数字0，1，2表示

![](https://img-blog.csdnimg.cn/20200512182800425.png " ")

  - 从sklearn包datasets读入数据集
```python
from sklearn.datasets import load_iris
x_data = datasets.load_iris().data		#读入iris数据集的所有输入特征
y_data = datasets.load_iris().target	#读入iris数据集所有标签
```
#### 数据预处理
  - 数据集乱序：随机打乱数据
```python
# seed: 随机数种子，是一个整数，当设置之后，每次生征和标签一一对应
np.random.seed(116)
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)
tf.random.set_seed(116)
```
  - 将数据集分成训练集和测试集

```python
# 训练集为前120行，测试集为后30行
x_train = x_data[:-30]
y_train = y_data[:-30]
x_test = x_data[-30:]
y_test = y_data[-30:]
```
  - 输入特征和标签值一一对应，把数据集分批次，每个批次batch(32)组数据
```python
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
```
#### 数据训练
  - 定义神经网络中所有参数可训练
```python
w1 = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1, seed=1))
b1 = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=1))
```
  - **<font color=red>嵌套循环迭代，with结构更新参数，显示当前loss</font>**
```python
for epoch in range(epoch):  # 数据集级别迭代
    for step, (x_train, y_train) in enumerate(train_db):  # batch级别的迭代
        with tf.GradientTape() as tape:  # 记录梯度信息
        	# 前向传播过程计算y
        	# 计算总loss
        grads = tape.gradient(loss, [w1, b1])	# 求导
        w1.assign_sub(lr * grads[0])  # 参数w1自更新
        b1.assign_sub(lr * grads[1])  # 参数b自更新
    # 每个epoch，打印loss信息
    print("Epoch {}, loss: {}".format(epoch, loss_all/4))
```
  - 计算当前参数前向传播后的准确率，显示当前acc（accuracy）

```python
    for x_test, y_test in test_db:
        y = tf.matmul(x_test, w1) + b1	# y为预测结果
        y = tf.nn.softmax(y)	# y符合概率分布
        pred = tf.argmax(y, axis=1)  # 返回y中最大值的索引，即预测的分类
        pred = tf.cast(pred, dtype=y_test.dtype)	# 调整参数类型与标签一致
        correct = tf.cast(tf.equal(pred, y_test), dtype=tf.int32)
        correct = tf.reduce_sum(correct)	# 将所有batch中的correct数加起来
        total_correct += int(correct)	# 将所有batch中的correct数加起来
        total_number += x_test.shape[0]
    acc = total_correct / total_number
    test_acc.append(acc)
    print("Test_acc:", acc)
```
#### 数据可视化
  - loss可视化

```python
plt.title('Loss Function Curve')  # 图片标题
plt.xlabel('Epoch')  # x轴变量名称
plt.ylabel('Loss')  # y轴变量名称
plt.plot(train_loss_results, label="$Loss$")  # 逐点画出trian_loss_results值并连线，连线图标是Loss
plt.legend()  # 画出曲线图标
plt.show()  # 画出图像
```

  - acc可视化

```python
plt.title('Acc Curve')  # 图片标题
plt.xlabel('Epoch')  # x轴变量名称
plt.ylabel('Acc')  # y轴变量名称
plt.plot(test_acc, label="$Accuracy$")  # 逐点画出test_acc值并连线，连线图标是Accuracy
plt.legend()
plt.show()
```

### 3.完整源码

```python
# 导入所需模块
import tensorflow as tf
from sklearn import datasets
from matplotlib import pyplot as plt
import numpy as np

# 导入数据，分别为输入特征和标签
x_data = datasets.load_iris().data
y_data = datasets.load_iris().target

# 随机打乱数据（因为原始数据是顺序的，顺序不打乱会成的随机数都一样
np.random.seed(116)  # 使用相同的seed，保证输入特影响准确率
# seed: 随机数种子，是一个整数，当设置之后，每次生征和标签一一对应
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)
tf.random.set_seed(116)

# 将打乱后的数据集分割为训练集和测试集，训练集为前120行，测试集为后30行
x_train = x_data[:-30]
y_train = y_data[:-30]
x_test = x_data[-30:]
y_test = y_data[-30:]

# 转换x的数据类型，否则后面矩阵相乘时会因数据类型不一致报错
x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)

# from_tensor_slices函数使输入特征和标签值一一对应。（把数据集分批次，每个批次batch组数据）
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# 生成神经网络的参数，4个输入特征故，输入层为4个输入节点；因为3分类，故输出层为3个神经元
# 用tf.Variable()标记参数可训练
# 使用seed使每次生成的随机数相同
w1 = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1, seed=1))
b1 = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=1))

lr = 0.1  # 学习率为0.1
train_loss_results = []  # 将每轮的loss记录在此列表中，为后续画loss曲线提供数据
test_acc = []  # 将每轮的acc记录在此列表中，为后续画acc曲线提供数据
epoch = 500  # 循环500轮
loss_all = 0  # 每轮分4个step，loss_all记录四个step生成的4个loss的和

# 训练部分
for epoch in range(epoch):  #数据集级别的循环，每个epoch循环一次数据集
    for step, (x_train, y_train) in enumerate(train_db):  #batch级别的循环 ，每个step循环一个batch
        with tf.GradientTape() as tape:  # with结构记录梯度信息
            y = tf.matmul(x_train, w1) + b1  # 神经网络乘加运算
            y = tf.nn.softmax(y)  # 使输出y符合概率分布（此操作后与独热码同量级，可相减求loss）
            y_ = tf.one_hot(y_train, depth=3)  # 将标签值转换为独热码格式，方便计算loss和accuracy
            loss = tf.reduce_mean(tf.square(y_ - y))  # 采用均方误差损失函数mse = mean(sum(y-out)^2)
            loss_all += loss.numpy()  # 将每个step计算出的loss累加，为后续求loss平均值提供数据，这样计算的loss更准确
        # 计算loss对各个参数的梯度
        grads = tape.gradient(loss, [w1, b1])

        # 实现梯度更新 w1 = w1 - lr * w1_grad    b = b - lr * b_grad
        w1.assign_sub(lr * grads[0])  # 参数w1自更新
        b1.assign_sub(lr * grads[1])  # 参数b自更新

    # 每个epoch，打印loss信息
    print("Epoch {}, loss: {}".format(epoch, loss_all/4))
    train_loss_results.append(loss_all / 4)  # 将4个step的loss求平均记录在此变量中
    loss_all = 0  # loss_all归零，为记录下一个epoch的loss做准备

    # 测试部分
    # total_correct为预测对的样本个数, total_number为测试的总样本数，将这两个变量都初始化为0
    total_correct, total_number = 0, 0
    for x_test, y_test in test_db:
        # 使用更新后的参数进行预测
        y = tf.matmul(x_test, w1) + b1
        y = tf.nn.softmax(y)
        pred = tf.argmax(y, axis=1)  # 返回y中最大值的索引，即预测的分类
        # 将pred转换为y_test的数据类型
        pred = tf.cast(pred, dtype=y_test.dtype)
        # 若分类正确，则correct=1，否则为0，将bool型的结果转换为int型
        correct = tf.cast(tf.equal(pred, y_test), dtype=tf.int32)
        # 将每个batch的correct数加起来
        correct = tf.reduce_sum(correct)
        # 将所有batch中的correct数加起来
        total_correct += int(correct)
        # total_number为测试的总样本数，也就是x_test的行数，shape[0]返回变量的行数
        total_number += x_test.shape[0]
    # 总的准确率等于total_correct/total_number
    acc = total_correct / total_number
    test_acc.append(acc)
    print("Test_acc:", acc)
    print("--------------------------")

# 绘制 loss 曲线
plt.title('Loss Function Curve')  # 图片标题
plt.xlabel('Epoch')  # x轴变量名称
plt.ylabel('Loss')  # y轴变量名称
plt.plot(train_loss_results, label="$Loss$")  # 逐点画出trian_loss_results值并连线，连线图标是Loss
plt.legend()  # 画出曲线图标
plt.show()  # 画出图像

# 绘制 Accuracy 曲线
plt.title('Acc Curve')  # 图片标题
plt.xlabel('Epoch')  # x轴变量名称
plt.ylabel('Acc')  # y轴变量名称
plt.plot(test_acc, label="$Accuracy$")  # 逐点画出test_acc值并连线，连线图标是Accuracy
plt.legend()
plt.show()

```
**<font size=5>[博客园链接](https://www.cnblogs.com/moonspace/p/12879489.html)</font>**

