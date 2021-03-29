# TensorFlow2.1入门学习笔记(12)——卷积神经网络


每个神经元与前后相邻层的每一个神经元都有连接关系，输入是特征，输出为预测的结果。随着隐藏层增多，网络规模的增大，待优化参数过多容易导致模型过拟合

## 卷积计算过程：
全连接NN：每个神经元与前后相邻层的每一个神经元都有连接关系，输入是特征，输出为预测的结果。

![](https://img-blog.csdnimg.cn/20200612195038324.png " ")

实际项目中的图片多是高分辨率彩色图
	
![](https://img-blog.csdnimg.cn/20200612195316182.png " ")

随着隐藏层增多，网络规模的增大，待优化参数过多容易导致模型过拟合

实际应用时会先对原始图像进行特征提取再把提取到的特征送给全连接网络

![](https://img-blog.csdnimg.cn/20200612195743308.png " ")

### 卷积（Convolutional）
- 卷积计算可是一种有效提取图像特征的方法
- 一般会用一个正方形的卷积核，按指定步长，在输入特征图上滑动，遍历输入特征图中的每个像素点。每一个步长，卷积核会与输入特征图出现重合区域，重合区域对应元素相乘、求和再加上偏置项得到输出特征的一个像素点
- 输入特征图的深度（channel数），决定了当前层卷积核的深度；当前层卷积核的个数，决定了当前层输出特征图的深度。
- 卷积核

	![](https://img-blog.csdnimg.cn/20200612222105316.png " ")
	![](https://img-blog.csdnimg.cn/20200612222800539.png " ")

- 卷积核的计算过程

	![](https://img-blog.csdnimg.cn/20200612223035455.gif " ")

## 感受野（Receptive Field）
卷积神经网络各输出特征图中的每个像素点，在原始输入图片上映射区域的大小。

例如：5x5x1的输入特征，经过2次3x3x1的卷积过程感受野是5；经过1次5x5x1的卷积过程感受野也是5，感受野相同，则特征提取能力相同。

![](https://img-blog.csdnimg.cn/20200612223641326.png " ")

- 感受野的选择

	![](https://img-blog.csdnimg.cn/20200612223955243.png " ")

当输入特征图边长大于10像素点时，两层3x3的卷积核比一层5x5的卷积性能要好，因此在神经网络卷积计算过程中常采用两层3x3的卷积代替已成5x5的卷积。

## 全零填充（Padding）

当需要卷积计算保持输入特征图的尺寸不变则使用全零填充，在输入特征的周围用零填充

- 在5x5x1的输入特征图经过全零填充后，在经过3x3x1的卷积核，进行步长为1的卷积计算，输出特征图仍是5x5x1

![](https://img-blog.csdnimg.cn/20200612233752911.png " ")

- 输出特征图维度的计算公式

<div>
$$
padding = \left\{
  \begin{array}{lr}
    SAME(全0填充)&\frac{入长}{步长}	(向上取整)\\
    VALID(不全零填充)&\frac{入长-核长+1}{步长}	(向上取整)
  \end{array}
\right.
$$
</div>

- TenaorFlow描述全零填充
	用参数padding = ‘SAME’ 或 padding = ‘VALID’表示
	![](https://img-blog.csdnimg.cn/2020061223504981.png " ")
## TF描述卷积层
```python
tf.keras.layers.Conv2D (
	filters = 卷积核个数,
	kernel_size = 卷积核尺寸, 			#正方形写核长整数，或（核高h，核宽w）
	strides = 滑动步长,					#横纵向相同写步长整数，或(纵向步长h，横向步长w)，默认1
	padding = “same” or “valid”, 		#使用全零填充是“same”，不使用是“valid”（默认）
	activation = “ relu ” or “ sigmoid ” or “ tanh ” or “ softmax”等 , 		#如有BN此处不写
	input_shape = (高, 宽 , 通道数)		#输入特征图维度，可省略
)
```

## 批标准化（BN）
神经网络对0附近的数据更敏感，单随网络层数的增加特征数据会出现偏离0均值的情况

- 标准化：使数据符合0均值，1为标准差的分布。
- 批标准化：对一小批数据（batch），做标准化处理。

标准化可以是数据重新回归到标准正态分布常用在卷积操作和激活操作之间

![](https://img-blog.csdnimg.cn/2020061300060593.png " ")

批标准化操作将原本偏移的特征数据重新拉回到0均值，使进入到激活函数的数据分布在激活函数线性区使得输入数据的微小变化更明显的提现到激活函数的输出，**提升了激活函数对输入数据的区分力**。但是这种简单的特征数据标准化使特征数据完全满足标准正态分布。集中在激活函数中心的线性区域，**使激活函数丧失了非线性特性**。因此在BN操作中为每个卷积核引入了两个可训练参数，**缩放因子$\gamma$和偏移因子$\beta$**。反向传播时缩放因子$\gamma$和偏移因子$\beta$会与其他带训练参数一同被训练优化，使标准状态分布后的特征数据。通过缩放因子和偏移因子优化了特征数据分布的宽窄和偏移量。保证了网络的非线性表的力。

![](https://img-blog.csdnimg.cn/2020061300234954.png " ")

- BN位于卷积层之后，激活层之前
- TensorFlow描述批标准化
	tf.keras.layers.BatchNormalization()


## 池化（Pooling）
池化用于减少特征数据量。最大值池化可提取图片纹理，均值池化可保留背景特征。

![](https://img-blog.csdnimg.cn/20200613003846171.png " ")

- TensorFlow描述池化

```python
tf.keras.layers.MaxPool2D(
	pool_size=池化核尺寸，	#正方形写核长整数，或（核高h，核宽w）
	strides=池化步长，		#步长整数， 或(纵向步长h，横向步长w)，默认为pool_size
	padding=‘valid’or‘same’ #使用全零填充是“same”，不使用是“valid”（默认）
)
tf.keras.layers.AveragePooling2D(
	pool_size=池化核尺寸，	#正方形写核长整数，或（核高h，核宽w）
	strides=池化步长，		#步长整数， 或(纵向步长h，横向步长w)，默认为pool_size
	padding=‘valid’or‘same’ #使用全零填充是“same”，不使用是“valid”（默认）
)
```

## 舍弃(Dropout)
为了缓解神经网络过拟合，在神经网络训练时，将隐藏层的部分神经元按照一定概率从神经网络中暂时舍弃。神经网络使用时，被舍弃的神经元恢复链接。

![](https://img-blog.csdnimg.cn/20200613004956902.png " ")

- TensorFlow描述舍弃
	tf.keras.layers.Dropout(舍弃的概率)




## 卷积神经网络
借助卷积核提取特征后，送入全连接网络。

卷积神经网络的主要模块：

- 卷积（Convolutional）
- 批标准化（BN）
- 激活（Activation）
- 池化（Pooling）
- 舍弃（Dropout）
- 全连接（FC）

```python
model = tf.keras.models.Sequential([
	Conv2D(filters=6, kernel_size=(5, 5), padding='same'),	#卷积层
	BatchNormalization(),									#BN层	
	Activation('relu'),										#激活层
	MaxPool2D(pool_size=(2, 2), strides=2, padding='same'),	#池化层
	Dropout(0.2),											#dropout层
])
```

