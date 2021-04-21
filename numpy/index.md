# TensorFlow2.1入门学习笔记(1)——Numpy科学计算库

在正式学习tensorflow2.0之前需要有一定的python基础，对numpy，matplotlib等库有基本的了解，笔者还是AI小白，通过写博客来记录自己的学习过程，同时对所学的东西进行总结。主要学习资料西安科技大学：[神经网络与深度学习——TensorFlow2.0实战](https://www.icourse163.org/learn/XUST-1206363802#/learn/announce)，北京大学：[人工智能实践Tensorflow笔记](https://www.icourse163.org/learn/PKU-1002536002#/learn/announce)。博客从tf常用的库开始，需要学习python基础的朋友推荐[菜鸟教程](https://www.runoob.com/python3/python3-tutorial.html)
<!-- more -->
### 1.多维数组的形状(Shape)
描述数组的维度，以及各维度内部元素个数

#### 一维数组 shape:(5,)
描述某位同学5门课程的成绩：

![一维数组](https://img-blog.csdnimg.cn/20200429113836745.png " ")

#### 二维数组 shape:(30,5)
描述某个班30位同学5门课成绩：

![二维数组](https://img-blog.csdnimg.cn/20200429113943373.png " ")
#### 三维数组 shape:(10,30,5)
描述某个学校10个班30位同学5门课成绩：

![三维数组](https://img-blog.csdnimg.cn/20200429115427228.png " ")
#### 四维数组 shape:(5,10,30,5)
描述某个地区5所学校10个班30位同学5门课成绩：

![四维数组](https://img-blog.csdnimg.cn/20200429120718407.png " ")
#### 五维数组 shape:(4,5,10,30,5)
描述某个某个国家4个地区5所学校10个班30位同学5门课成绩：

![五维数组](https://img-blog.csdnimg.cn/20200429120814872.png " ")
更高维以此类推
### 2.创建Nump
#### 安装Numpy库

```python
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy
```
#### 导入Numpy库

```python
import numpy as np 
import numpy import *		#可直接调用库，但不推荐，容易和其他包冲突
```

#### 创建数组

```python
m = np.array([[[4, 5, 8, 3],[3, 6, 9, 0],[8, 4, 5, 6]],
				[[4, 5, 8, 3],[3, 6, 9, 0],[8, 4, 2, 1]]])
# 数组属性
m.ndim				#3 维度
m.shape				#(2,3,4) 形状
m.size				#24	元素的总个数
m.dtype				#int32 数据类型
m.itemsize			#4 每个元素的字节数
```

创建特殊的数组

![特殊数组](https://img-blog.csdnimg.cn/20200429133042175.png " ")

```python
# np.arrange(start=0,stop,num=1,dtype) 前闭后开，不包含结束值
n=np.arange(4)					#array([0, 1, 2, 3])
a=np.arange(0,2,0.3)			#array([0., 0.3, 0.6, 0.9, 1.2, 1.5, 1.8])
np.ones((3,2),dtype =np.int64)	#array([[1,1],[1, 1],[1, 1]],dtype = int64)
np.eye(2, 3)					#array([[1., 0., 0.], [0., 1., 0.]])创建一个单位矩阵
# np.logspace(stat,stop,num,base,dtype)参数：起始指数，结束指数，基，元素数据类型，包含结束值
np.logspace(1, 5, 5, base=2)	#array([2., 4., 8, 16, 32])
```

![](https://img-blog.csdnimg.cn/20200429135208645.png " ")

### 3.数组计算
需要了解几个常见的数组数据处理函数

```python
# 数组元素切片
a = np.array([0,1,2,3])		#一维数组
print(a[:3])				#array([0,1,2]) 输出前三个数
b = np.array([[0,1,2,3],[3,4,5,6],[6,7,8,9]])	#二维数组
print(b[:2])				#array([[0,1,2,3],[3,4,5,6]]) 输出前两行
# 改变数组的形状
c = np.arange(12)
d = c.reshape(3,4)			
print(d)					#array([[0,1,2,3],[4,5,6,7],[8,9,10,11]]) 不改变当前数组，按照shape创建新的数组
c.reshape(-1,1)
print(c)					#array([[0],[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11]])
c.resize(3,4)			
print(c)					#array([[0,1,2,3],[4,5,6,7],[8,9,10,11]]) 改变当前数组，按照shape创建新的数组
```

#### 数组间的运算
1.数组间的元素运算

![](https://img-blog.csdnimg.cn/20200502182836982.png " ")

```python
a = np.arange(4)
print(np.sum(a))		#6
print(np.sqrt(a))		#array([0.         ,1.         ,1.41421356, 1.73205081])
```

数组的轴和秩

![](https://img-blog.csdnimg.cn/20200502183402824.png " ")

数组的堆叠运算

![](https://img-blog.csdnimg.cn/20200502185456990.png " ")

```python
x = np.array([1,2,3])
y = np.array([4,5,6])
print(np.stack((x,y),axis = 0)) #array([1,2,3],[4,5,6])
print(np.stack((x,y),axis = 1))	#array([1,4],[2,5],[3,6])
```

```python
a = np.arrange(12).reshape(3,4)	#a = ([0,1,2,3],[4,5,6,7],[8,9,10,11])
print(np.sum(b,axis=0))		#array([12,15,18,21])
print(np.sum(b,axis=1))		#array([6,22,38])
```

2.数组加减法，对应元素相加减（进行运算的数组长度要一致）

```python
a = np.ones([3,3])
b = np.eye(3,3)
print(a+b)			#array([[2，1,1],[1,2,1],[1,1,2]])
```

3.一维数组可以和多维数组相加，相加时将会将一维数组扩展至多维

```python
a = np.array([1,2,3])
b = np.array([1,1,1],[2,2,2])
print(a+b)		#array([2,3,4],[3,4,5])
print(b**2)		#array([1,1,1],[4,4,4])
```

SUMMARIZE:数组间的四则运算，是对应元素加减乘除；
					   当数组中元素的数据类型不同时，精度低的数据类型会转换成精度高的数据类型，然后再运算
					   
#### 矩阵运算
1.矩阵乘法，按矩阵相乘的规则运算

```python
A = np.array([[1,2],[2,3]])
B = np.array([[4,2,1],[1,5,2]])
print(np.matmul(A,B))		#array([6,12,5],[11,19,8])
```

2.转置和求逆

```python
#转置
print(np.transpose(A))		#array([1,2],[2,3])
#求逆
print(np.linalg.inv(A))		#array([-3,2],[2,-1])
```
**<font size=5>[博客园链接](https://www.cnblogs.com/moonspace/p/12826438.html)</font>**


