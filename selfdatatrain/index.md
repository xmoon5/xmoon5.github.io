# TensorFlow2.1入门学习笔记(11)——自制数据集，并记录训练模型



以MNIST的[sequential模型](https://blog.aimoon.top/2020/05/tf_keras_mnist/#%E4%BD%BF%E7%94%A8sequential%E5%AE%9E%E7%8E%B0%E6%89%8B%E5%86%99%E5%AD%97%E4%BD%93%E8%AF%86%E5%88%AB)为base-line，通过读取自己的数据，训练模型并存储模型，最后达到绘图实物的运用。


## 自制数据集，解决本领域应用
### 观察数据结构
给x_train、y_train、x_test、y_test赋值


![](https://img-blog.csdnimg.cn/20200604000506672.png)

![](https://img-blog.csdnimg.cn/20200604212835629.png " ")

### def generateds(图片路径,标签文件)：

```python
def generateds(path, txt):
    f = open(txt, 'r')  # 以只读形式打开txt文件
    contents = f.readlines()  # 读取文件中所有行
    f.close()  # 关闭txt文件
    x, y_ = [], []  # 建立空列表
    for content in contents:  # 逐行取出
        value = content.split()  # 以空格分开，图片路径为value[0] , 标签为value[1] , 存入列表
        img_path = path + value[0]  # 拼出图片路径和文件名
        img = Image.open(img_path)  # 读入图片
        img = np.array(img.convert('L'))  # 图片变为8位宽灰度值的np.array格式
        img = img / 255.  # 数据归一化 （实现预处理）
        x.append(img)  # 归一化后的数据，贴到列表x
        y_.append(value[1])  # 标签贴到列表y_
        print('loading : ' + content)  # 打印状态提示

    x = np.array(x)  # 变为np.array格式
    y_ = np.array(y_)  # 变为np.array格式
    y_ = y_.astype(np.int64)  # 变为64位整型
    return x, y_  # 返回输入特征x，返回标签y_
```



## 数据增强，扩充数据集
数据增强（增大数据量），可以简单的扩展数据集，对图像的数据增强就是对图像的简单形变。

tensorflow2中的数据增强函数

```python
image_gen_train = tf.keras.preprocessing.image.ImageDataGenerator(
	rescale = 所有数据将乘以该数值
	rotation_range = 随机旋转角度数范围
	width_shift_range = 随机宽度偏移量
	height_shift_range = 随机高度偏移量
	水平翻转：horizontal_flip = 是否随机水平翻转
	随机缩放：zoom_range = 随机缩放的范围 [1-n，1+n] )
image_gen_train.fit(x_train)

### 例 ###
image_gen_train = ImageDataGenerator(
	rescale=1. / 1., # 如为图像，分母为255时，可归至0～1
	rotation_range=45, # 随机45度旋转
	width_shift_range=.15, # 宽度偏移
	height_shift_range=.15, # 高度偏移
	horizontal_flip=False, # 水平翻转
	zoom_range=0.5 # 将图像随机缩放阈量50％)
image_gen_train.fit(x_train)
```
其中image_gen_train.fit(x_train)中的fit需要一个四维数组
即：

```python
x_train = x_train.reshape(x_train[0], 28, 28, 1)
```
(60000, 28, 28) $\Rightarrow$ (60000, 28, 28, 1)
将60000张28行28列的数据转换成60000张28行28列单通道的数据集，其中“1”是灰度值
model.fit()同步更新为model.fit(image_gen_train.flow(x_train, y_train,batch_size=32), ……)

<center>model.fit(x_train, y_train,batch_size=32, ……)</center>

$$\Downarrow$$

<center>model.fit(image_gen_train.flow(x_train, y_train,batch_size=32), ……)</center>


加入数据增强的的代码训练后

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)  # 给数据增加一个维度,从(60000, 28, 28)reshape为(60000, 28, 28, 1)

image_gen_train = ImageDataGenerator(
    rescale=1. / 1.,  # 如为图像，分母为255时，可归至0～1
    rotation_range=45,  # 随机45度旋转
    width_shift_range=.15,  # 宽度偏移
    height_shift_range=.15,  # 高度偏移
    horizontal_flip=False,  # 水平翻转
    zoom_range=0.5  # 将图像随机缩放阈量50％
)
image_gen_train.fit(x_train)

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

model.fit(image_gen_train.flow(x_train, y_train, batch_size=32), epochs=5, validation_data=(x_test, y_test),
          validation_freq=1)
model.summary()

```

![](https://img-blog.csdnimg.cn/20200604231346403.png " ")

- 随着模型迭代轮数的增加，模型的准确率不断提高
- 数据在小数据量上可以增加模型的泛化性


## 断点续训，存取模型
### 读取保存模型
load_weights(路径文件名)


```python
checkpoint_save_path = "./checkpoint/fashion.ckpt"	#先定义出存放模型的路径和文件名，“.ckpt”文件在生成时会同步生成索引表
if os.path.exists(checkpoint_save_path + '.index'):		#判断是否有索引表，就可以知道是否报存过模型，如果有索引表，就会调用load_weights()即模型
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)
```


### 保存模型
使用tensorflow给出的回调函数直接保存训练的参数：

tf.keras.callbacks.ModelCheckpoint(filepath=路径文件名,save_weights_only=True/False,save_best_only=True/False)

history = model.fit（ callbacks=[cp_callback] ）

```python
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,		# 文件存储路径
                                                 save_weights_only=True,			# 是否只保留模型参数
                                                 save_best_only=True)				# 是否只保留模型最优参数

history = model.fit(x_train, y_train, batch_size=32, epochs=5, 						# 加入callbacks选项，记录到history中
					validation_data=(x_test, y_test), validation_freq=1,
                    callbacks=[cp_callback])
```

加入断点续训的完整代码：

```python
import tensorflow as tf
import os		# 引入os模块，（文件处理）

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

checkpoint_save_path = "./checkpoint/fashion.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)

history = model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), validation_freq=1,
                    callbacks=[cp_callback])
model.summary()

```
第一次运行：

![](https://img-blog.csdnimg.cn/20200604233937210.png " ")
![](https://img-blog.csdnimg.cn/20200604234150899.png " ")

第二次运行：

![](https://img-blog.csdnimg.cn/20200604234607210.png " ")
![](https://img-blog.csdnimg.cn/20200604234949472.png " ")

加载了第一次保存的参数，准确率在第一次的基础上提高



## 参数提取，把参数存入文本
- 提取可训练参数
	model.trainable_variables 返回模型中可训练的参数
- 设置print输出格式
	np.set_printoptions(threshold=超过多少省略显示)
```python
np.set_printoptions(threshold=np.inf) # np.inf表示无限大
```
- 将可训练参数存入文本

```python
print(model.trainable_variables)
file = open('./weights.txt', 'w')
for v in model.trainable_variables:
	file.write(str(v.name) + '\n')
	file.write(str(v.shape) + '\n')
	file.write(str(v.numpy()) + '\n')
file.close()
```

在断点续训的基础上加入参数提取

```python
import tensorflow as tf
import os
import numpy as np

np.set_printoptions(threshold=np.inf)

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

checkpoint_save_path = "./checkpoint/fashion.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)

history = model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), validation_freq=1,
                    callbacks=[cp_callback])
model.summary()

print(model.trainable_variables)
file = open('./weights.txt', 'w')
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

```

运行得到weights.txt文件

![](https://img-blog.csdnimg.cn/20200605000228419.png " ")



## acc/loss可视化，查看训练效果

- 将训练过程可视化出来
	在history中同步记录了训练集loss、测试机loss、训练集准确率和测试集准确率
- history=model.fit(训练集数据, 训练集标签, batch_size=, epochs=,validation_split=用作测试数据的比例,validation_data=测试集,validation_freq=测试频率)
- history
	训练集loss： loss
	测试集loss： val_loss
	训练集准确率： sparse_categorical_accuracy
	测试集准确率： val_sparse_categorical_accuracy
	
```python
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']	#使用history.histor[]提取
```

加入绘制图像的代码：

```python
import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt		# 导入绘图模块

np.set_printoptions(threshold=np.inf)

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

checkpoint_save_path = "./checkpoint/fashion.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)

history = model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), validation_freq=1,
                    callbacks=[cp_callback])
model.summary()

print(model.trainable_variables)
file = open('./weights.txt', 'w')
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

###############################################    show   ###############################################

# 显示训练集和验证集的acc和loss曲线
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

```

可视化结果：

![](https://img-blog.csdnimg.cn/2020060500162050.png " ")

## 应用程序，给图识物
前面已经将模型训练好了，下面将编写一套运用程序实现给图识物

![](https://img-blog.csdnimg.cn/2020060500240012.png " ")

- predict(输入特征，batch_size=整数)
	返回前向传播计算结果

前向传播执行应用：
1. 复现模型（前向传播）
2. 加载参数：model.load_weights(model_save_path)
3. 预测结果：result = model.predict(x_predict)

源码：

```python
from PIL import Image
import numpy as np
import tensorflow as tf

model_save_path = './checkpoint/mnist.ckpt'		

model = tf.keras.models.Sequential([						# 复现网络
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')])
    
model.load_weights(model_save_path) 						# 加载参数

preNum = int(input("input the number of test pictures:"))	# 准备预测多少个数

for i in range(preNum):										# 读入待识别的图片
    image_path = input("the path of test picture:")
    img = Image.open(image_path)							
    img = img.resize((28, 28), Image.ANTIALIAS)				# 转换成（28，28）的类型，与训练数据类型匹配
    img_arr = np.array(img.convert('L'))					# 转换成灰度图

    img_arr = 255 - img_arr									# 将“白底黑字”反转成“黑底白字”

	#####or#####
#	for i in range(28):							# 转换成高对比度的图，过滤噪声
#		for j in range(28):
#			if img_arr[i][j] < 200:
#				img_arr[i][j] = 255
#            else:
#            	img_arr[i][j] = 0

   
    img_arr = img_arr / 255.0								# 归一化
    print("img_arr:",img_arr.shape)
    x_predict = img_arr[tf.newaxis, ...]					# 由于是按每个batch送入网络，故添加一个维度
    print("x_predict:",x_predict.shape)
    result = model.predict(x_predict)						#预测结果
    
    pred = tf.argmax(result, axis=1)
    
    print('\n')
    tf.print(pred)

```


