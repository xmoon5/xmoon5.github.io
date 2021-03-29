# TensorFlow2.1入门学习笔记(3)——Pillow数字图像处理


在正式学习tensorflow2.0之前需要有一定的python基础，对numpy，matplotlib等库有基本的了解，笔者还是AI小白，通过写博客来记录自己的学习过程，同时对所学的东西进行总结。主要学习的资料西安科技大学：[神经网络与深度学习——TensorFlow2.0实战](https://www.icourse163.org/learn/XUST-1206363802#/learn/announce)，北京大学：[人工智能实践Tensorflow笔记](https://www.icourse163.org/learn/PKU-1002536002#/learn/announce)博客从tf常用的库开始，需要学习python基础的朋友推荐[菜鸟教程](https://www.runoob.com/python3/python3-tutorial.html)
<!--more-->
### 1.数字图像的基本概念
在处理数据过程中绝大多数的数据来自图像，图像数据处理是人工智能的重要组成
**图像的离散化**
连续图像：人眼能直接感受到的图像
数字图像：把连续图像数字化、离散化之后的图像，他是对连续图像的一种近似
**像素（Pixel）：**图像中的最小单位
**位图（bitmap）**：通过记录每一个像素值来存储和表达的图像
**色彩深度/位深度**：位图中的每个像素点要用多少个二进制来表示
例如：位深度为24表示每个像素点用24个二进制位来表示（通常RGB三个字节，每个字节8位）

![](https://img-blog.csdnimg.cn/20200506213918319.png " ")

**EMP**：Windows系统的标准位图格式

**二值图像（Binary Image）**
每个像素只有2种可能的取值，用1位二进制来表示，位深度为1

![](https://img-blog.csdnimg.cn/2020050621453166.png " ")

黑白图像：只有黑色和白色两种颜色，在图像处理和分析时，通常先对图像二值化处理 

![](https://img-blog.csdnimg.cn/2020050621493753.png " ")

**PS：**只要仅有两种颜色的图像，都可以被称为二值图像，区分于灰度图

![](https://img-blog.csdnimg.cn/20200506215046530.png " ")

**灰度图像（Gray Image）**
每个像素使用一个字节表示，位深度为8，可以表示256种级别的灰度，0表示黑色，255表示白色
例：存储512x512的灰度图像，512x512x8bit=256KB

![](https://img-blog.csdnimg.cn/20200506215451824.png " ")

**彩色图像**
每个像素都有红(R),绿(G),蓝(B)三个分量；一个像素使用三个字节，位深度为24位；可以表示256^3种颜色

![R			G			B			颜色
0			0			0			黑色
255		255		255		白色
255		0			0			红色](https://img-blog.csdnimg.cn/20200506220211564.png)


RGB为24位真彩色

![](https://img-blog.csdnimg.cn/20200506220248489.png " ")

**RGBA图像——32位真彩色**
RGB图像+8位透明度信息Alpha，1一个像素使用4个字节，位深度为32位

![](https://img-blog.csdnimg.cn/20200506220527434.png " ")

**256色彩色图像**
对每个像素使用8位二进制表示，是彩色调色板中的索引值；对于不同的图像，所对应的256种颜色的集合是不一样的；在保存和加载这种类型的位图时，需要将调色板和图像一同保存和加载

![](https://img-blog.csdnimg.cn/20200506221249424.png " ")

**图像的压缩**
适当降低图像的质量来减它所占有的空间；不同的压缩算法对应不同的图像格式
**图像格式**
BMP格式：占用空间大，不支持文件压缩，不适用于网页
JPEG格式：有损压缩，压缩效率高，所占空间小
适合于色彩丰富，细节清晰细腻的大图

![](https://img-blog.csdnimg.cn/20200506221939776.png " ")

不适合所含颜色较少，具有大块颜色相近的区域或亮度差异十分明显的简单照片

![](https://img-blog.csdnimg.cn/20200506222154244.png " ")

PNG格式：无损压缩；适合有规律的渐变色彩的图像，广泛运用于网络

![](https://img-blog.csdnimg.cn/20200506222443823.png " ")

GIF格式：支持静态格式和动态格式；动态图片由多幅图片保存为一个图片，循环显示，形成动画效果；只支持256色，适用于色彩简单。颜色较少的小图像
**色彩模式**
二值图像、灰度图像、RGB图像、RGBA图像
CMYK——印刷四分色：C（cyan=青色）、M（magenta=洋红色）、Y（yellow=黄色）、K（black=黑色）
YCbCr——Y（亮度）、Cb（蓝色色度）、Cr（红色色度）
HSI——H（色调）、S（饱和度）、I（亮度）
图像类型
序列图像：时间上有一定顺序和间隔、内容上相关的一组图像，其中每幅图像称为帧图像，帧图像之间的时间间隔是固定的

![](https://img-blog.csdnimg.cn/20200506223625438.png " ")

深度图像：是一种三维场景信息的表达式；每个像素的取值代表这个点离相机的距离；采用灰度图表示，每个像素点由一个字节表示；像素点的取值不代表距离，颜色的深浅只代表相对距离的远近

![](https://img-blog.csdnimg.cn/20200506224003295.png " ")

### 2.Pillow图像处理库
**安装和导入包/模块**
Pillow的安装

```python
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pillow
```
Pillow.image的导入

```python
from PIL import Image
```

#### 常用函数
**Image.open()函数：打开图像**
Image.open(路径)：返回值为image对象

![](https://img-blog.csdnimg.cn/20200506232929541.png " ")

**Image.save()函数：保存图像**
图像对象.save(文件名)：保存图像，改变文件名后缀名，可转换图像格式

![](https://img-blog.csdnimg.cn/20200507005134246.png " ")

**图像对象的主要属性**
图像对象.format		图像格式
图像对象.size			图像尺寸
图像对象.mode		色彩模式

```python
from PIL import Image   #导入库
img = Image.open("TF.jpg")    #当前目录下图片名称（路径）
print(img.format)       #JPEG 图像格式
print(img.size)         #(473, 349) 图像尺寸
print(img.mode)         #RGB 色彩模式
```
**imshow()显示图像**
需要使用matplotlib库
plt.imshow(image对象/Numpy数组)

```python
from PIL import Image 
import matplotlib.pyplot as plt 
img = Image.open("TF.jpg")
plt.figure(figsize=(5,5))      #创建画布
plt.imshow(img)                #画图
plt.title(img.format)          #在标题显示图片格式
plt.show()                     #显示
```

![](https://img-blog.csdnimg.cn/20200506235315581.png " ")

**convert()函数——转换图像的色彩模式**
图像对象.convert(色彩模式)

![](https://img-blog.csdnimg.cn/20200506235637975.png " ")

```python
img_gray=img.convert("L")
print(img_gray.mode)
plt.figure(figsize=(5,5))
plt.imshow(img)
plt.show()
```
运行结果

![](https://img-blog.csdnimg.cn/20200507001516772.png " ")

**颜色通道的分离与合并**
通道分离：图像对象.split()
图像合并：Image.merge(色彩模式,图像列表)
```python
from PIL import Image 
import matplotlib.pyplot as plt #导入库
img = Image.open("TF.jpg")      #打开文件
img_r,img_g,img_b = img.split() #通道分离
plt.figure(figsize=(10,10))     #创建画布

plt.subplot(2,2,1)              #创建2x2的子图
plt.axis("off")                 #不显示坐标轴
plt.imshow(img_r,cmap="gray")   #以灰度图显示r通道
plt.title("R",fontsize=20)      #创建子图标题

plt.subplot(2,2,2)
plt.axis("off")
plt.imshow(img_g,cmap="gray")
plt.title("G",fontsize=20)

plt.subplot(2,2,3)
plt.axis("off")
plt.imshow(img_b,cmap="gray")
plt.title("B",fontsize=20)

plt.subplot(2,2,4)
img_rgb=Image.merge("RGB",[img_r,img_g,img_b])  #通道合并
plt.axis("off")
plt.imshow(img_rgb)
plt.title("RGB",fontsize=20)
plt.show()
```

运行结果：

![](https://img-blog.csdnimg.cn/20200507003138713.png " ")

**转为数组**
np.array(图像对象)

```python
from PIL import Image 
import numpy as np 
import matplotlib.pyplot as plt 
img = Image.open("TF.jpg")
arr_img=np.array(img)
print(arr_img.shape)			#(349, 473, 3) 
print(arr_img)
```

部分结果

![](https://img-blog.csdnimg.cn/20200507004515673.png " ")

**对图像的颜色反向，缩放，旋转和镜像**
颜色反向：255-图像数组（arr_ima_new=255-arr_img）
缩放图像：
图像对象.resize((width,heigth))，不对原图进行修改
图像对象.thumbnail((width,heigth))，直接对图像对象本身进行缩放
旋转，镜像：图像对象.transpose(旋转方式)
旋转方式

![](https://img-blog.csdnimg.cn/2020050710063575.png " ")

实例：

```python
from PIL import Image 
import numpy as np 
import matplotlib.pyplot as plt #导入库

plt.rcParams['font.sans-serif']="SimHei"
img = Image.open("TF.jpg")      #打开文件

plt.figure(figsize=(10,10))     #创建画布

plt.subplot(3,2,1)
plt.axis("off")
plt.imshow(img)
plt.title("原图",fontsize=20)

plt.subplot(3,2,2)
plt.axis("off")
img_arr=np.array(img)
img_arr_new=255-img_arr             #颜色反向处理
plt.imshow(img_arr_new)
plt.title("颜色反向",fontsize=20)

plt.subplot(3,2,3)
plt.axis("off")
plt.imshow(img)
plt.title("原图",fontsize=20)

plt.subplot(3,2,4)
plt.axis("off")
img_flr=img.transpose(Image.FLIP_LEFT_RIGHT)
plt.imshow(img_flr)
plt.title("左右翻转",fontsize=20)

plt.subplot(3,2,5)
plt.axis("off")
img_r90=img.transpose(Image.ROTATE_90)
plt.imshow(img_r90)
plt.title("逆时针旋转90度",fontsize=20)

plt.subplot(3,2,6)
plt.axis("off")
img_tp=img.transpose(Image.TRANSPOSE)
plt.imshow(img_tp)
plt.title("转置",fontsize=20)

plt.show()
```
运行结果：

![](https://img-blog.csdnimg.cn/20200507102842309.png " ")

**裁剪图像**
在图像上的指定位置裁剪出一个矩形区域
图像对象.crop((x0,y0,x1,y1)) ，返回图像对象，(x0,y0)是左上角的像素位置，(x1,y1)是右下角的像素位置

```python
from PIL import Image 
import matplotlib.pyplot as plt 
img = Image.open("TF.jpg")
plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
plt.imshow(img)

plt.subplot(1,2,2)
img_region=img.crop((100,100,300,300))
plt.imshow(img_region)

plt.show()
```

运行结果：

![](https://img-blog.csdnimg.cn/2020050710442963.png " ")

SUMMARIZE:

![](https://img-blog.csdnimg.cn/20200507104657103.png " ")

**<font size=5>[博客园链接](https://www.cnblogs.com/moonspace/p/12841533.html)</font>**

