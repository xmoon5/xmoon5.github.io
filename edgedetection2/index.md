# 区域检测——Blob & SIFT


针对Harris无法拟合尺度问题而提出
目标:独立检测同一图像缩放版本的对应区域
需要通过尺度选择机制来寻找与图像变换协变的特征区域大小

**“当尺度改变时控制每个圆内的内容不变”**

![](https://img-blog.csdnimg.cn/20200721145623774.png " ")

## Laplacian核

具体的算法是在边缘检测中使用的高斯一阶偏导核转换为高斯二阶偏导核

![](https://img-blog.csdnimg.cn/20200721150547677.png " ")

使用Laplacian核与图像进行卷积操作
**边缘：**出现波纹的地方
**尺度信息：**当波纹重叠并出现极值的地方

![](https://img-blog.csdnimg.cn/20200721151228568.png " ")

空间选择:**如果Laplacian的尺度与blob的尺度“匹配”，则Laplacian响应的幅度将在blob的中心达到最大值**

在实际运用的过程中是使用**模板匹配信号**，即不断改变Laplacian的参数$\sigma$取处理后的结果达到峰值时的$\sigma$，随着参数的增大会导致后面的特征消失（高斯偏导的面积公式中的$\sigma$在分母）

![](https://img-blog.csdnimg.cn/20200721152642323.png " ")

为了保持响应不变(尺度不变)，必须将高斯导数乘以$\sigma$
拉普拉斯导数是二阶高斯导数，所以它必须乘以$\sigma^2$

![](https://img-blog.csdnimg.cn/20200721153839169.png " ")

## 二维空间的Blob的检测

高斯的拉普拉斯算子:用于二维检测的圆对称算子

$$\nabla^2 g=\frac{\partial^2 g}{\partial x^2}+\frac{\partial^2 g}{\partial y^2}\Longrightarrow \nabla_{norm}^2 g=\sigma^2(\frac{\partial^2 g}{\partial x^2}+\frac{\partial^2 g}{\partial y^2})$$

![](https://img-blog.csdnimg.cn/20200721154210200.png " ")

**Laplcain算子中的$\sigma$与检测对象画出的圆的半径$r$的关系**

为了得到最大响应，Laplacian的零点必须**与圆对齐**

令:$$\nabla_{norm}^2 g=0即：\sigma^2(\frac{\partial^2 g}{\partial x^2}+\frac{\partial^2 g}{\partial y^2})=0$$
化简后：
$$
(x^2+y^2-2\sigma^2)e^{-\frac{x^2+y^2}{2\sigma^2}}=0
$$

$$
\Downarrow
$$

$$
x^2+y^2-2\sigma^2=0
$$
得到：<font color=red>$r=\sqrt{2}\sigma$</font>

![](https://img-blog.csdnimg.cn/20200721155411683.png " ")

**特征尺度**

将图像的特征尺度r定义为blob中心产生拉普拉斯响应峰值的尺度

![](https://img-blog.csdnimg.cn/20200721162107675.png " ")

**示例：**

尺度选择过程中将逐步增加参数$\sigma$，每个$\sigma$逐像素计算最大响应，每相邻取九个像素取响应值最大的像素，再与上下两层不同尺度的最大相应取最大（即在一个3x3x3共27个的响应值中取最大的响应值对应的像素点和尺度值）
![](https://img-blog.csdnimg.cn/20200721164517641.png " ")

## SIFT特征

在实际运用过程中，使用Laplacian核可以很好的处理尺度变换的问题，但是需要大量的计算，使用SIFT方法可以简化计算

**DoG模板**
DoG的函数图像与Laplacian核很相似，具有相似的性质，但使用的时两个高斯差分来定义，大的高斯核可以使用小的高斯核来计算，大大减少了计算量

![](https://img-blog.csdnimg.cn/20200721170219616.png " ")


$$G(x,y,k\sigma)-G(x,y,\sigma)\approx(k-1)\sigma^2\nabla^2G$$


1. 高斯空间中的模板利用DoG算法直接从前一层的基础上计算，这样就形成一个DoG空间，得到的模板与与高斯空间相差一个常数项$(k-1)$
2. 计算大尺度的模板时不改变参数值，改变图像大小，例如：将图像缩小一倍，不改变模板尺度得到效果和增大模板尺度不改变图像大小的效果相同，计算四倍尺度的值就将图像缩小四倍，$\sqrt{2}\sigma$的尺度在缩小一倍的图像上的对应尺度为$2\sqrt{2}\sigma$
3. $k=2^{1/s}$：$s$表示要输出的尺度有多少个，利用$s$来计算$k$,例如下图是输出尺度为$s=2$时的示例，此时$k=\sqrt{2}$,二倍尺度状态下的起始模板可以由一倍尺度的$k^2\sigma=2\sigma$下采样得到 
4. 模板尺度通常取$2$的等比数列$(1,2,4,8,16……)$


![](https://img-blog.csdnimg.cn/20200721173645358.png " ")

**SIFT仿射变换**

当视角改变时，即使是同一个圆，其中的内容也有很大的差异

![](https://img-blog.csdnimg.cn/20200722120322624.png " ")

使用[$M$矩阵](https://blog.aimoon.top/2020/07/localfeature/#%E7%9F%A9%E9%98%B5m)将圆具有自适应性，使结果更具鲁棒特性

![](https://img-blog.csdnimg.cn/20200722121115726.png " ")

1. 先确定一个圆
2. 将圆内的所有像素拿出来计算$M$矩阵
3. 比较计算出来的$\lambda_1,\lambda_2$
4. 将较小的$\lambda$的方向进行缩小
5. 再将上一步缩小后的区域（椭圆）内的像素拿出来计算$M$矩阵
6. 重复上述步骤，逐步迭代。直至$\lambda_1,\lambda_2$近似相等，说明区域边缘的梯度变化近似一致

	![](https://img-blog.csdnimg.cn/20200722123152560.png " ")

7. 将椭圆转换到一样大小的圆中

	![](https://img-blog.csdnimg.cn/20200722124752269.png " ")

**梯度方向法**

通过仿射自适应变换后，内容基本一致，但方向不同，对应的像素差异较大，无法识别。

1. 计算圆内每个像素的梯度强度和方向
2. 将梯度方向量化成八份，给对应的直方图投票，票数就是梯度的大小

	![](https://img-blog.csdnimg.cn/20200722125348388.png " ")

3. 统计完之后选择票数最高的方向作为，作为圆内像素整体的梯度方向，将方向转换到$0^\circ$，将整个圆进行相同的旋转


	![](https://img-blog.csdnimg.cn/20200722130018559.png " ")
	
4. **决绝明暗不一致**：将圆均分成16格，每个格代表一个区域，统计每个区域的方向量化梯度（两化成八个角度，长度代表梯度大小），每个区域中由一个“8位”向量表示，将16个区域的向量拉直就得到一个 $8\times16=128$ 的向量来描述这个圆内的内容，最后比较每个圆的128个数来判断两个圆内容的相似程度

	![](https://img-blog.csdnimg.cn/20200722132214374.png " ")

<br>

**总结**：**<font size=5>SIFT算法</font> 可以解决<font color=red>方向，视角，明暗，位置</font>等常见图像变化的问题**

<br>
<br>

**学习资源：[北京邮电大学计算机视觉——鲁鹏](https://www.bilibili.com/video/BV1nz4y197Qv)**



