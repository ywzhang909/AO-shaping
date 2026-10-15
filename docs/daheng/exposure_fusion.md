# 图像处理（十一）exposure fusion 图像曝光融合

原文：https://zhuanlan.zhihu.com/p/455674916  
作者：[山与水你和我](https://www.zhihu.com/people/fluence2crane)  
编辑于 2023-02-12 11:42・上海


## 前言

本意是为了学习 HDR，高动态范围成像，但 HDR 概念对于我这种小白来说概念实在是太乱了，涉及到硬件和软件。我的目标是跳过硬件和 tone mapping，做最简单的多曝光融合，如下列图像

![](https://pic4.zhimg.com/v2-26dc119e85ff4afc18838e14b02238b3_r.jpg)  
*图片 1*

![](https://pic3.zhimg.com/v2-fa4d5fa68ff40a9b6681afbb801de4dc_r.jpg)  
*图片 2*

![](https://pic2.zhimg.com/v2-c412c25938b866b711bf8771059893c5_r.jpg)  
*图片 3*

![](https://pic1.zhimg.com/v2-bbccadb3e213969bf24fa87db6d826bc_r.jpg)  
*图片 4*

融合成一张

![](https://pic3.zhimg.com/v2-9dc7fbf026bc924e487c68c8e4069b30_r.jpg)  
*融合目标*

网络一搜索，曝光融合，跳出来的就是这篇《exposure fusion》的文章，像素级别的融合，再加上多尺度的平滑处理，从现在的眼光来看，是非常简单的，OpenCV 还实现了这个算法，在 lightroom 中也有应用。

虽然原理简单，但实现上有坑，C++ 写了好几遍过不了，换成 Python 也错误，看了官方的 matlab 源码才知道坑。

## 原理

论文提出了三种，计算每个像素融合权重的方法

### 对比度

在过曝图像或严重低曝光的区域，图像都基本是 255 或 0，因此这些区域是很平滑的，梯度信息几乎为 0，这些也是曝光融合中要舍弃的区域——过曝部分缺乏细节，低曝光区域缺乏细节和亮度。因此，论文提出用 laplace 算子提取图像的梯度信息，梯度信息越小的像素在融合时占据的比重就越小，相反梯度信息越大的像素的比重就越大。对输入的第 $k$ 张图像 $I_k$

$C(I_k) = |\Delta{gray(I_{k})}|$

### 饱和度

同样针对过曝或严重低曝光，二者的饱和度是接近于 0 的，而那些饱和度高的区域常常是需要保留的区域，因此论文通过三通道的标准差来衡量饱和度大小，以此得到不同的比重

$\begin{aligned} S_{k, i,j, k} &= \sqrt{\frac{{(I^B_{k, i,j} - M_{k, i, j})}^2 + {(I^G_{k, i,j} - M_{k, i, j})}^2 + {(I^R_{k, i,j} - M_{k, i, j} )}^2}{3}} \\ M_{k, i, j} &= \frac{I^B_{k, i,j} + I^G_{k, i,j} + I^R_{k, i,j}}{3} \end{aligned}$

### 亮度

亮度越接近 0 或者 255 的，很可能就是处在过曝或严重低曝光区域，而亮度在 128 附近的常常可以认为是曝光良好的区域，这些区域的信息在融合过程中要尽量保留，比重尽量大，如下

$E_{k, i, j} = e^{-\frac{(I^B_{k, i,j} - 0.5)^2}{2\sigma^2}} \times e^{-\frac{(I^G_{k, i,j} - 0.5)^2}{2\sigma^2}} \times e^{-\frac{(I^R_{k, i,j} - 0.5)^2}{2\sigma^2}}$

$\sigma$ 是方差，论文中取 0.2。以上公式是我具体实现时使用的，三通道的比重相乘，为了简单，也可以提取出 YCrCb 通道，用 Y 通道算也可以。良好曝光不一定取 0.5。

### 验证

上面是人为提出的一些先验知识, 接下来验证下是否有效

![](https://pic4.zhimg.com/v2-29bf17cc674f9cad711e456b6ed07611_r.jpg)  
*输入图*

![](https://pica.zhimg.com/v2-b2de5f825e83be9f2fe3dcd4a79d79e0_r.jpg)  
*对比度*

![](https://pic1.zhimg.com/v2-cebb05bfea97623d2a1c26ba7c16f600_r.jpg)  
*饱和度*

![](https://picx.zhimg.com/v2-f063ee8a2162ca03a78d51477438650d_r.jpg)  
*亮度*

可以看出，对于严重低曝光的图像，响应值比较高的就是光源附近的细节，这些细节得以保留，而过曝图像往往在这些地方是没有细节的

![](https://pic2.zhimg.com/v2-afbd8249fd995eb02faefb12536bba35_r.jpg)  
*输入图*

![](https://pic3.zhimg.com/v2-361c6e075d9f6cd1db22f51b76fd663a_r.jpg)  
*对比度*

![](https://pica.zhimg.com/v2-6d4ad181d94384501a569db866b9efe6_r.jpg)  
*饱和度*

![](https://pic3.zhimg.com/v2-afc01a5a7d62c365b128ab6b0c0a0508_r.jpg)  
*亮度*

同上，以亮度为例，该图在天空区域的曝光是良好的，而且具有一定的细节，得以保留

![](https://pic4.zhimg.com/v2-5e7ecef026e9762fdeca536c22da6b57_r.jpg)  
*输入图*

![](https://pic2.zhimg.com/v2-309684b1480c8b9e4ed6db22f50b3353_r.jpg)  
*对比度*

![](https://pic1.zhimg.com/v2-3ea11b3aa6a17df4484840d2bc9cebf2_r.jpg)  
*饱和度*

![](https://picx.zhimg.com/v2-9dbaf7d18c724f4a2f4b11de4c6409af_r.jpg)  
*亮度*

对于这一类局部过曝的图像，背光区域反而得到了较好的曝光，所以图中气球的细节和亮度响应值比较高，得以保留，而天空和光源部分的灰度值偏离 128、梯度也较小，因此这些过曝的区域的响应值（比重）很小，在融合时被舍弃。

### 融合

上面根据对比度、饱和度、曝光三方面可以得到每个像素点在融合时的比重，三者合在一起

$W_{k, i, j} = (C_{k,i,j})^{\alpha_{C}} \times (S_{k,i,j})^{\alpha_{S}} \times (E_{k,i,j})^{\alpha_{E}}$

为了防止多张图像融合，出现过高或者过低的值，要对 $W$ 对归一化

$norm(W_{k,i,j}) =  \frac{W_{k,i,j}}{ \sum_{u}{W_{u,i,j}}}$

多张图像 $\{I_1,I_2,...,I_K\}$ ，各自对应一张归一化的比重图 $\{ W_1, W_2,...,W_K \}$ ，融合

$O = \sum_{i=1}^{K} I_i \cdot W_i$

上面的方法非常简单，但会得到十分粗糙的结果。我在具体实现时，得到的结果会出现很多黑色和纯白色的斑点（灰度值为0），而且存在一些光晕，如下：

![](https://pic3.zhimg.com/v2-5d2a9e78fce68a69d6339c1a431a8fb6_r.jpg)  
*直接融合的结果*

![](https://picx.zhimg.com/v2-8047e5430f3cd2e344341592f3cd091f_1440w.jpg)  
*噪声明显，灰度突变明显*

经过我追本溯源，发现这是由于多张图像之间的灰度跳变太大导致的——假设现在有图像序列 $\{ I_1, I_2, I_3, I_4\}$ ，在点 $(u,v)$ 处，比重值分别是 $\{0, \,0.99, \,0, \,0.01\}$ ，因此在融合时， $(u,v)$ 点的灰度值主要取决于 $I_2$ ；在旁边一个点， $(u + 1, v)$ 的比重值却是 $\{0, \,0.01, \,0.99, \,0\}$ ，融合时就主要取决于 $I_3$ ，而从 $I_2 \rightarrow I_3$ 的灰度值跳变太大，因此最后看起来有很多噪声。相邻点的比重相差会这么大，个人看法是比重由对比度、饱和度、亮度相乘得到，实际上，很多区域都是接近于 0，差不多 1e-4 的值，三个相乘量级就到了 1e-12，任意一个指标突然变大变小，权重就一边倒了，基本集中在某一张图像上。理想情况是一系列对比度、饱和度、亮度平滑变化的图像，就不会出现这种问题，但实际也不可能拍这么多精细的图像。

如何解决？一个很直观的想法就是对每张图的权重图做平滑处理，缓解灰度跳变，重新归一化后，再融合，效果如下：

![](https://pic2.zhimg.com/v2-53819276802785cbf7a4896c04487fcd_r.jpg)  
*高斯滤波直径 7，标准差 1.2*

看得出来，有所改善，但还是不够平滑，存在一定程度的光晕，继续加大力度

![](https://pic4.zhimg.com/v2-c9845e5fe96ed64369c9821b8c5b0307_r.jpg)  
*高斯滤波直径 21，标准差 3.4*

又好一点了，但天空还是很明显

![](https://pica.zhimg.com/v2-2b3fd260f63c5aa35cd0d6e73ffa682e_r.jpg)  
*高斯滤波直径 101，标准差 17*

好像又好一点了！但还是存在一些光晕（后面怎么解决的还不清楚）。

但是每次都对原始分辨率做这么大的计算吗？

这又回到了上次总结的拉普拉斯金字塔融合，构造金字塔解决分辨率过大和滤波核过大的问题，**这也是论文提出的方法**

> **[链接]** [山与水你和我：图像处理基础（十）拉普拉斯金字塔、压缩、图像融合](https://zhuanlan.zhihu.com/p/454085730)

使用 laplace 金字塔，对每张图像分解出 laplace 金字塔，对应的比重图分解出高斯金字塔，根据比重对 laplace 金字塔做融合，从低分辨率开始上采样叠加同尺度的融合后 laplace 细节，即可得到结果，如下：

![](https://pic3.zhimg.com/v2-9dc7fbf026bc924e487c68c8e4069b30_r.jpg)  
*laplace 金字塔多曝光融合的结果*

效果有点惊艳，不仅融合了多张图像中的细节，还保持了良好的曝光，和之前的对比还是很明显的，如过曝这个区域的细节还是很不错的！但为什么和之前直径 101 的高斯模糊相比，解决了光晕，这一点是我暂时无法理解的。

![](https://pica.zhimg.com/v2-aa4740247dbd90adddd4eee5012308e6_r.jpg)  
*过曝图像的细节*

## 实现

### 坑

算法十分简单，只要按部就班算出每个点的对比度、饱和度、曝光，结合起来，再加一个 laplace 金字塔分解，问题貌似就解决了。但我在实际写的时候，得到的结果却是这样的：

![](https://pic1.zhimg.com/v2-8dea4132d2403c38c1b4354758be0202_r.jpg)  
*失败的结果*

饱和度太高了，而且细节也没有得到保留。。。。。。一开始还以为是 C++，我索引写错了之类的，重写了几遍都过不了，换成 python 也是一样的，也是错的，后面还是参考了[官方的 matlab 代码](https://link.zhihu.com/?target=https%3A//mericam.github.io/exposure_fusion/exposure_fusion.zip)，才发现不同：

- 一般，构造拉普拉斯金字塔，都是从图像的高斯金字塔中分辨率最低的图像开始，一路上采样，然后对应尺度的高斯金字塔结果减去这个上采样结果，在这次的上采样上继续上采样
- 但这次，却是，从高斯金字塔的低分辨率上采样，对应尺度的高斯金字塔结果减去这个上采样结果，**然后，**下一次的上采样结果不是通过之前的上采样结果得到的，而是通过高斯金字塔的结果上采样得到的。。。就很坑

即下面两张图的对比

![](https://pic3.zhimg.com/v2-cd69adb1412bb7dd27ed49b48b7d4432_r.jpg)  
*之前的 laplace 金字塔融合*

![](https://pic4.zhimg.com/v2-80cd2b3bdd0c39ba1d03b812c588a765_r.jpg)  
*修改后的 laplace 金字塔融合*

```matlab
% recursively build pyramid
pyr = cell(nlev,1);
filter = pyramid_filter;
J = I;
for l = 1:nlev - 1
% apply low pass filter, and downsample
I = downsample(J,filter);
odd = 2*size(I) - size(J);  % for each dimension, check if the upsampled version has to be odd
% in each level, store difference between image and upsampled low pass version
pyr{l} = J - upsample(I,odd,filter);
J = I; % continue with low pass image
end
pyr{nlev} = J;
```

当我把上采样的对象改成了官方的形式，就正常了。

至于为什么会这么差, 我想可能是因为一路上采样,丢失的信息太多了, 所以采用之前的高斯金字塔的次一级分辨率上采样可以减少信息损失。

### 代码

见我的 github

> **[链接]** [image_processing/fusion/exposure_fusion at main · hermosayhl/image_processing](https://link.zhihu.com/?target=https%3A//github.com/hermosayhl/image_processing/tree/main/fusion/exposure_fusion)

Python 3

```python3
# Python
import os
import sys
import math
import copy
# 3rd party
import cv2
import numpy

def cv_show(image, message="crane"):
cv2.imshow(message, image)
cv2.waitKey(0)
cv2.destroyAllWindows()

def exposure_fusion(sequence, alphas=(1.0, 1.0, 1.0), best_illumination=0.5, sigma=0.2, eps=1e-12, 
use_lappyr=True, use_gaussi=False, layers_num=5, scale=2.0):
# 转化成 float 数据
sequence = numpy.stack([it.astype("float32") / 255 for it in sequence], axis=0)
S = len(sequence)
H, W, C = sequence[0].shape
# 准备一些中间变量
laplace_kernel = numpy.array(([0,  1, 0], [1, -4, 1], [0,  1, 0]), dtype="float32")
mse = lambda l, r: (l - r) * (l - r)
best_illumination = numpy.full((H, W), best_illumination, dtype='float32')
# 存放每张图像的权重图
weights = []
# 每张图像都求一个初始权重
for s in range(S):
# 从拉普拉斯求对比度
gray = cv2.cvtColor(sequence[s], cv2.COLOR_BGR2GRAY)
contrast = cv2.filter2D(gray, -1, laplace_kernel, borderType=cv2.BORDER_REPLICATE)
contrast = numpy.abs(contrast)
# 求饱和度
mean = numpy.mean(sequence[s], axis=-1)
saturation = numpy.sqrt(numpy.mean([mse(sequence[s][:, :, ch], mean) for ch in range(3)], axis=0))
# 求亮度
illumination = [numpy.exp(-0.5 * mse(sequence[s][:, :, ch], best_illumination) / (sigma * sigma)) for ch in range(3)]
illumination = numpy.prod(illumination, axis=0)
# 三者加权
cur_weight = numpy.power(contrast, alphas[0]) * numpy.power(saturation, alphas[1]) * numpy.power(illumination, alphas[2])
weights.append(cur_weight)
# 归一化
normalize = lambda x: x / numpy.expand_dims(numpy.sum(x, axis=0), axis=0)
weights = numpy.stack(weights, axis=0)
weights += eps
weights = normalize(weights)
# 这里要把 sequence 还原回来
sequence *= 255;
# 粗糙的融合
origin_fusion = numpy.sum(numpy.stack([weights, weights, weights], axis=-1) * sequence, axis=0)
origin_fusion = numpy.clip(origin_fusion, 0, 255).astype("uint8")
results = {"naive": origin_fusion}
# 是否用高斯
if(use_gaussi == True):
# 使用高斯模糊对 weights 做模糊
smoothed_weights = numpy.stack([cv2.GaussianBlur(w, (101, 101), 17) for w in weights], axis=0)
smoothed_weights = normalize(smoothed_weights)
smoothed_fusion = numpy.sum(numpy.stack([smoothed_weights, smoothed_weights, smoothed_weights], axis=-1) * sequence, axis=0)
smoothed_fusion = numpy.clip(smoothed_fusion, 0, 255).astype("uint8")
results.update({"gaussi_smoothed": smoothed_fusion})
# 是否用拉普拉斯
if(use_lappyr == True):

# 根据最高分辨率的图像 high_res, 得到高度 layers 的高斯金字塔
def build_gaussi_pyramid(high_res, layers):
this_flash = [high_res]
for i in range(1, layers):
# 先对当前权重做高斯模糊, 然后下采样
blurred = cv2.GaussianBlur(this_flash[i - 1], (5, 5), 0.83)
blurred = blurred[::2, ::2]
this_flash.append(blurred)
return this_flash

# 根据已知的高斯金字塔, 从最底层开始上采样, 得到每一个尺度的 laplace 细节
def build_laplace_pyramaid(gaussi_pyramid, layers):
upsampled = gaussi_pyramid[layers - 1]
pyramid = [upsampled]
for i in range(layers - 1, 0, -1):
size = (gaussi_pyramid[i - 1].shape[1], gaussi_pyramid[i - 1].shape[0])
# upsampled = cv2.resize(upsampled, size) # 假如我一直上采样呢
upsampled = cv2.resize(gaussi_pyramid[i], size)
pyramid.append(gaussi_pyramid[i - 1] - upsampled)
pyramid.reverse() # 目前分辨率都是从高到低排列的
return pyramid

# 求每张图的权重的高斯金字塔
sequence_weights_pyramids = [build_gaussi_pyramid(weights[s], layers_num) for s in range(S)]
# 求每张图的高斯金字塔, 以求 laplace
sequence_gaussi_pyramids = [build_gaussi_pyramid(sequence[s], layers_num) for s in range(S)]
# 求每张图的 laplace 金字塔
sequence_laplace_pyramids = [build_laplace_pyramaid(sequence_gaussi_pyramids[s], layers_num) for s in range(S)]
# 这里可以归一化
# 每一个尺度, 融合一系列图像的的 laplace 细节, 得到一个融合的 laplace 金字塔
fused_laplace_pyramid = [numpy.sum([sequence_laplace_pyramids[k][n] * 
numpy.atleast_3d(sequence_weights_pyramids[k][n]) for k in range(S)], axis=0) for n in range(layers_num)]

# 先从最底层的图像开始, 每次上采样都加上同等尺度的 laplace 细节
start = fused_laplace_pyramid[layers_num - 1]
for i in range(layers_num - 2, -1, -1):
upsampled = cv2.resize(start, (fused_laplace_pyramid[i].shape[1], fused_laplace_pyramid[i].shape[0]))
start = fused_laplace_pyramid[i] + upsampled
# 灰度值截断在 0-255 之间
start = numpy.clip(start, 0, 255).astype("uint8")
# 放到结果列表中
results.update({"laplace_pyramid": start})

return results

images_dir = "../images/input/5"
save_dir = "../images/output/5"
os.makedirs(save_dir, exist_ok=True)

# 读取图片
images_list = [os.path.join(images_dir, it) for it in os.listdir(images_dir)]
sequence = numpy.stack([cv2.imread(name) for name in images_list])

# 曝光融合
fused_results = exposure_fusion(sequence, use_lappyr=True, use_gaussi=True, layers_num=7)

# 展示与保存
for l, r in fused_results.items():
cv_show(r, l)
cv2.imwrite(os.path.join(save_dir, l + ".png"), r, [cv2.IMWRITE_PNG_COMPRESSION, 0])
```

## 鬼影

本篇讨论的多曝光图像融合，默认都是对齐的，但实际上不同时间拍的图像，会因相机抖动和环境的风吹草动，而导致内容有差别，使得最后的融合结果可能出现鬼影，而鬼影(ghost effect)也是 HDR 融合要解决的主要问题之一。

### 移动

考虑下面这个例子

![](https://pic1.zhimg.com/v2-b4f3738d5dc27b9709af5f90ddb5377e_r.jpg)

![](https://pic1.zhimg.com/v2-b72e1fb4e80a88122dc0bc0e3ab0efca_r.jpg)

![](https://pica.zhimg.com/v2-5f75a8666ebedca3816d0c90a1fe1df0_r.jpg)

可以看到画面中的三个人物都存在较明显的移动，经过上面的 exposure fusion 像素级融合，就会得到下面的结果

![](https://pic3.zhimg.com/v2-beb9020531f51b26d1164ec5cab870ce_r.jpg)  
*鬼影出现*

![](https://pic2.zhimg.com/v2-ef04db439ad92ad67177f9ef625f6ee3_1440w.jpg)  
*鬼影 1*

![](https://pic1.zhimg.com/v2-8b888603a6caef7339abb19ca5919d9a_r.jpg)  
*鬼影 2*

![](https://picx.zhimg.com/v2-5e4d9ff435815afa51deaa12f835f30f_1440w.jpg)  
*鬼影 3*

这一类情况下，不保证像素级的对齐，就会出现鬼影。

### 模糊

和多曝光融合类似的是多焦点融合, 不同图像存在不同的局部模糊

![](https://pic1.zhimg.com/v2-a0104cd49dd86248d4eb974124d8ffa6_1440w.jpg)  
*图像 1*

![](https://pic4.zhimg.com/v2-f4bb159452117b34693ed39439a81d41_1440w.jpg)  
*图像 2*

保留对比度、饱和度, 不计算曝光, 得到的融合结果如下:

![](https://pic2.zhimg.com/v2-71e18629dbf568dca1b6ac919c6411af_1440w.jpg)  
*多焦点融合结果*

效果感觉还行, 但又不是很行 ! 如 IMAGE 那块区域, 第一张图有点类似于运动模糊的效果了, 在曝光融合这里就近似于"没对齐", 因此融合的结果有一些虚影、鬼影 !

虽然存在如上问题，但这个方法还是给后续的一些研究提供了思路，融合不断频段的信息做图像融合，不仅仅是曝光融合，多焦点融合等。

## 参考资料

1. Mertens T, Kautz J, Van Reeth F. Exposure fusion[C]//15th Pacific Conference on Computer Graphics and Applications (PG'07). IEEE, 2007: 382-390.
2. Mertens T, Kautz J, Van Reeth F. Exposure fusion: A simple and practical alternative to high dynamic range photography[C]//Computer graphics forum. Oxford, UK: Blackwell Publishing Ltd, 2009, 28(1): 161-171.
3. [https://mericam.github.io/exposure_fusion/index.html](https://link.zhihu.com/?target=https%3A//mericam.github.io/exposure_fusion/index.html)
4. 多曝光融合图像来源 [http://ivc.uwaterloo.ca/database/MEF.html](https://link.zhihu.com/?target=http%3A//ivc.uwaterloo.ca/database/MEF.html)
5. 多曝光、真实的 HDR 数据来源 [Deep High Dynamic Range Imaging of Dynamic Scenes](https://link.zhihu.com/?target=https%3A//cseweb.ucsd.edu/~viscomp/projects/SIG17HDR/)
6. 多焦点融合图像来源 [Multifocus Image Fusion](https://link.zhihu.com/?target=https%3A//dsp.etfbl.net/mif/)