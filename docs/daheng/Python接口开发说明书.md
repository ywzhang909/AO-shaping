## Python 接口开发说明书

本手册中所提及的其它软硬件产品的商标与名称，都属于相应公司所有。

本手册的版权属于中国大恒（集团）有限公司北京图像视觉技术分公司所有。未得到本公司的正式许可，任何组织或个人均不得以任何手段和形式对本手册内容进行复制或传播。

本手册的内容若有任何修改，恕不另行通知。

© 2021中国大恒（集团）有限公司北京图像视觉技术分公司版权所有

网 站： www.daheng-imaging.com

公 司 总 机： 010-82828878

客户服务热线：400-999-7595

销 售 信 箱： sales@daheng-imaging.com

支 持 信 箱： support@daheng-imaging.com

## 目录

1. 相机工作流程 …… 1
1.1. 整体工作流程 …… 1
1.2. 功能控制流程 …… 2
1.3. 整体代码样例 …… 3
2. 编程指引 …… 5
2.1. 搭建编程环境 …… 5
2.1.1. Linux …… 5
2.1.2. Windows …… 6
2.2. 快速上手 …… 6
2.2.1. 引入库 …… 6
2.2.2. 枚举设备 …… 7
2.2.3. 打开关闭设备 …… 8
2.2.4. 采集控制 …… 9
2.2.5. 图像处理 …… 10
2.2.6. 相机控制 …… 12
2.2.7. 导入导出相机配置参数 …… 15
2.2.8. 错误处理 …… 16
3. 附录 …… 17
3.1. 属性参数 …… 17
3.1.1. 设备属性参数 …… 17
3.1.2. 流属性参数 …… 24
3.2. 功能类定义 …… 25
3.2.1. Feature …… 25
3.2.2. IntFeature …… 26
3.2.3. FloatFeature …… 28
3.2.4. EnumFeature …… 29
3.2.5. BoolFeature …… 31
3.2.6. StringFeature …… 32
3.2.7. BufferFeature …… 34
3.2.8. CommandFeature …… 35
3.3. 数据类型定义 …… 36
3.3.1. GxDeviceClassList …… 36

3.3.2. GxAccessStatus .... 36
3.3.3. GxAccessMode .... 36
3.3.4. GxPixelFormatEntry .... 36
3.3.5. GxFrameStatusList .... 37
3.3.6. GxDeviceTemperatureSelectorEntry .... 37
3.3.7. GxPixelSizeEntry .... 37
3.3.8. GxPixelColorFilterEntry .... 38
3.3.9. GxAcquisitionModeEntry .... 38
3.3.10. GxTriggerSourceEntry .... 38
3.3.11. GxTriggerActivationEntry .... 38
3.3.12. GxExposureModeEntry .... 38
3.3.13. GxUserOutputSelectorEntry .... 39
3.3.14. GxUserOutputModeEntry .... 39
3.3.15. GxGainSelectorEntry .... 39
3.3.16. GxBlackLevelSelectEntry .... 39
3.3.17. GxBalanceRatioSelectorEntry .... 39
3.3.18. GxAALightEnvironmentEntry .... 39
3.3.19. GxUserSetEntry .... 39
3.3.20. GxAWBLampHouseEntry .... 40
3.3.21. GxUserDataFieldSelectorEntry .... 40
3.3.22. GxTestPatternEntry .... 40
3.3.23. GxTriggerSelectorEntry .... 40
3.3.24. GxLineSelectorEntry .... 40
3.3.25. GxLineModeEntry .... 41
3.3.26. GxLineSourceEntry .... 41
3.3.27. GxEventSelectorEntry .... 41
3.3.28. GxLutSelectorEntry .... 42
3.3.29. GxTransferControlModeEntry .... 42
3.3.30. GxTransferOperationModeEntry .... 42
3.3.31. GxTestPatternGeneratorSelectorEntry .... 42
3.3.32. GxChunkSelectorEntry .... 42
3.3.33. GxBinningHorizontalModeEntry .... 42
3.3.34. GxBinningVerticalModeEntry .... 42
3.3.35. GxSensorShutterModeEntry .... 42
3.3.36. GxC acquisitionStatusSelectorEntry .... 43
3.3.37. GxExposureTimeModeEntry .... 43
3.3.38. GxGammaModeEntry .... 43
3.3.39. GxLightSourcePresetEntry .... 43
3.3.40. GxColorTransformationModeEntry .... 43
3.3.41. GxColorTransformationValueSelectorEntry .... 43
3.3.42. GxAutoEntry .... 44
3.3.43. GxSwitchEntry .... 44
3.3.44. GxRegionSendModeEntry .... 44
3.3.45. GxRegionSelectorEntry .... 44 

3.3.46. GxTimerSelectorEntry .... 44
3.3.47. GxTimerTriggerSourceEntry .... 44
3.3.48. GxCounterSelectorEntry .... 44
3.3.49. GxCounterEventSourceEntry .... 45
3.3.50. GxCounterResetSourceEntry .... 45
3.3.51. GxCounterResetActivationEntry .... 45
3.3.52. GxCounterTriggerSourceEntry .... 45
3.3.53. GxTimerTriggerActivationEntry .... 45
3.3.54. GxStopAcquisitionModeEntry .... 46
3.3.55. GxDSSreamBufferHandlingModeEntry .... 46
3.3.56. GxResetDeviceModeEntry .... 46
3.3.57. DxBayerConvertType .... 46
3.3.58. DxValidBit .... 46
3.3.59. DxImageMirrorMode.... 46
3.3.60. DxRGBChannelOrder.... 46

3.4. 模块接口定义.... 47
3.4.1. DeviceManager.... 47
3.4.2. Device.... 52
3.4.3. DataStream.... 55
3.4.4. RGBImage.... 57
3.4.5. RawImage.... 60
3.4.6. Buffer.... 67
3.4.7. Utility.... 69

4. 常见问题解答.... 72

5. 版本说明.... 73

## 1. 相机工作流程

## 1.1. 整体工作流程

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-15/f2a72e6b-de8e-4057-b4f2-8dc0b0d19138/99592c40ede8c13c7e9b77d806c63518d583f2d88a2ae5202db26b06970caa94.jpg)


## 1.2. 功能控制流程

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-15/f2a72e6b-de8e-4057-b4f2-8dc0b0d19138/b4248a1dc88d85ea5413e8d43a306a63a19f71124347577375edaa1a05165b87.jpg)


cam.LUTValueAll.set_buffer(value) 

cam.LUTValueAll.get_buffer() 

布尔型

cam.UserOutputValue.set(value) 

cam.UserOutputValue.get() 

## 1.3. 整体代码样例

```python
# 用户可自定义调用前缀，样例中使用了 gx
import gxipy as gx

# 枚举设备。dev_info_list 是设备信息列表，列表的元素个数为枚举到的设备个数，列表元素是字
# 典，其中包含设备索引（index）、ip 信息（ip）等设备信息
device_manager = gx.DeviceManager()
dev_num, dev_info_list = device_manager.update_device_list()
if dev_num == 0:
    sys.exit(1)

# 打开设备
# 获取设备基本信息列表
strSN = dev_info_list[0].get("sn")
# 通过序列号打开设备
cam = device_manager.open_device_by_sn(strSN)

# 开始采集
cam.stream_on()

# 获取流通道个数
# 如果 int_channel_num == 1，设备只有一个流通道，列表 data_stream 元素个数为 1
# 如果 int_channel_num > 1，设备有多个流通道，列表 data_stream 元素个数大于 1
# 目前千兆网相机、USB3.0、USB2.0 相机均不支持多流通道。
# int_channel_num = cam.get_stream_channel_num()

# 获取数据
# num 为采集图片次数
num = 1
for i in range(num):
    # 从第 0 个流通道获取一幅图像
    raw_image = cam.data_stream[0].get_image()
    # 从彩色原始图像获取 RGB 图像
    rgb_image = raw_image.convert("RGB")
    if rgb_image is None:
    continue
    # 从 RGB 图像数据创建 numpy 数组
    numpy_image = rgb_image.get_numpy_array()
    if numpy_image is None:
    continue
    # 显示并保存获得的 RGB 图片
    image = Image.fromarray(numpy_image, 'RGB')
    image.show()
```

image.save("image.jpg") 

# 停止采集，关闭设备

cam.stream_off() 

cam.close_device() 

## 2. 编程指引

## 2.1. 搭建编程环境

推荐用户优先使用 Python2.7、Python3.5 版本，本接口库已在上述两版本环境中测试通过。

## 2.1.1. Linux

1) 安装 pycharm-community（社区免费版本）

sudo apt-get install pycharm-community 

2) 新建 project，选择已安装 gxipy 库的 python 解释器，如果没有可选项，点击右侧的“设置”图标按钮，添加对应版本的 python 解释器。

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-15/f2a72e6b-de8e-4057-b4f2-8dc0b0d19138/9833bcef043bc10830b6749e1f36a067a0c8c37a5d2b6a42b8787c47b38bb651.jpg)


3) File->Settings->Project->Project Interpreter 可以查看 python 解释器已安装的 Package，如图表示 gxipy 库已正常安装，可以导入 gxipy 模块使用。

<table><tr><td>Package</td><td></td></tr><tr><td>Pillow</td><td>5.2.0</td></tr><tr><td>Pillow-PIL</td><td>0.1.dev0</td></tr><tr><td>Pygments</td><td>2.1</td></tr><tr><td>Pypubsub</td><td>4.0.0</td></tr><tr><td>adium-theme-ubuntu</td><td>0.3.4</td></tr><tr><td>argparse</td><td>1.2.1</td></tr><tr><td>backports.functools-l.ru-cache</td><td>1.5</td></tr><tr><td>chardet</td><td>2.3.0</td></tr><tr><td>configobj</td><td>5.0.6</td></tr><tr><td>cycler</td><td>0.10.0</td></tr><tr><td>decorator</td><td>4.0.6</td></tr><tr><td>dulwich</td><td>0.12.0</td></tr><tr><td>fastimport</td><td>0.9.4</td></tr><tr><td>gxipy</td><td>1.0.1807.9182</td></tr><tr><td>ipython</td><td>2.4.1</td></tr><tr><td>kiwisolver</td><td>1.0.1</td></tr><tr><td>matplotlib</td><td>2.2.2</td></tr><tr><td>meld</td><td>3.14.2</td></tr></table>

## 2.1.2. Windows

以 Python2.7、pycharm2018 平台为例，在 windows 7 64 位操作系统环境下演示安装 gxipy 库。初次安装 gxipy 库时：

1) 打开 pycharm2018 的工程配置 File->Settings->Project->Project Interpreter，点击下图红色方框出现的Add。

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-15/f2a72e6b-de8e-4057-b4f2-8dc0b0d19138/4ece275987eda0ad400c4b4d7809463cd66ed3043131ad54b44fb42bca3c753d.jpg)


2) 选择 New environment，选中 Inherit global site-packages 和 Make available to all projects，点击 OK。这时用户将在新建的 interpreter 中看到 gxipy 库已被成功加载到当前工程环境中。

```txt
Project: test_111.py > Project Interpreter For current project Reset
Project Interpreter: Python 2.7 (venv) E:\010_2018\050_python_interface\python_APT\private\dev\test\venv\Scripts\python.exe
Package Version Latest
Pillow 5.1.1 5.2.0
PyCapture2 2.11.425
gxipy 1.0.1807.9021
numpy 1.14.5 1.15.0rc1
pip 10.0.1 10.0.1
setuptools 39.2.0 39.2.0
wheel 0.31.1 0.31.1 
```

## 2.2. 快速上手

## 2.2.1. 引入库

为使用已安装的库，在程序的开始引入库。在代码中任何使用库中的类、方法、数据类型时，请加自定义的前缀。

代码样例：

```txt
# 用户可自定义调用前缀，样例中使用了 gx
import gxipy as gx

device_manager = gx.DeviceManager()
```

## 2.2.2. 枚举设备

用户通过调用 DeviceManager.update_device_list()枚举当前所有可用设备，函数返回值为设备数量dev_num 和设备信息列表 dev_info_list。设备信息列表的元素个数为枚举到的设备个数，列表中元素的数据类型为字典，字典的键详见下表：

<table><tr><td>键名称</td><td>意义</td><td>类型</td></tr><tr><td>index</td><td>设备索引</td><td>整型</td></tr><tr><td>vendor_name</td><td>厂商名称</td><td>字符串</td></tr><tr><td>model_name</td><td>设备型号</td><td>字符串</td></tr><tr><td>sn</td><td>设备序列号</td><td>字符串</td></tr><tr><td>display_name</td><td>设备显示名称</td><td>字符串</td></tr><tr><td>device_id</td><td>设备标识</td><td>字符串</td></tr><tr><td>user_id</td><td>用户自定义名称</td><td>字符串</td></tr><tr><td>access_status</td><td>权限状态</td><td>GxAccessStatus</td></tr><tr><td>device_class</td><td>设备类</td><td>GxDeviceClassList</td></tr><tr><td>mac</td><td>mac 地址(GEV 相机特有)</td><td>字符串</td></tr><tr><td>ip</td><td>ip 地址(GEV 相机特有)</td><td>字符串</td></tr><tr><td>subnet_mask</td><td>子网掩码(GEV 相机特有)</td><td>字符串</td></tr><tr><td>gateway</td><td>网关(GEV 相机特有)</td><td>字符串</td></tr><tr><td>nic_mac</td><td>nic_mac 地址(GEV 相机特有)</td><td>字符串</td></tr><tr><td>nic_ip</td><td>nic_ip 地址(GEV 相机特有)</td><td>字符串</td></tr><tr><td>nic_subnet_mask</td><td>nic 子网掩码(GEV 相机特有)</td><td>字符串</td></tr><tr><td>nic_gateway</td><td>nic 网关(GEV 相机特有)</td><td>字符串</td></tr><tr><td>nic_description</td><td>nic 描述(GEV 相机特有)</td><td>字符串</td></tr></table>

## 枚举设备代码片段如下：

```python
# 枚举设备
device_manager = gx.DeviceManager()
dev_num, dev_info_list = device_manager.update_device_list()
if dev_num == 0:
    sys.exit(1)
```

## 注意：

除上面的枚举接口，DeviceManager 还提供了另一个枚举接口 DeviceManager.update_all_device_list()：

1) 对于非千兆网相机来说，这两个枚举接口功能上是一样的；

2) 对千兆网相机来说，库内部使用的枚举机制不一样：

update_all_device_list：使用全网枚举，能够枚举到局域网内的所有千兆网相机

update_device_list：使用子网枚举，只能枚举到局域网内的同一网段的千兆网相机

```txt
ip 为设备 ip 地址（非千兆网相机不可用）
```

## 2.2.3. 打开关闭设备

用户通过调用以下五种不同的方式打开设备：

```python
DeviceManager.open_device_by_sn(self, sn, access_mode=GxAccessMode.CONTROL)
DeviceManager.open_device_by_user_id(self, user_id, access_mode=GxAccessMode.CONTROL)
DeviceManager.open_device_by_index(self, index, access_mode=GxAccessMode.CONTROL)
DeviceManager.open_device_by_ip(ip, access_mode=GxAccessMode.CONTROL)
DeviceManager.open_device_by_mac(mac, access_mode=GxAccessMode.CONTROL) 
```

## 其中：

sn 为设备序列号

use_id 为用户自定义名称

index 为设备索引（1,2,3...）

mac 为设备mac地址（非千兆网相机不可用）

## 注意：

最后两个函数只针对千兆网相机使用。

用户可以调用 Device 提供的 Device. close_device()接口来关闭设备，释放所有设备资源。

## 代码样例：

```python
# 用户可自定义调用前缀，样例中使用了 gx
import gxipy as gx

# 枚举设备。dev_info_list 是设备信息列表，列表的元素个数为枚举到的设备个数，列表元素是
# 字典，其中包含设备索引（index）、ip 信息（ip）等设备信息
device_manager = gx.DeviceManager()
dev_num, dev_info_list = device_manager.update_device_list()
if dev_num == 0:
    sys.exit(1)

# 打开设备

# 方法一
# 获取设备基本信息列表
str_sn = dev_info_list[0].get("sn")
# 通过序列号打开设备
cam = device_manager.open_device_by_sn(str_sn)

# 方法二
# 通过用户 ID 打开设备
# str_user_id = dev_info_list[0].get("user_id")
# cam = device_manager.open_device_by_user_id(str_user_id)
```

```python
# 方法三
# 通过索引打开设备
# str_index = dev_info_list[0].get("index")
# cam = device_manager.open_device_by_index(str_index)
# 下面为只针对千兆网相机使用的打开方式
# 方法四
# 通过 ip 地址打开设备
# str_ip = dev_info_list[0].get("ip")
# cam = device_manager.open_device_by_ip(str_ip)
# 方法五
# 通过 mac 地址打开设备
# str_mac = dev_info_list[0].get("mac")
# cam = device_manager.open_device_by_mac(str_mac)
# 关闭设备
cam.close_device()
```

## 2.2.4. 采集控制

在设备正常开启并设置好相机采集参数后，用户可调用 Device.stream_on()和 Device.stream_off()执行开停采：

与采集相关的接口和控制都由 DataStream 提供，可通过设置循环次数控制采集图像次数。通过Device.get_stream_channel_num()接口获得设备的流通道个数。获取的图像使用 RawImage.get_status()判断是否为残帧。代码如下：

##  get_image 方式

```python
# 开始采集
cam.stream_on()

# 获取流通道个数
# 如果 int_channel_num == 1，设备只有一个流通道，列表 data_stream 元素个数为 1
# 如果 int_channel_num > 1，设备有多个流通道，列表 data_stream 元素个数大于 1
# 目前千兆网相机、USB3.0、USB2.0 相机均不支持多流通道
# int_channel_num = cam.get_stream_channel_num()
# 获取数据
# num 为采集图片次数
num = 1
for i in range(num):
    # 打开第 0 通道数据流
```

```python
raw_image = cam.data_stream[0].get_image()
if raw_image.get_status() == gx.GxFrameStatusList.INCOMPLETE:
    print("incomplete frame")

# 停止采集
cam.stream_off()
```

##  回调方式

```python
# 定义采集回调函数
def capture_callback(raw_image):
    if raw_image.get_status() == gx.GxFrameStatusList.INCOMPLETE:
    print("incomplete frame")

# 注册回调
cam.data_stream[0].register_capture_callback(capture_callback)

# 开始采集
cam.stream_on()

# 等待一段时间，这段时间会自动调用采集回调函数
time.sleep(1)

# 停止采集
cam.stream_off()

# 注销回调
cam.data_stream[0].unregister_capture_callback()
```

## 2.2.5. 图像处理

##  图像格式转换

图像格式转换的对象是采集图像 DataStream.get_image()得到的 raw_image。

## 功能描述：

将 Bayer 格式图像转换成 RGB 格式图像。详见 RawImage 的接口 RawImage.convert()。

代码样例：

## 1) 彩色相机

```python
raw_image = cam.data_stream[0].get_image()
# 保存 raw 图
raw_image.save_raw("raw_image.raw")
# 从彩色原始图像获取 RGB 图像
rgb_image = raw_image.convert("RGB")
if rgb_image is None:
    continue
# 从 RGB 图像数据创建 numpy 数组
```

```python
numpy_image = rgb_image.get_numpy_array()
if numpy_image is None:
    continue
# 之后，用户可根据获取的 numpy_array 显示、保存图像
```

## 2) 黑白相机

```python
raw_image = cam.data_stream[0].get_image()
# 从黑白原始图像获取 numpy 数组
numpy_image = raw_image.get_numpy()
if numpy_image is None:
    continue
# 之后，用户可根据获取的 numpy_array 显示、保存图像
```

##  图像质量提升

本接口库还提供了软件端的图像质量提升接口，用户可以有选择的进行颜色校正、对比度、Gamma等图像质量提升的操作。用户可调用 RGBImage 的接口 RGBImage.image_improvement()实现功能。一个简单的使用范例如下：

```python
# 设置图像质量提升的参数
if cam.GammaParam.is_readable():
    gamma_value = cam.GammaParam.get()
    gamma_lut = gx.Utility.get_gamma_lut(gamma_value)
else:
    gamma_lut = None
if cam.ContrastParam.is_readable():
    contrast_value = cam.ContrastParam.get()
    contrast_lut = gx.Utility.get_contrast_lut(contrast_value)
else:
    contrast_lut = None
color_correction_param = cam.ColorCorrectionParam.get()

# 采集获取图像、格式转换
# ......
# 实现图像质量提升
rgb_image.image_improvement(color_correction_param, contrast_lut, gamma_lut)
```

## 1) 颜色校正

设备属性参数名称：ColorCorrectionParam

名词解释：提高相机色彩还原度，使图像更加接近人眼视觉感受。

## 2) 对比度调节

Contrast 值：整型，范围[-50, 100]，缺省值为 0

设备属性参数名称：ContrastParam

名词解释：图像明亮部分与黑暗部分的亮度比称为对比度，又叫反差。对比度高或者反差大的图像，其中被摄景物的轮廓较清楚，图像也较清晰；反之，对比度低的图像轮廓不清，图像也不太清晰。

## 3) Gamma 调节

Gamma 值：整型或浮点型，范围[0.1, 10.0]，缺省值为 1

设备属性参数名称： GammaParam

名词解释：Gamma调节是为了让显示器的输出尽量接近输入。

##  图像显示和保存

调用 PIL(Python Imaging Library)的接口 Image.fromarray()，将 numpy 数组转换成 Image 图像，显示并保存。代码如下：

## 1) 黑白相机

```txt
# 显示并保存获得的黑白图片
image = Image.fromarray(numpy_image, 'L')
image.show()
image.save("acquisition_mono_image.jpg")
```

## 2) 彩色相机

```txt
# 显示并保存获得的彩色图片
image = Image.fromarray(numpy_image, 'RGB')
image.show()
image.save("acquisition_RGB_image.jpg")
```

## 2.2.6. 相机控制

##  属性参数访问类型

属性参数的访问分三种类型：是否实现、是否可读、是否可写。接口设计如下：

<table><tr><td>Feature.is implement</td><td>当前属性控制器是否支持此功能</td></tr><tr><td>Feature.is readable</td><td>此功能是否可读</td></tr><tr><td>Feature.is writable</td><td>此功能是否可写</td></tr></table>

建议用户在操作属性参数前，先查询属性的访问类型。

## 代码样例：

```python
# 获取功能是否实现
is_ implemented = cam.PixelFormat.is_ implemented()
if is_ implemented == True:
    # 获取是否可写
    is_writable = cam.PixelFormat.is_writable()
    if is_writable == True:
    # 设置像素格式
    cam.PixelFormat.set(gx.GxPixelFormatEntry.MON08)
# 获取是否可读
is_readable = cam.PixelFormat.is_readable()
if is_readable == True:
```

```txt
# 打印像素格式
print(cam.PixelFormat.get())
```

##  属性控制

属性控制分为设备属性参数和流属性参数两类。

区别在于属性参数的控制功能有所不同：

设备属性参数：设备信息，比如宽高、曝光增益等；

流属性参数：关于采集控制和采集数据统计的属性访问控制器。

当获得相机实例化对象后，可通过访问其属性参数，访问到属性参数的控制接口，这些接口在控制相机功能中。

例如：

```kotlin
cam.UserSetSelector.set(gx.GxUserSetEntry.DEFAULT)
cam.UserSetLoad.send_command()
cam.UserSetSelector.set(gx.GxUserSetEntry.USER_SETO)
cam.UserSetLoad.send_command()
cam.UserSetDefault.set(gx.GxUserSetEntry.DEFAULT) 
```

相机实例化对象属性参数控制属性参数方法

```txt
cam.data_stream[0].StreamLostFrameCount.get()
相机实例化对象 流对象 属性参数 控制属性参数方法
```

接口调用根据属性参数的类型的不同分类：

## 整型：

相关接口：

```txt
IntFeature.set(int_value)    //设置
IntFeature.get()    //读取
IntFeature.get_range()    //获取最小值、最大值、步长
```

代码样例：

```python
# 获取图像宽度可设置范围
int_range = cam.Width.get_range()
# 设置当前图像宽度为范围内任意值
cam.Width.set(800)
# 获取当前图像宽度
int_Width_value = cam.Width.get()
```

## 浮点型：

相关接口：

```txt
相关接口：
StringFeature.set(string_value)    //设置
```

```txt
FloatFeature.set(float_value)    //设置
FloatFeature.get()    //读取
FloatFeature.get_range()    //获取最小值、最大值、步长、单位、单位是否有效
```

## 代码样例：

```python
# 获取曝光值可设置范围和最大值
float_range = cam.ExposureTime.get_range()
float_max = float_range["max"]
# 设置当前曝光值范围内任意值
cam.ExposureTime.set(10.0)
# 获取当前曝光值
float_exposure_value = cam.ExposureTime.get()
```

## 枚举型：

```txt
相关接口：
EnumFeature.set(enum_value)    //设置
EnumFeature.get()    //读取
EnumFeature.get_range()    //获取字典
```

## 代码样例：

```python
# 获取枚举值可设置范围。
enum_range = cam.PixelFormat.get_range()
# 设置当前枚举值
cam.PixelFormat.set(gx.GxPixelFormatEntry.MON08)
# 打印当前枚举值
enum_PixelForamt_value, enum_PixelFormat_key = cam.PixelFormat.get()
```

## 布尔型：

相关接口：

## 代码样例：

```python
设置当前布尔值
cam.LineInverter.set(True)
# 读取布尔值
bool_LineInverter_value = cam.LineInverter.get()
```

## 字符串型：

```txt
StringFeature.get()    //读取
StringFeature.get_string_max_length()    //获取字符串型属性参数的最大长度值
```

## 代码样例：

```txt
# 读取最长字符串长度
string_max_length = cam.DeviceUserID.get_string_max_length()
# 读取当前字符串值
current_string = cam.DeviceUserID.get()
# 设置字符串值
cam.DeviceUserID.set("MyUserID")
```

## Buffer 型：

## 相关接口：

```txt
BufferFeature.set_buffer(buf)    //设置
BufferFeature.get_buffer()    //读取
BufferFeature.get_buffer_length()    //获取 Buffer 型属性参数的长度
```

## 代码样例：

```python
import gxipy as gx

# 读取缓冲数据长度
buffer_length = cam.UserData.get_buffer_length()

# 设置缓冲数据
cam.UserData.set_buffer(gx.Buffer.from_string(b'BufferFeature Test!'))
# 读取缓冲数据
buffer_data = cam.UserData.get_buffer()
print("UserData: %s" % (buffer_data.get_data().decode())
```

## Command 型：

## 相关接口：

```txt
CommandFeature.send_command //发送命令
```

## 代码样例：

```cmake
# 发送命令：开停采
cam.AcquisitionStart.send_command()
cam.AcquisitionStop.send_command()
```

不同类型的设备具备的属性功能也略有差别。

在附录【属性参数】中可获取相机所有的属性参数。

## 2.2.7. 导入导出相机配置参数

在接口库中有供用户调用的导入导出设备配置文件的接口代码样例：

```txt
# 导入配置相机配置参数文件
cam.import_config_file("import_config_file.txt")
```

```txt
# 导出配置相机配置参数文件
cam.export_config_file("export_config_file.txt")
```

## 2.2.8. 错误处理

当调用接口函数内部出现异常时，错误处理机制会检测并抛出不同类型异常，异常类型均继承自Exception。

一个典型的错误处理代码样例：

```python
try:
# 调用接口函数时函数内部抛异常
dev_num, dev_info_list = device_manager.update_device_list()
except Exception as exception:
    print("打印错误信息：%s" % exception)
    exit(1)
```

用户也可通过判断捕获的具体错误类型，进行分类处理：

```python
if isinstance (exception, OutOfRange):
    print("OutOfRange: %s" % exception)
elif isinstance (exception, OffLine):
    print("OffLine: %s" % exception)
else:
    print("Other Error Type %s" % exception) 
```

## 异常类型：

<table><tr><td>异常类型</td><td>意义</td></tr><tr><td>UnexpectedError</td><td>未预测</td></tr><tr><td>NotFoundTL</td><td>没找到 TL</td></tr><tr><td>NotFoundDevice</td><td>未找到设备</td></tr><tr><td>OffLine</td><td>掉线</td></tr><tr><td>InvalidParameter</td><td>参数无效</td></tr><tr><td>InvalidHandle</td><td>句柄无效</td></tr><tr><td>InvalidCall</td><td>无效回调</td></tr><tr><td>InvalidAccess</td><td>无效获取</td></tr><tr><td>NeedMoreBuffer</td><td>Buffer 不足</td></tr><tr><td>FeatureTypeError</td><td>功能码错误</td></tr><tr><td>OutOfRange</td><td>超过范围</td></tr><tr><td>NotInitApi</td><td>未初始化</td></tr><tr><td>Timeout</td><td>超时</td></tr><tr><td>ParameterTypeError</td><td>参数类型错误</td></tr></table>

## 3. 附录

## 3.1. 属性参数

## 3.1.1. 设备属性参数

<table><tr><td>属性参数</td><td>解释</td><td>属性类</td></tr><tr><td colspan="3">DeviceInformation Section</td></tr><tr><td>DeviceVendorName</td><td>厂商名称</td><td>StringFeature</td></tr><tr><td>DeviceModelName</td><td>设备型号</td><td>StringFeature</td></tr><tr><td>DeviceFirmwareVersion</td><td>设备固件版本</td><td>StringFeature</td></tr><tr><td>DeviceVersion</td><td>设备版本</td><td>StringFeature</td></tr><tr><td>DeviceSerialNumber</td><td>设备序列号</td><td>StringFeature</td></tr><tr><td>FactorySettingVersion</td><td>出厂参数版本</td><td>StringFeature</td></tr><tr><td>DeviceUserID</td><td>用户自定义名称</td><td>StringFeature</td></tr><tr><td>DeviceLinkSelector</td><td>设备链路选择,详见C软件开发说明书</td><td>IntFeature</td></tr><tr><td>DeviceLinkThroughputLimit</td><td>设备链路带宽限制</td><td>IntFeature</td></tr><tr><td>DeviceLinkThroughputLimitMode</td><td>设备带宽限制模式,详见GxSwitchEntry</td><td>EnumFeature</td></tr><tr><td>DeviceLinkCurrentThroughput</td><td>当前设备采集带宽</td><td>IntFeature</td></tr><tr><td>DeviceReset</td><td>设备复位</td><td>CommandFeature</td></tr><tr><td>TimestampTickFrequency</td><td>时间戳时钟频率</td><td>IntFeature</td></tr><tr><td>TimestampLatch</td><td>时间戳锁存</td><td>CommandFeature</td></tr><tr><td>TimestampReset</td><td>重置时间戳</td><td>CommandFeature</td></tr><tr><td>TimestampLatchReset</td><td>重置时间戳锁存</td><td>CommandFeature</td></tr><tr><td>TimestampLatchValue</td><td>时间戳锁存值</td><td>IntFeature</td></tr><tr><td>DevicePHYVersion</td><td>设备网络芯片版本</td><td>StringFeature</td></tr><tr><td>DeviceTemperatureSelector</td><td>设备温度选择,详见GxDeviceTemperatureSelectorEntry</td><td>EnumFeature</td></tr><tr><td>DeviceTemperature</td><td>设备温度</td><td>FloatFeature</td></tr><tr><td colspan="3">ImageFormat Section</td></tr><tr><td>SensorWidth</td><td>传感器宽度</td><td>IntFeature</td></tr><tr><td>SensorHeight</td><td>传感器高度</td><td>IntFeature</td></tr><tr><td>WidthMax</td><td>最大宽度</td><td>IntFeature</td></tr><tr><td>HeightMax</td><td>最大高度</td><td>IntFeature</td></tr><tr><td>OffsetX</td><td>水平偏移</td><td>IntFeature</td></tr><tr><td>OffsetY</td><td>垂直偏移</td><td>IntFeature</td></tr><tr><td>Width</td><td>图像宽度</td><td>IntFeature</td></tr><tr><td>Height</td><td>图像高度</td><td>IntFeature</td></tr><tr><td>BinningHorizontal</td><td>水平像素 Binning</td><td>IntFeature</td></tr><tr><td>BinningVertical</td><td>垂直像素 Binning</td><td>IntFeature</td></tr><tr><td>DecimationHorizontal</td><td>水平像素抽样</td><td>IntFeature</td></tr><tr><td>DecimationVertical</td><td>垂直像素抽样</td><td>IntFeature</td></tr><tr><td>PixelSize</td><td>像素位深,详见 GxPixelSizeEntry</td><td>EnumFeature</td></tr><tr><td>PixelColorFilter</td><td>Bayer 格式,详见 GxPixelColorFilterEntry</td><td>EnumFeature</td></tr><tr><td>PixelFormat</td><td>像素格式,详见 GxPixelFormatEntry</td><td>EnumFeature</td></tr><tr><td>ReverseX</td><td>水平翻转</td><td>BoolFeature</td></tr><tr><td>ReverseY</td><td>垂直翻转</td><td>BoolFeature</td></tr><tr><td>TestPattern</td><td>测试图,详见 GxTestPatternEntry</td><td>EnumFeature</td></tr><tr><td>TestPatternGeneratorSelector</td><td>测试图源选择,详见 C 软件开发说明书和 GxTestPatternGeneratorSelectorEntry</td><td>EnumFeature</td></tr><tr><td>RegionSendMode</td><td>ROI 输出模式,详见 GxRegionSendModeEntry</td><td>EnumFeature</td></tr><tr><td>RegionMode</td><td>区域开关,详见 GxSwitchEntry</td><td>EnumFeature</td></tr><tr><td>RegionSelector</td><td>区域选择,详见 C 软件开发说明书和 GxRegionSelectorEntry</td><td>EnumFeature</td></tr><tr><td>CenterWidth</td><td>窗口宽度</td><td>IntFeature</td></tr><tr><td>CenterHeight</td><td>窗口高度</td><td>IntFeature</td></tr><tr><td>BinningHorizontalMode</td><td>水平像素 Binning 模式,详见 GxBinningHorizontalModeEntry</td><td>EnumFeature</td></tr><tr><td>BinningVerticalMode</td><td>垂直像素 Binning 模式,详见 GxBinningVerticalModeEntry</td><td>EnumFeature</td></tr><tr><td>SensorShutterMode</td><td>Sensor 曝光时间模式,详见 GxSensorShutterModeEntry</td><td>EnumFeature</td></tr><tr><td colspan="3">TransportLayer Section</td></tr><tr><td>PayloadSize</td><td>数据大小</td><td>IntFeature</td></tr><tr><td>CenterWidth</td><td>窗口宽度</td><td>IntFeature</td></tr><tr><td>CenterHeight</td><td>窗口高度</td><td>IntFeature</td></tr><tr><td>GevCurrentIPConfigurationLLA</td><td>LLA 方式配置 IP</td><td>BoolFeature</td></tr><tr><td>GevCurrentIPConfigurationDHCP</td><td>DHCP 方式配置 IP</td><td>BoolFeature</td></tr><tr><td>GevCurrentIPConfigurationPersistentIP</td><td>永久 IP 方式配置 IP</td><td>BoolFeature</td></tr><tr><td>EstimatedBandwidth</td><td>预估带宽</td><td>IntFeature</td></tr><tr><td>GevHeartbeatTimeout</td><td>心跳超时时间</td><td>IntFeature</td></tr><tr><td>GevSCPSPacketSize</td><td>流通道包长</td><td>IntFeature</td></tr><tr><td>GevSCPD</td><td>流通道包间隔</td><td>IntFeature</td></tr><tr><td>GevLinkSpeed</td><td>连接速度</td><td>IntFeature</td></tr><tr><td colspan="3">DigitalIO Section</td></tr><tr><td>UserOutputSelector</td><td>用户自定义输出选择,详见C软件开发说明书和GxUserOutputSelectorEntry</td><td>EnumFeature</td></tr><tr><td>UserOutputValue</td><td>用户自定义输出值</td><td>BoolFeature</td></tr><tr><td>UserOutputMode</td><td>用户IO输出模式,详见GxUserOutputModeEntry</td><td>EnumFeature</td></tr><tr><td>StrobeSwitch</td><td>闪光灯开关,详见GxSwitchEntry</td><td>EnumFeature</td></tr><tr><td>LineSelector</td><td>引脚选择,详见C软件开发说明书和GxLineSelectorEntry</td><td>EnumFeature</td></tr><tr><td>LineMode</td><td>引脚方向,详见GxLineModeEntry</td><td>EnumFeature</td></tr><tr><td>LineSource</td><td>引脚输出源,详见GxLineSourceEntry</td><td>EnumFeature</td></tr><tr><td>LineInverter</td><td>引脚电平反转</td><td>BoolFeature</td></tr><tr><td>LineStatus</td><td>引脚状态</td><td>BoolFeature</td></tr><tr><td>LineStatusAll</td><td>所有引脚的状态</td><td>IntFeature</td></tr><tr><td colspan="3">AnalogControls Section</td></tr><tr><td>GainAuto</td><td>自动增益,详见GxAutoEntry</td><td>EnumFeature</td></tr><tr><td>GainSelector</td><td>增益通道选择,详见C软件开发说明书和GxGainSelectorEntry</td><td>EnumFeature</td></tr><tr><td>BlackLevelAuto</td><td>自动黑电平,详见GxAutoEntry</td><td>EnumFeature</td></tr><tr><td>BlackLevelSelector</td><td>黑电平通道选择,详见C软件开发说明书和GxBlackLevelSelectEntry</td><td>EnumFeature</td></tr><tr><td>BalanceWhiteAuto</td><td>自动白平衡,详见GxAutoEntry</td><td>EnumFeature</td></tr><tr><td>BalanceRatioSelector</td><td>白平衡通道选择,详见C软件开发说明书和GxBalanceRatioSelectorEntry</td><td>EnumFeature</td></tr><tr><td>BalanceRatio</td><td>白平衡系数</td><td>FloatFeature</td></tr><tr><td>DeadPixelCorrect</td><td>坏点校正,详见GxSwitchEntry</td><td>EnumFeature</td></tr><tr><td>Gain</td><td>增益</td><td>FloatFeature</td></tr><tr><td>BlackLevel</td><td>黑电平</td><td>FloatFeature</td></tr><tr><td>GammaEnable</td><td>Gamma使能</td><td>BoolFeature</td></tr><tr><td>GammaMode</td><td>Gamma模式,详见GxGammaModeEntry</td><td>EnumFeature</td></tr><tr><td>Gamma</td><td>Gamma</td><td>FloatFeature</td></tr><tr><td>DigitalShift</td><td>数字移位</td><td>IntFeature</td></tr><tr><td>LightSourcePreset</td><td>环境光源预设,详见GxLightSourcePresetEntry</td><td>EnumFeature</td></tr><tr><td colspan="3">CustomFeature Section</td></tr><tr><td>ADCLevel</td><td>AD 转换级别</td><td>IntFeature</td></tr><tr><td>HBlanking</td><td>水平消隐</td><td>IntFeature</td></tr><tr><td>VBlanking</td><td>垂直消隐</td><td>IntFeature</td></tr><tr><td>UserPassword</td><td>用户加密区密码</td><td>StringFeature</td></tr><tr><td>VerifyPassword</td><td>用户加密区校验密码</td><td>StringFeature</td></tr><tr><td>UserData</td><td>用户加密区内容</td><td>BufferFeature</td></tr><tr><td>ExpectedGrayValue</td><td>期望灰度值</td><td>IntFeature</td></tr><tr><td>AALightEnvironment</td><td>自动曝光、自动增益,光照环境类型,详见 GxAALightEnvironmentEntry</td><td>EnumFeature</td></tr><tr><td>ImageGrayRaiseSwitch</td><td>图像亮度拉伸开关,详见 GxSwitchEntry</td><td>EnumFeature</td></tr><tr><td>AAROIOffsetX</td><td>自动调节感兴趣区域 X 坐标</td><td>IntFeature</td></tr><tr><td>AAROIOffsetY</td><td>自动调节感兴趣区域 Y 坐标</td><td>IntFeature</td></tr><tr><td>AAROIWidth</td><td>自动调节感兴趣区域宽度</td><td>IntFeature</td></tr><tr><td>AAROIHeight</td><td>自动调节感兴趣区域高度</td><td>IntFeature</td></tr><tr><td>AutoGainMin</td><td>自动增益最小值</td><td>FloatFeature</td></tr><tr><td>AutoGainMax</td><td>自动增益最大值</td><td>FloatFeature</td></tr><tr><td>AutoExposureTimeMin</td><td>自动曝光最小值</td><td>FloatFeature</td></tr><tr><td>AutoExposureTimeMax</td><td>自动曝光最大值</td><td>FloatFeature</td></tr><tr><td>ContrastParam</td><td>对比度参数</td><td>IntFeature</td></tr><tr><td>ColorCorrectionParam</td><td>颜色校正系数</td><td>IntFeature</td></tr><tr><td>AWBROIOffsetX</td><td>自动白平衡感兴趣区域 X 坐标</td><td>IntFeature</td></tr><tr><td>AWBROIOffsetY</td><td>自动白平衡感兴趣区域 Y 坐标</td><td>IntFeature</td></tr><tr><td>AWBROIWidth</td><td>自动白平衡感兴趣区域宽度</td><td>IntFeature</td></tr><tr><td>AWBROIHeight</td><td>自动白平衡感兴趣区域高度</td><td>IntFeature</td></tr><tr><td>GammaParam</td><td>伽马参数</td><td>FloatFeature</td></tr><tr><td>AWBLampHouse</td><td>自动白平衡光源,详见GxAWBLampHouseEntry</td><td>EnumFeature</td></tr><tr><td>SharpnessMode</td><td>锐化模式,详见 GxSwitchEntry</td><td>EnumFeature</td></tr><tr><td>Sharpness</td><td>锐度</td><td>FloatFeature</td></tr><tr><td>FrameInformation</td><td>图像帧信息</td><td>BufferFeature</td></tr><tr><td>DataFieldSelector</td><td>用户选择 Flash 数据区域,详见GxUserDataFieldSelectorEntry</td><td>EnumFeature</td></tr><tr><td>DataFieldValue</td><td>用户区内容</td><td>BufferFeature</td></tr><tr><td>FlatFieldCorrection</td><td>平场校正开关,详见GxSwitchEntry</td><td>EnumFeature</td></tr><tr><td>NoiseReductionMode</td><td>降噪开关,详见GxSwitchEntry</td><td>EnumFeature</td></tr><tr><td>NoiseReduction</td><td>降噪</td><td>FloatFeature</td></tr><tr><td>FFCLoad</td><td>获取平场校正参数</td><td>BufferFeature</td></tr><tr><td>FFCSave</td><td>设置平场校正参数</td><td>BufferFeature</td></tr><tr><td>StaticDefectCorrection</td><td>静态坏点校正开关,详见GxSwitchEntry</td><td>EnumFeature</td></tr><tr><td colspan="3">UserSetControl Section</td></tr><tr><td>UserSetLoad</td><td>加载参数组</td><td>CommandFeature</td></tr><tr><td>UserSetSave</td><td>保存参数组</td><td>CommandFeature</td></tr><tr><td>UserSetSelector</td><td>参数组选择,详见C软件开发说明书和GxUserSetEntry</td><td>EnumFeature</td></tr><tr><td>UserSetDefault</td><td>启动参数组,详见GxUserSetEntry</td><td>EnumFeature</td></tr><tr><td colspan="3">Event Section</td></tr><tr><td>EventSelector</td><td>事件源选择,详见GxEventSelectorEntry</td><td>EnumFeature</td></tr><tr><td>EventNotification</td><td>事件使能开关,详见GxSwitchEntry</td><td>EnumFeature</td></tr><tr><td>EventExposureEnd</td><td>曝光结束事件ID</td><td>IntFeature</td></tr><tr><td>EventExposureEndTimestamp</td><td>曝光结束事件时间戳</td><td>IntFeature</td></tr><tr><td>EventExposureEndFrameID</td><td>曝光结束事件帧ID</td><td>IntFeature</td></tr><tr><td>EventBlockDiscard</td><td>数据块丢失事件ID</td><td>IntFeature</td></tr><tr><td>EventBlockDiscardTimestamp</td><td>数据块丢失事件时间戳</td><td>IntFeature</td></tr><tr><td>EventOverrun</td><td>事件队列溢出事件ID</td><td>IntFeature</td></tr><tr><td>EventOverrunTimestamp</td><td>事件队列溢出事件时间戳</td><td>IntFeature</td></tr><tr><td>EventFrameStartOvertrigger</td><td>触发信号被屏蔽事件ID</td><td>IntFeature</td></tr><tr><td>EventFrameStartOvertriggerTimestamp</td><td>触发信号被屏蔽事件时间戳</td><td>IntFeature</td></tr><tr><td>EventBlockNotEmpty</td><td>帧存不为空事件ID</td><td>IntFeature</td></tr><tr><td>EventBlockNotEmptyTimestamp</td><td>帧存不为空事件时间戳</td><td>IntFeature</td></tr><tr><td>EventInternalError</td><td>内部错误事件ID</td><td>IntFeature</td></tr><tr><td>EventInternalErrorTimestamp</td><td>内部错误事件时间戳</td><td>IntFeature</td></tr><tr><td>EventFrameBurstStartOvertrigger</td><td>多帧触发屏蔽事件ID</td><td>IntFeature</td></tr><tr><td>EventFrameBurstStartOvertriggerFrameID</td><td>多帧触发屏蔽事件帧ID</td><td>IntFeature</td></tr><tr><td>EventFrameBurstStartOvertriggerTimestamp</td><td>多帧触发屏蔽事件时间戳</td><td>IntFeature</td></tr><tr><td>EventFrameStartWait</td><td>帧等待事件ID</td><td>IntFeature</td></tr><tr><td>EventFrameStartWaitTimestamp</td><td>帧等待事件时间戳</td><td>IntFeature</td></tr><tr><td>EventFrameBurstStartWait</td><td>多帧等待事件 ID</td><td>IntFeature</td></tr><tr><td>EventFrameBurstStartWaitTimestamp</td><td>多帧等待事件时间戳</td><td>IntFeature</td></tr><tr><td>EventBlockDiscardFrameID</td><td>数据块丢失事件帧 ID</td><td>IntFeature</td></tr><tr><td>EventFrameStartOvertriggerFrameID</td><td>触发信号被屏蔽事件帧 ID</td><td>IntFeature</td></tr><tr><td>EventBlockNotEmptyFrameID</td><td>帧存不为空事件帧 ID</td><td>IntFeature</td></tr><tr><td>EventFrameStartWaitFrameID</td><td>帧等待事件帧 ID</td><td>IntFeature</td></tr><tr><td>EventFrameBurstStartWaitFrameID</td><td>多帧等待事件帧 ID</td><td>IntFeature</td></tr><tr><td colspan="3">LUT Section</td></tr><tr><td>LUTValueAll</td><td>查找表内容</td><td>BufferFeature</td></tr><tr><td>LUTSelector</td><td>查找表选择,详见 C 软件开发说明书和 GxLutSelectorEntry</td><td>EnumFeature</td></tr><tr><td>LUTEnable</td><td>查找表使能</td><td>BoolFeature</td></tr><tr><td>LUTIndex</td><td>查找表索引</td><td>IntFeature</td></tr><tr><td>LUTValue</td><td>查找表值</td><td>IntFeature</td></tr><tr><td colspan="3">Color Transformation Control</td></tr><tr><td>ColorTransformationMode</td><td>颜色转换模式,详见 GxColorTransformationModeEntry</td><td>EnumFeature</td></tr><tr><td>ColorTransformationEnable</td><td>颜色转换使能</td><td>BoolFeature</td></tr><tr><td>ColorTransformationValueSelector</td><td>颜色转换矩阵元素选择,详见 GxColorTransformationValueSelectorEntry</td><td>EnumFeature</td></tr><tr><td>ColorTransformationValue</td><td>颜色转换矩阵元素</td><td>FloatFeature</td></tr><tr><td>SaturationMode</td><td>饱和度模式开关,详见 GxSwitchEntry</td><td>EnumFeature</td></tr><tr><td>Saturation</td><td>饱和度</td><td>IntFeature</td></tr><tr><td colspan="3">ChunkData Section</td></tr><tr><td>ChunkModeActive</td><td>帧信息使能</td><td>BoolFeature</td></tr><tr><td>ChunkEnable</td><td>单项帧信息使能</td><td>BoolFeature</td></tr><tr><td>ChunkSelector</td><td>帧信息项选择,详见 C 软件开发说明书和 GxChunkSelectorEntry</td><td>EnumFeature</td></tr><tr><td colspan="3">Device Feature</td></tr><tr><td>DeviceCommandTimeout</td><td>命令超时</td><td>IntFeature</td></tr><tr><td>DeviceCommandRetryCount</td><td>命令重试次数</td><td>IntFeature</td></tr><tr><td colspan="3">AcquisitionTrigger Section</td></tr><tr><td>FrameBufferOverwriteActive</td><td>帧存覆盖使能</td><td>BoolFeature</td></tr><tr><td>AcquisitionStart</td><td>开始采集</td><td>CommandFeature</td></tr><tr><td>AcquisitionStop</td><td>停止采集</td><td>CommandFeature</td></tr><tr><td>TriggerSoftware</td><td>软触发</td><td>CommandFeature</td></tr><tr><td>TransferStart</td><td>开始传输</td><td>CommandFeature</td></tr><tr><td>AcquisitionMode</td><td>采集模式,详见GxAcquisitionModeEntry</td><td>EnumFeature</td></tr><tr><td>TriggerMode</td><td>触发模式,详见GxSwitchEntry</td><td>EnumFeature</td></tr><tr><td>TriggerActivation</td><td>触发极性,详见GxTriggerActivationEntry</td><td>EnumFeature</td></tr><tr><td>ExposureAuto</td><td>自动曝光,详见GxAutoEntry</td><td>EnumFeature</td></tr><tr><td>TriggerSource</td><td>触发源,详见GxTriggerSourceEntry</td><td>EnumFeature</td></tr><tr><td>ExposureMode</td><td>曝光模式,详见GxExposureModeEntry</td><td>EnumFeature</td></tr><tr><td>TriggerSelector</td><td>触发类型选择,详见C软件开发说明书和GxTriggerSelectorEntry</td><td>EnumFeature</td></tr><tr><td>TransferControlMode</td><td>传输控制模式,详见GxTransferControlModeEntry</td><td>EnumFeature</td></tr><tr><td>TransferOperationMode</td><td>传输操作模式,详见GxTransferOperationModeEntry</td><td>EnumFeature</td></tr><tr><td>AcquisitionFrameRateMode</td><td>采集帧率调节模式,详见GxSwitchEntry</td><td>EnumFeature</td></tr><tr><td>FixedPatternNoiseCorrectMode</td><td>模板噪声校正,详见GxSwitchEntry</td><td>EnumFeature</td></tr><tr><td>ExposureTime</td><td>曝光时间</td><td>FloatFeature</td></tr><tr><td>TriggerFilterRaisingEdge</td><td>上升沿触发滤波</td><td>FloatFeature</td></tr><tr><td>TriggerFilterFallingEdge</td><td>下降沿触发滤波</td><td>FloatFeature</td></tr><tr><td>TriggerDelay</td><td>触发延迟</td><td>FloatFeature</td></tr><tr><td>AcquisitionFrameRate</td><td>采集帧率</td><td>FloatFeature</td></tr><tr><td>CurrentAcquisitionFrameRate</td><td>当前采集帧率</td><td>FloatFeature</td></tr><tr><td>TransferBlockCount</td><td>传输帧数</td><td>IntFeature</td></tr><tr><td>TriggerSwitch</td><td>外触发开关,详见GxSwitchEntry</td><td>EnumFeature</td></tr><tr><td>AcquisitionSpeedLevel</td><td>采集速度级别</td><td>IntFeature</td></tr><tr><td>AcquisitionFrameCount</td><td>多帧采集帧数</td><td>IntFeature</td></tr><tr><td>AcquisitionBurstFrameCount</td><td>高速连拍帧数</td><td>IntFeature</td></tr><tr><td>AcquisitionStatusSelector</td><td>采集状态选择,详见C软件开发说明书和GxAcquisitionStatusSelectorEntry</td><td>EnumFeature</td></tr><tr><td>AcquisitionStatus</td><td>采集状态</td><td>BoolFeature</td></tr><tr><td>ExposureDelay</td><td>曝光延迟</td><td>FloatFeature</td></tr><tr><td>ExposureOverlapTimeMax</td><td>交叠曝光时间最大值</td><td>FloatFeature</td></tr><tr><td>ExposureTimeMode</td><td>曝光时间模式,详见GxExposureTimeModeEntry</td><td>EnumFeature</td></tr><tr><td colspan="3">CounterAndTimerControl Section</td></tr><tr><td>TimerSelector</td><td>计时器选择,详见GxTimerSelectorEntry</td><td>EnumFeature</td></tr><tr><td>TimerDuration</td><td>计时器持续时间</td><td>FloatFeature</td></tr><tr><td>TimerDelay</td><td>计时器延迟</td><td>FloatFeature</td></tr><tr><td>TimerTriggerSource</td><td>计时器触发源,详见GxTimerTriggerSourceEntry</td><td>EnumFeature</td></tr><tr><td>CounterSelector</td><td>计数器选择,详见GxCounterSelectorEntry</td><td>EnumFeature</td></tr><tr><td>CounterEventSource</td><td>计数器事件触发源,详见GxCounterEventSourceEntry</td><td>EnumFeature</td></tr><tr><td>CounterResetSource</td><td>计数器复位源,详见GxCounterResetSourceEntry</td><td>EnumFeature</td></tr><tr><td>CounterResetActivation</td><td>计数器复位信号极性,详见GxCounterResetActivationEntry</td><td>EnumFeature</td></tr><tr><td>CounterReset</td><td>计数器复位</td><td>CommandFeature</td></tr><tr><td>CounterTriggerSource</td><td>计数器触发源,详见GxCounterTriggerSourceEntry</td><td>EnumFeature</td></tr><tr><td>CounterDuration</td><td>计数器持续时间</td><td>IntFeature</td></tr><tr><td>TimerTriggerActivation</td><td>计时器触发极性,详见GxTimerTriggerActivationEntry</td><td>EnumFeature</td></tr><tr><td colspan="3">RemoveParameterLimitControl Section</td></tr><tr><td>RemoveParameterLimit</td><td>取消参数范围限制开关,详见GxSwitchEntry</td><td>EnumFeature</td></tr></table>

## 3.1.2. 流属性参数

<table><tr><td>属性参数</td><td>解释</td><td>属性类</td></tr><tr><td>StreamAnnouncedBufferCount</td><td>声明的 Buffer 个数</td><td>IntFeature</td></tr><tr><td>StreamDeliveredFrameCount</td><td>接收帧个数(包括残帧)</td><td>IntFeature</td></tr><tr><td>StreamLostFrameCount</td><td>buffer 不足导致的丢帧个数</td><td>IntFeature</td></tr><tr><td>StreamIncompleteFrameCount</td><td>接收的残帧个数</td><td>IntFeature</td></tr><tr><td>StreamDeliveredPacketCount</td><td>接收到的包数</td><td>IntFeature</td></tr><tr><td>StreamResendPacketCount</td><td>重传包个数</td><td>IntFeature</td></tr><tr><td>StreamRescuedPacketCount</td><td>重传成功包个数</td><td>IntFeature</td></tr><tr><td>StreamResendCommandCount</td><td>重传命令次数</td><td>IntFeature</td></tr><tr><td>StreamUnexpectedPacketCount</td><td>异常包个数</td><td>IntFeature</td></tr><tr><td>MaxPacketCountInOneBlock</td><td>数据块最大重传包数</td><td>IntFeature</td></tr><tr><td>MaxPacketCountInOneCommand</td><td>一次重传命令最大包含的包数</td><td>IntFeature</td></tr><tr><td>ResendTimeout</td><td>重传超时时间</td><td>IntFeature</td></tr><tr><td>MaxWaitPacketCount</td><td>最大等待包数</td><td>IntFeature</td></tr><tr><td>ResendMode</td><td>重传模式,详见 GxSwitchEntry</td><td>EnumFeature</td></tr><tr><td>StreamMissingBlockIDCount</td><td>BlockID 丢失个数</td><td>IntFeature</td></tr><tr><td>BlockTimeout</td><td>数据块超时时间</td><td>IntFeature</td></tr><tr><td>MaxNumQueueBuffer</td><td>采集队列最大 Buffer 个数</td><td>IntFeature</td></tr><tr><td>PacketTimeout</td><td>包超时时间</td><td>IntFeature</td></tr><tr><td>StreamTransferSize</td><td>传输数据块大小</td><td>IntFeature</td></tr><tr><td>StreamTransferNumberUrb</td><td>传输数据块数量</td><td>IntFeature</td></tr><tr><td>SocketBufferSize</td><td>套接字缓冲区大小</td><td>IntFeature</td></tr><tr><td>StopAcquisitionMode</td><td>停采模式。详见GxStopAcquisitionModeEntry</td><td>EnumFeature</td></tr><tr><td>StreamBufferHandlingMode</td><td>Buffer 处理模式,详见GxDSStreamBufferHandlingModeEntry</td><td>EnumFeature</td></tr></table>

## 3.2. 功能类定义

## 3.2.1. Feature

负责查看各种数据类型功能的基础功能，判断其是否已实现、可读、可写。

Feature类是IntFeature/FloatFeature/EnumFeature/BoolFeature/StringFeature/BufferFeature/CommandFeature属性类的父类。

接口列表：

<table><tr><td>is_implemented()</td><td>判断属性参数是否已实现</td></tr><tr><td>is_readable()</td><td>判断属性参数是否可读</td></tr><tr><td>is_writable()</td><td>判断属性参数是否可写</td></tr></table>

##  接口说明

##  is_implemented

声明：

Feature.is_implemented() 

意义：

判断属性参数是否已实现

返回值：

Ture：实现

False：未实现

异常处理：

1) 如果属性参数是无效参数，则返回False。

2) 因为其他原因导致的获取属性参数是否实现失败，则抛出异常，异常类型详见错误处理。

 is_readable 

声明：Feature.is_readable()

意义：

判断属性参数是否可读

返回值：

Ture：可读

False：不可读

异常处理：

1) 如果功能未实现，则函数返回False。

2) 如果获取属性参数是否可读失败，则抛出异常，异常类型详见错误处理

```txt
is_writable 
```

声明：

意义：

判断属性参数是否可写。

返回值：

Ture：可写

False：不可写

异常处理：

1) 如果功能未实现，则返回False。

2) 如果获取属性参数是否可写失败，则抛出异常，异常类型详见错误处理

## 3.2.2. IntFeature

负责查看、控制相机的整型值功能，继承自Feature 类。

接口列表：

<table><tr><td>is_implemented()</td><td>判断整型属性参数是否已实现</td></tr><tr><td>is_readable()</td><td>判断整型属性参数是否可读</td></tr><tr><td>is_writable()</td><td>判断整型属性参数是否可写</td></tr><tr><td>get_range()</td><td>获取整型属性参数范围字典</td></tr><tr><td>get()</td><td>读取整型属性参数值</td></tr><tr><td>set(int_value)</td><td>设置整型属性参数值</td></tr></table>

##  接口说明

```txt
> set
声明：
IntFeature.set(self, int_value)
```

```javascript
声明：IntFeature.get()
```

 is_implemented详见 Feature::is_implemented()。

 is_readable详见 Feature::is_readable()。

 is_writable详见 Feature::is_writable()。

 get_range 

声明：IntFeature.get_range()

意义：

获取整型属性参数范围字典

返回值：

记录整型属性参数范围字典。键包含：min最小值，max最大值，step 步长

异常处理：

1) 如果该整型属性参数功能未实现，则打印不支持该整型属性参数获取范围的信息，函数返回None。

2) 如果获取该整型属性参数范围不成功，则抛出异常，异常类型详见错误处理。

 get 

意义：

读取整型属性参数值

返回值：

获取的整型值

异常处理：

1) 如果该整型属性参数功能未实现或不可读，则打印该整型属性参数不可读的信息，函数返回 None。

2) 如果获取该整型属性参数值不成功，则抛出异常，异常类型详见错误处理

意义：

设置整型属性参数值

形参：

设置的整型数值

异常处理：

1) 如果输入参数不是整型值，则抛出 ParameterTypeError 异常。

2) 如果该整型属性参数功能未实现或不可写，则打印该整型属性参数不可写的信息，函数返回 None。

3) 如果输入参数不在该整型属性参数的范围内，则打印超过该整型属性参数范围的信息并打印范围，函数返回None。

4) 如果设置该整型属性参数不成功，则抛出异常，异常类型详见错误处理。

## 3.2.3. FloatFeature

负责查看、控制相机的浮点型值功能，继承自Feature类。

接口列表：

<table><tr><td>is_implemented()</td><td>判断浮点型属性参数是否已实现</td></tr><tr><td>is_readable()</td><td>判断浮点型属性参数是否可读</td></tr><tr><td>is_writable()</td><td>判断浮点型属性参数是否可写</td></tr><tr><td>get_range()</td><td>获取浮点型属性参数范围字典</td></tr><tr><td>get()</td><td>读取浮点型属性参数值</td></tr><tr><td>set(float_value)</td><td>设置浮点型属性参数值</td></tr></table>

##  接口说明

##  is_implemented

详见 Feature::is_implemented()。

 is_readable 

详见 Feature::is_readable()。

 is_writable 

详见 Feature::is_writable()。

 get_range 

声明：

FloatFeature.get_range() 

意义：

获取浮点型属性参数范围字典

返回值：

记录浮点型属性参数值范围的字典。键包含：min最小值，max 最大值，inc步长，unit单位，inc_is_valid单位是否有效

异常处理：

1) 如果该浮点型属性参数功能未实现，则打印不支持该浮点型属性参数获取范围的信息，函数返回None。

2) 如果获取该浮点型属性参数范围不成功，则抛出异常，异常类型详见错误处理

get 

声明：

FloatFeature.get() 

意义：

读取浮点型属性参数值

返回值：

获取的浮点型属性参数值

异常处理：

1) 如果该浮点型属性参数未实现或不可读，则打印该浮点型属性参数不可读的信息，函数返回 None。

2) 如果获取浮点型属性参数不成功，则抛出异常，异常类型详见错误处理。

set 

声明：

FloatFeature.set(float_value) 

意义：

设置浮点型属性参数值

形参：

[in]float_value 设置的浮点型数值

异常处理：

1) 如果输入参数不是浮点型值，则抛出 ParameterTypeError 异常。

2) 如果该浮点型属性参数功能未实现或不可写，则打印该浮点型属性参数不可写的信息，函数返回None。

3) 如果输入参数不在该浮点型属性参数的范围内，则打印超过该浮点型属性参数范围的信息并打印范围，函数返回None。

4) 如果设置该浮点型属性参数不成功，则抛出异常，异常类型详见错误处理

## 3.2.4. EnumFeature

负责查看、控制相机的枚举型值功能，继承自Feature类。

接口列表：

<table><tr><td>is_implemented()</td><td>判断枚举型属性参数是否已实现</td></tr><tr><td>is_readable()</td><td>判断枚举型属性参数是否可读</td></tr><tr><td>is_writable()</td><td>判断枚举型属性参数是否可写</td></tr><tr><td>get_range()</td><td>获取枚举型属性参数范围字典</td></tr><tr><td>get()</td><td>读取枚举型属性参数的值和字符串</td></tr><tr><td>set(enum_value)</td><td>设置枚举型属性参数数值</td></tr></table>

##  接口说明

 is_implemented 

详见 Feature::is_implemented()。

 is_readable 

详见 Feature::is_readable()。

 is_writable 

详见 Feature::is_writable()。

 get_range 

声明：

EnumFeature.get_range() 

意义：

获取枚举型属性参数范围字典

返回值：

记录枚举型属性参数范围的字典

异常处理：

1) 如果该枚举型属性参数功能未实现，则打印不支持该枚举型属性参数获取范围的信息，函数返回None。

2) 如果获取该枚举型属性参数范围不成功，则抛出异常，异常类型详见错误处理

 get 

声明：

EnumFeature.get() 

意义：

读取枚举型属性参数的值和字符串

返回值：

1) 枚举型属性参数的数值

2) 枚举型属性参数的字符串

异常处理：

1) 如果该枚举型属性参数功能未实现或不可读，则打印该枚举型属性参数不可读的信息，函数返回None。

2) 如果获取枚举型属性参数值不成功，则抛出异常，异常类型详见错误处理。

 set 

声明：

EnumFeature.set( enum_value) 

意义：

设置枚举型属性参数值

形参：

[in]enum_value 设置的枚举型数值

异常处理：

1) 如果输入参数不是整型值，则抛出 ParameterTypeError 异常。

2) 如果该枚举型属性参数功能未实现或不可写，则打印该枚举型属性参数不可写的信息，函数返回None。

3) 如果输入参数不在枚举型属性参数“值”的范围内，则打印超过该枚举型属性参数范围的信息并打印范围，函数返回 None。

4) 如果设置该枚举型属性参数不成功，则抛出异常，异常类型详见错误处理。

## 3.2.5. BoolFeature

负责查看、控制相机的布尔型值功能，继承自Feature类。

接口列表：

<table><tr><td>is_implemented()</td><td>判断布尔型属性参数是否已实现</td></tr><tr><td>is_readable()</td><td>判断布尔型属性参数是否可读</td></tr><tr><td>is_writable()</td><td>判断布尔型属性参数是否可写</td></tr><tr><td>get()</td><td>读取布尔型属性参数值</td></tr><tr><td>set(bool_value)</td><td>设置布尔型属性参数值</td></tr></table>

##  接口说明

##  is_implemented

详见 Feature::is_implemented()。

 is_readable 

详见 Feature::is_readable()。

##  is_writable

详见 Feature::is_writable()。

 get 

声明：

BoolFeature.get() 

意义：

读取布尔型属性参数值

返回值：

获取的布尔值

异常处理：

1) 如果该布尔型属性参数功能未实现或不可读，则打印该布尔型属性参数不可读的信息，函数返回None。

2) 如果获取布尔型属性参数值不成功，则抛出异常，异常类型详见错误处理

```txt
set 
```

声明：

```txt
BoolFeature.set(bool_value) 
```

意义：

设置浮点型属性参数值

形参：

```txt
[in]bool_value 设置的布尔型数值
```

异常处理：

1) 如果输入参数不是布尔型值，则抛出 ParameterTypeError 异常。

2) 如果该布尔型属性参数功能未实现或不可写，则打印该布尔型属性参数不可写的信息，函数返回None。

3) 如果设置该布尔型属性参数不成功，则抛出异常，异常类型详见错误处理

## 3.2.6. StringFeature

负责查看、控制相机的字符串型值功能，继承自Feature类。

接口列表：

<table><tr><td>is_implemented()</td><td>判断字符串型属性参数是否已实现</td></tr><tr><td>is_readable()</td><td>判断字符串型属性参数是否可读</td></tr><tr><td>is_writable()</td><td>判断字符串型属性参数是否可写</td></tr><tr><td>get_string_max_length()</td><td>获取字符串型属性参数值可设置的最长长度</td></tr><tr><td>get()</td><td>读取字符串型属性参数值</td></tr><tr><td>set(input_string)</td><td>设置字符串型属性参数值</td></tr></table>

##  接口说明

##  is_implemented

详见 Feature::is_implemented()。

##  is_readable

详见 Feature::is_readable()。

##  is_writable

详见 Feature::is_writable()。

 get_string_max_length 

声明：

意义：

获取字符串型属性参数可设置的最大长度

返回值：

字符串型属性参数可设置的最大长度

异常处理：

1) 如果该字符串型属性参数功能未实现，则打印不支持获取该字符串型属性参数的信息，函数返回None。

2) 如果获取字符串型属性参数值可设置最大长度不成功，则抛出异常，异常类型详见错误处理。

 get 

声明：

意义：

读取字符串型属性参数值

返回值：

获取的字符串型属性参数值

异常处理：

1) 如果该字符串型属性参数功能未实现或不可读，则打印该字符串型属性参数不可读的信息，函数返回None。

2) 如果获取该字符串型属性参数值不成功，则抛出异常，异常类型详见错误处理

声明：

意义：

设置字符串型属性参数值

形参：

[in]input_string 设置的字符串型数值

异常处理：

1) 如果输入参数不是字符串型值，则抛出 ParameterTypeError 异常。

2) 如果该字符串型属性参数功能未实现或不可写，则打印该字符串型属性参数不可写的信息，函数返回 None。

3) 如果输入参数长度大于可设置最大长度，则打印超过该字符串型属性参数长度最大值的信息并打印最大值，函数返回 None。

4) 如果设置该字符串型属性参数不成功，则抛出异常，异常类型详见错误处理。

## 3.2.7. BufferFeature

负责查看、控制相机的缓冲功能，继承自 Feature 类。

接口列表：

<table><tr><td>is_implemented()</td><td>判断 Buffer 型属性参数是否已实现</td></tr><tr><td>is_readable()</td><td>判断 Buffer 型属性参数是否可读</td></tr><tr><td>is_writable ()</td><td>判断 Buffer 型属性参数是否可写</td></tr><tr><td>get_buffer_length()</td><td>获取 Buffer 型属性参数的长度</td></tr><tr><td>get_buffer()</td><td>读取 Buffer 型属性参数数据</td></tr><tr><td>set_buffer(buf)</td><td>设置 Buffer 型属性参数数据</td></tr></table>

##  接口说明

##  is_implemented

详见 Feature::is_implemented()。

##  is_readable

详见 Feature::is_readable()。

##  is_writable

详见 Feature::is_writable()。

##  get_buffer_length

声明：

BufferFeature.get_buffer_length() 

意义：

获取Buffer型属性参数的长度

返回值：

Buffer型属性参数的长度

异常处理：

1) 如果该 Buffer 型属性参数功能未实现，则打印不支持该 Buffer 属性参数获取范围的信息，函数返回 None。

2) 如果获取Buffer型属性参数长度不成功，则抛出异常，异常类型详见错误处理。

##  get_buffer

声明：

BufferFeature.get_buffer() 

意义：

读取Buffer型属性参数数据

返回值：

Buffer 对象

异常处理：

1) 如果该 Buffer 型属性参数功能未实现或不可读，则打印该 Buffer 型属性参数不可读的信息，函数返回 None。

2) 如果获取该Buffer型属性参数数据不成功，则抛出异常，异常类型详见错误处理。

```txt
set_buffer 
```

声明：

```txt
BufferFeature.set_buffer(buf) 
```

意义：

设置Buffer型属性参数数据

形参：

[in]buffer 设置的缓冲数据[Buffer 类型]

异常处理：

1) 如果输入参数不是 Buffer 类型，则抛出 ParameterTypeError 异常。

2) 如果该 Buffer 型属性参数功能未实现或不可写，则打印该 Buffer 型属性参数不可写的信息，函数返回 None。

3) 如果输入 Buffer 型属性参数数据长度大于最大长度，则打印超过该 Buffer 型属性参数最大长度的信息并打印最大值，函数返回 None。

4) 如果设置该Buffer型属性参数不成功，则抛出异常，异常类型详见错误处理。

## 3.2.8. CommandFeature

负责制相机的命令型值功能，继承自Feature类。

接口列表：

<table><tr><td>is_implemented()</td><td>判断命令型属性参数是否已实现</td></tr><tr><td>is_readable()</td><td>判断命令型属性参数是否可读</td></tr><tr><td>is_writable ()</td><td>判断命令型属性参数是否可写</td></tr><tr><td>send_command()</td><td>发送命令</td></tr></table>

##  接口说明

##  is_implemented

详见 Feature::is_implemented()。

##  is_readable

详见 Feature::is_readable()。

##  is_writable

详见 Feature::is_writable()。

##  send_command

声明：

CommandFeature.send_command() 

意义：

发送命令

异常处理：

1) 如果该 Command 型属性参数功能未实现，则打印不支持该 Command 型属性参数的信息，函数返回 None。

2) 如果发送命令不成功，则抛出异常，异常类型详见错误处理。

## 3.3. 数据类型定义


3.3.1. GxDeviceClassList


<table><tr><td>定义</td><td>值</td><td>解释</td></tr><tr><td>UNKNOWN</td><td>0</td><td>未知设备种类</td></tr><tr><td>USB2</td><td>1</td><td>USB2.0 相机</td></tr><tr><td>GEV</td><td>2</td><td>千兆网相机 (GigE Vision)</td></tr><tr><td>U3V</td><td>3</td><td>USB3.0 相机 (USB3 Vision)</td></tr></table>


3.3.2. GxAccessStatus


<table><tr><td>定义</td><td>值</td><td>解释</td></tr><tr><td>UNKNOWN</td><td>0</td><td>设备当前状态未知</td></tr><tr><td>READWRITE</td><td>1</td><td>设备当前可读可写</td></tr><tr><td>READONLY</td><td>2</td><td>设备当前仅支持读</td></tr><tr><td>NOACCESS</td><td>3</td><td>设备当前既不支持读,又不支持写</td></tr></table>


3.3.3. GxAccessMode


<table><tr><td>定义</td><td>值</td><td>解释</td></tr><tr><td>READONLY</td><td>2</td><td>以只读方式打开设备</td></tr><tr><td>CONTROL</td><td>3</td><td>以控制方式打开设备</td></tr><tr><td>EXCLUSIVE</td><td>4</td><td>以独占方式打开设备</td></tr></table>

## 3.3.4. GxPixelFormatEntry

<table><tr><td>定义</td><td>值</td><td>解释</td></tr><tr><td>UNDEFINED</td><td>0x00000000</td><td>未定义</td></tr><tr><td>MONO8</td><td>0x01080001</td><td>Monochrome 8-bit</td></tr><tr><td>MONO8_SIGNED</td><td>0x01080002</td><td>Monochrome 8-bit signed</td></tr><tr><td>MONO10</td><td>0x01100003</td><td>Monochrome 10-bit unpacked</td></tr><tr><td>MONO12</td><td>0x01100005</td><td>Monochrome 12-bit unpacked</td></tr><tr><td>MONO14</td><td>0x01100025</td><td>Monochrome 14-bit unpacked</td></tr><tr><td>MONO16</td><td>0x01100007</td><td>Monochrome 16-bit</td></tr><tr><td>BAYER_GR8</td><td>0x01080008</td><td>Bayer Green-Red 8-bit</td></tr><tr><td>BAYER_RG8</td><td>0x01080009</td><td>Bayer Red-Green 8-bit</td></tr><tr><td>BAYER_GB8</td><td>0x0108000A</td><td>Bayer Green-Blue 8-bit</td></tr><tr><td>BAYER_BG8</td><td>0x0108000B</td><td>Bayer Blue-Green 8-bit</td></tr><tr><td>BAYER_GR10</td><td>0x0110000C</td><td>Bayer Green-Red 10-bit</td></tr><tr><td>BAYER_RG10</td><td>0x0110000D</td><td>Bayer Red-Green 10-bit</td></tr><tr><td>BAYER_GB10</td><td>0x0110000E</td><td>Bayer Green-Blue 10-bit</td></tr><tr><td>BAYER_BG10</td><td>0x0110000F</td><td>Bayer Blue-Green 10-bit</td></tr><tr><td>BAYER_GR12</td><td>0x01100010</td><td>Bayer Green-Red 12-bit</td></tr><tr><td>BAYER_RG12</td><td>0x01100011</td><td>Bayer Red-Green 12-bit</td></tr><tr><td>BAYER_GB12</td><td>0x01100012</td><td>Bayer Green-Blue 12-bit</td></tr><tr><td>BAYER_BG12</td><td>0x01100013</td><td>Bayer Blue-Green 12-bit</td></tr><tr><td>BAYER_GR16</td><td>0x0110002E</td><td>Bayer Green-Red 16-bit</td></tr><tr><td>BAYER_RG16</td><td>0x0110002F</td><td>Bayer Red-Green 16-bit</td></tr><tr><td>BAYER_GB16</td><td>0x01100030</td><td>Bayer Green-Blue 16-bit</td></tr><tr><td>BAYER_BG16</td><td>0x01100031</td><td>Bayer Blue-Green 16-bit</td></tr><tr><td>RGB8_PLANAR</td><td>0x02180021</td><td>Red-Green-Blue 8-bit planar</td></tr><tr><td>RGB10_PLANAR</td><td>0x02300022</td><td>Red-Green-Blue 10-bit planar</td></tr><tr><td>RGB12_PLANAR</td><td>0x02300023</td><td>Red-Green-Blue 12-bit planar</td></tr><tr><td>RGB16_PLANAR</td><td>0x02300024</td><td>Red-Green-Blue 16-bit planar</td></tr></table>

## 3.3.5. GxFrameStatusList

<table><tr><td>定义</td><td>值</td><td>解释</td></tr><tr><td>SUCCESS</td><td>0</td><td>正常帧</td></tr><tr><td>IMCOMPLETE</td><td>-1</td><td>残帧</td></tr></table>

## 3.3.6. GxDeviceTemperatureSelectorEntry

<table><tr><td>定义</td><td>值</td><td>解释</td></tr><tr><td>SENSOR</td><td>1</td><td>传感器温度</td></tr><tr><td>MAINBOARD</td><td>2</td><td>主板温度</td></tr></table>

## 3.3.7. GxPixelSizeEntry

<table><tr><td>定义</td><td>值</td><td>解释</td></tr><tr><td>BPP8</td><td>8</td><td>像素大小 BPP8</td></tr><tr><td>BPP10</td><td>10</td><td>像素大小 BPP10</td></tr><tr><td>BPP12</td><td>12</td><td>像素大小 BPP12</td></tr><tr><td>BPP16</td><td>16</td><td>像素大小 BPP16</td></tr><tr><td>BPP24</td><td>24</td><td>像素大小 BPP24</td></tr><tr><td>BPP30</td><td>30</td><td>像素大小 BPP30</td></tr><tr><td>BPP32</td><td>32</td><td>像素大小 BPP32</td></tr><tr><td>BPP36</td><td>36</td><td>像素大小 BPP36</td></tr><tr><td>BPP48</td><td>48</td><td>像素大小 BPP48</td></tr><tr><td>BPP64</td><td>64</td><td>像素大小 BPP64</td></tr></table>

## 3.3.8. GxPixelColorFilterEntry

<table><tr><td>定义</td><td>值</td><td>解释</td></tr><tr><td>NONE</td><td>0</td><td>无</td></tr><tr><td>BAYER_RG</td><td>1</td><td>RG 格式</td></tr><tr><td>BAYER_GB</td><td>2</td><td>GB 格式</td></tr><tr><td>BAYER_GR</td><td>3</td><td>GR 格式</td></tr><tr><td>BAYER_BG</td><td>4</td><td>BG 格式</td></tr></table>

## 3.3.9. GxAcquisitionModeEntry

<table><tr><td>定义</td><td>值</td><td>解释</td></tr><tr><td>SINGLE_FRAME</td><td>0</td><td>单帧模式</td></tr><tr><td>MULITI_FRAME</td><td>1</td><td>多帧模式</td></tr><tr><td>CONTINUOUS</td><td>2</td><td>连续模式</td></tr></table>

## 3.3.10. GxTriggerSourceEntry

<table><tr><td>定义</td><td>值</td><td>解释</td></tr><tr><td>SOFTWARE</td><td>0</td><td>软触发</td></tr><tr><td>LINE0</td><td>1</td><td>触发源 0</td></tr><tr><td>LINE1</td><td>2</td><td>触发源 1</td></tr><tr><td>LINE2</td><td>3</td><td>触发源 2</td></tr><tr><td>LINE3</td><td>4</td><td>触发源 3</td></tr><tr><td>COUNTER2END</td><td>5</td><td>COUNTER2END 触发信号</td></tr></table>

## 3.3.11. GxTriggerActivationEntry

<table><tr><td>定义</td><td>值</td><td>解释</td></tr><tr><td>FALLING_EDGE</td><td>0</td><td>下降沿触发</td></tr><tr><td>RISING_EDGE</td><td>1</td><td>上升沿触发</td></tr></table>

## 3.3.12. GxExposureModeEntry

<table><tr><td>定义</td><td>值</td><td>解释</td></tr><tr><td>TIMED</td><td>1</td><td>曝光时间寄存器控制曝光时间</td></tr><tr><td>TRIGGER_WIDTH</td><td>2</td><td>触发信号宽度控制曝光时间</td></tr></table>

## 3.3.13. GxUserOutputSelectorEntry

<table><tr><td>定义</td><td>值</td><td>解释</td></tr><tr><td>OUTPUT0</td><td>1</td><td>输出 0</td></tr><tr><td>OUTPUT1</td><td>2</td><td>输出 1</td></tr><tr><td>OUTPUT2</td><td>4</td><td>输出 2</td></tr></table>

## 3.3.14. GxUserOutputModeEntry

<table><tr><td>定义</td><td>值</td><td>解释</td></tr><tr><td>STROBE</td><td>0</td><td>闪光灯</td></tr><tr><td>USER_DEFINED</td><td>1</td><td>用户自定义</td></tr></table>


3.3.15. GxGainSelectorEntry


<table><tr><td>定义</td><td>值</td><td>解释</td></tr><tr><td>ALL</td><td>0</td><td>所有增益通道</td></tr><tr><td>RED</td><td>1</td><td>红通道增益</td></tr><tr><td>GREEN</td><td>2</td><td>绿通道增益</td></tr><tr><td>BLUE</td><td>3</td><td>蓝通道增益</td></tr></table>

## 3.3.16. GxBlackLevelSelectEntry

<table><tr><td>定义</td><td>值</td><td>解释</td></tr><tr><td>ALL</td><td>0</td><td>所有黑电平通道</td></tr><tr><td>RED</td><td>1</td><td>红通道黑电平</td></tr><tr><td>GREEN</td><td>2</td><td>绿通道黑电平</td></tr><tr><td>BLUE</td><td>3</td><td>蓝通道黑电平</td></tr></table>


3.3.17. GxBalanceRatioSelectorEntry


<table><tr><td>定义</td><td>值</td><td>解释</td></tr><tr><td>RED</td><td>0</td><td>红通道</td></tr><tr><td>GREEN</td><td>1</td><td>绿通道</td></tr><tr><td>BLUE</td><td>2</td><td>蓝通道</td></tr></table>

## 3.3.18. GxAALightEnvironmentEntry

<table><tr><td>定义</td><td>值</td><td>解释</td></tr><tr><td>NATURE_LIGHT</td><td>0</td><td>自然光</td></tr><tr><td>AC50HZ</td><td>1</td><td>50 赫兹日光灯</td></tr><tr><td>AC60HZ</td><td>2</td><td>60 赫兹日光灯</td></tr></table>

## 3.3.19. GxUserSetEntry

<table><tr><td>定义</td><td>值</td><td>解释</td></tr><tr><td>DEFAULT</td><td>0</td><td>默认参数组</td></tr><tr><td>USER_SET0</td><td>1</td><td>用户参数组 0</td></tr></table>

## 3.3.20. GxAWBLampHouseEntry

<table><tr><td>定义</td><td>值</td><td>解释</td></tr><tr><td>ADAPTIVE</td><td>0</td><td>自适应光源</td></tr><tr><td>D65</td><td>1</td><td>指定色温 6500k</td></tr><tr><td>FLUORESCENCE</td><td>2</td><td>指定荧光灯</td></tr><tr><td>INCANDESCENT</td><td>3</td><td>指定白炽灯</td></tr><tr><td>D75</td><td>4</td><td>指定色温 7500k</td></tr><tr><td>D50</td><td>5</td><td>指定色温 5000k</td></tr><tr><td>U30</td><td>6</td><td>指定色温 3000k</td></tr></table>

## 3.3.21. GxUserDataFieldSelectorEntry

<table><tr><td>定义</td><td>值</td><td>解释</td></tr><tr><td>FIELD_0</td><td>0</td><td>Flash 数据区域 0</td></tr><tr><td>FIELD_1</td><td>1</td><td>Flash 数据区域 1</td></tr><tr><td>FIELD_2</td><td>2</td><td>Flash 数据区域 2</td></tr><tr><td>FIELD_3</td><td>3</td><td>Flash 数据区域 3</td></tr></table>

## 3.3.22. GxTestPatternEntry

<table><tr><td>定义</td><td>值</td><td>解释</td></tr><tr><td>OFF</td><td>0</td><td>关闭</td></tr><tr><td>GRAY_FRAME_RAMP_MOVING</td><td>1</td><td>静止灰度递增</td></tr><tr><td>SLANT_LINE_MOVING</td><td>2</td><td>滚动斜条纹</td></tr><tr><td>VERTICAL_LINE_MOVING</td><td>3</td><td>滚动竖条纹</td></tr><tr><td>HORIZONTAL_LINE_MOVING</td><td>4</td><td>滚动横条纹</td></tr><tr><td>GREY_VERTICAL_RAMP</td><td>5</td><td>垂直灰度递增</td></tr><tr><td>SLANT_LINE</td><td>6</td><td>静止斜条纹</td></tr></table>

## 3.3.23. GxTriggerSelectorEntry

<table><tr><td>定义</td><td>值</td><td>解释</td></tr><tr><td>FRAME_START</td><td>1</td><td>采集一帧</td></tr><tr><td>FRAME_BURST_START</td><td>2</td><td>帧高速连拍开始</td></tr></table>

## 3.3.24. GxLineSelectorEntry

<table><tr><td>定义</td><td>值</td><td>解释</td></tr><tr><td>LINE0</td><td>0</td><td>引脚 0</td></tr><tr><td>LINE1</td><td>1</td><td>引脚 1</td></tr><tr><td>LINE2</td><td>2</td><td>引脚 2</td></tr><tr><td>LINE3</td><td>3</td><td>引脚 3</td></tr><tr><td>LINE4</td><td>4</td><td>引脚 4</td></tr><tr><td>LINE5</td><td>5</td><td>引脚 5</td></tr><tr><td>LINE6</td><td>6</td><td>引脚 6</td></tr><tr><td>LINE7</td><td>7</td><td>引脚 7</td></tr><tr><td>LINE8</td><td>8</td><td>引脚 8</td></tr><tr><td>LINE9</td><td>9</td><td>引脚 9</td></tr><tr><td>LINE10</td><td>10</td><td>引脚 10</td></tr><tr><td>LINE_STROBE</td><td>11</td><td>专用闪光灯引脚</td></tr></table>


3.3.25. GxLineModeEntry


<table><tr><td>定义</td><td>值</td><td>解释</td></tr><tr><td>INPUT</td><td>0</td><td>输入</td></tr><tr><td>OUTPUT</td><td>1</td><td>输出</td></tr></table>


3.3.26. GxLineSourceEntry


<table><tr><td>定义</td><td>值</td><td>解释</td></tr><tr><td>OFF</td><td>0</td><td>关闭</td></tr><tr><td>STROBE</td><td>1</td><td>闪光灯</td></tr><tr><td>USER_OUTPUT0</td><td>2</td><td>用户自定义输出 0</td></tr><tr><td>USER_OUTPUT1</td><td>3</td><td>用户自定义输出 1</td></tr><tr><td>USER_OUTPUT2</td><td>4</td><td>用户自定义输出 2</td></tr><tr><td>EXPOSURE_ACTIVE</td><td>5</td><td>曝光有效</td></tr><tr><td>FRAME_TRIGGER_WAIT</td><td>6</td><td>单帧触发等待</td></tr><tr><td>ACQUISITION_TRIGGER_WAIT</td><td>7</td><td>多帧触发等待</td></tr><tr><td>TIMER1_ACTIVE</td><td>8</td><td>定时器 1 有效</td></tr><tr><td>USER_OUTPUT3</td><td>9</td><td>用户自定义输出 3</td></tr><tr><td>USER_OUTPUT4</td><td>10</td><td>用户自定义输出 4</td></tr><tr><td>USER_OUTPUT5</td><td>11</td><td>用户自定义输出 5</td></tr><tr><td>USER_OUTPUT6</td><td>12</td><td>用户自定义输出 6</td></tr></table>


3.3.27. GxEventSelectorEntry


<table><tr><td>定义</td><td>值</td><td>解释</td></tr><tr><td>EXPOSURE_END</td><td>0x0004</td><td>曝光结束</td></tr><tr><td>BLOCK_DISCARD</td><td>0x9000</td><td>图像帧丢弃</td></tr><tr><td>EVENT_OVERRUN</td><td>0x9001</td><td>事件队列溢出</td></tr><tr><td>FRAME_START_OVER_TRIGGER</td><td>0x9002</td><td>触发信号溢出</td></tr><tr><td>BLOCK_NOT_EMPTY</td><td>0x9003</td><td>图像帧存不为空</td></tr><tr><td>INTERNAL_ERROR</td><td>0x9004</td><td>内部错误事件</td></tr><tr><td>FRAME_BURST_START_OVERT RIGGER</td><td>0x9005</td><td>多帧触发屏蔽事件</td></tr><tr><td>FRAME_START_WAIT</td><td>0x9006</td><td>帧等待事件</td></tr><tr><td>FRAME_BURST_START_WAIT</td><td>0x9007</td><td>多帧等待事件</td></tr></table>

## 3.3.28. GxLutSelectorEntry

<table><tr><td>定义</td><td>值</td><td>解释</td></tr><tr><td>LUMINANCE</td><td>0</td><td>亮度</td></tr></table>

## 3.3.29. GxTransferControlModeEntry

<table><tr><td>定义</td><td>值</td><td>解释</td></tr><tr><td>BASIC</td><td>0</td><td>基础模式</td></tr><tr><td>USER_CONTROLED</td><td>1</td><td>用户控制模式</td></tr></table>

## 3.3.30. GxTransferOperationModeEntry

<table><tr><td>定义</td><td>值</td><td>解释</td></tr><tr><td>MULTI_BLOCK</td><td>0</td><td>指定发送帧数</td></tr></table>

## 3.3.31. GxTestPatternGeneratorSelectorEntry

<table><tr><td>定义</td><td>值</td><td>解释</td></tr><tr><td>SENSOR</td><td>0</td><td>sensor 的测试图</td></tr><tr><td>REGION0</td><td>1</td><td>FPGA 的测试图</td></tr></table>

## 3.3.32. GxChunkSelectorEntry

<table><tr><td>定义</td><td>值</td><td>解释</td></tr><tr><td>FRAME_ID</td><td>1</td><td>帧号</td></tr><tr><td>TIME_STAMP</td><td>2</td><td>时间戳</td></tr><tr><td>COUNTER_VALUE</td><td>3</td><td>计数器值</td></tr></table>

## 3.3.33. GxBinningHorizontalModeEntry

<table><tr><td>定义</td><td>值</td><td>解释</td></tr><tr><td>SUM</td><td>0</td><td>BINNING 水平值和</td></tr><tr><td>AVERAGE</td><td>1</td><td>BINNING 水平值平均值</td></tr></table>

## 3.3.34. GxBinningVerticalModeEntry

<table><tr><td>定义</td><td>值</td><td>解释</td></tr><tr><td>SUM</td><td>0</td><td>BINNING 垂直值和</td></tr><tr><td>AVERAGE</td><td>1</td><td>BINNING 垂直值平均值</td></tr></table>

## 3.3.35. GxSensorShutterModeEntry

<table><tr><td>定义</td><td>值</td><td>解释</td></tr><tr><td>GLOBAL</td><td>0</td><td>所有的像素同时曝光且曝光时间相等</td></tr><tr><td>ROLLING</td><td>1</td><td>所有的像素曝光时间相等,但曝光起始时间不同</td></tr><tr><td>GLOBALRESET</td><td>2</td><td>所有的像素曝光起始时间相同,但曝光时间不相等</td></tr></table>


3.3.36. GxAcquisitionStatusSelectorEntry


<table><tr><td>定义</td><td>值</td><td>解释</td></tr><tr><td>ACQUISITION_TRIGGER_WAIT</td><td>0</td><td>采集触发等待</td></tr><tr><td>FRAME_TRIGGER_WAIT</td><td>1</td><td>帧触发等待</td></tr></table>

## 3.3.37. GxExposureTimeModeEntry

<table><tr><td>定义</td><td>值</td><td>解释</td></tr><tr><td>ULTRASHORT</td><td>0</td><td>极小曝光</td></tr><tr><td>STANDARD</td><td>1</td><td>标准</td></tr></table>

## 3.3.38. GxGammaModeEntry

<table><tr><td>定义</td><td>值</td><td>解释</td></tr><tr><td>SRGB</td><td>0</td><td>默认 Gamma 校正</td></tr><tr><td>USER</td><td>1</td><td>用户自定义 Gamma 校正</td></tr></table>

## 3.3.39. GxLightSourcePresetEntry

<table><tr><td>定义</td><td>值</td><td>解释</td></tr><tr><td>OFF</td><td>0</td><td>关闭</td></tr><tr><td>CUSTOM</td><td>1</td><td>用户自定义</td></tr><tr><td>DAYLIGHT_6500K</td><td>2</td><td>标准日光灯(6500K)</td></tr><tr><td>DAYLIGHT_5000K</td><td>3</td><td>标准日光灯(5000K)</td></tr><tr><td>COOL_WHITE_FLUORESCENCE</td><td>4</td><td>冷白光源(4150K)</td></tr><tr><td>INCA</td><td>5</td><td>螺旋钨丝灯(2856K)</td></tr></table>

## 3.3.40. GxColorTransformationModeEntry

<table><tr><td>定义</td><td>值</td><td>解释</td></tr><tr><td>RGB_TO_RGB</td><td>0</td><td>默认颜色校正</td></tr><tr><td>USER</td><td>1</td><td>用户自定义颜色校正</td></tr></table>

## 3.3.41. GxColorTransformationValueSelectorEntry

<table><tr><td>定义</td><td>值</td><td>解释</td></tr><tr><td>GAIN00</td><td>0</td><td>颜色转换分量增益值 GAIN00</td></tr><tr><td>GAIN01</td><td>1</td><td>颜色转换分量增益值 GAIN01</td></tr><tr><td>GAIN02</td><td>2</td><td>颜色转换分量增益值 GAIN02</td></tr><tr><td>GAIN10</td><td>3</td><td>颜色转换分量增益值 GAIN10</td></tr><tr><td>GAIN11</td><td>4</td><td>颜色转换分量增益值 GAIN11</td></tr><tr><td>GAIN12</td><td>5</td><td>颜色转换分量增益值 GAIN12</td></tr><tr><td>GAIN20</td><td>6</td><td>颜色转换分量增益值 GAIN20</td></tr><tr><td>GAIN21</td><td>7</td><td>颜色转换分量增益值 GAIN21</td></tr><tr><td>GAIN22</td><td>8</td><td>颜色转换分量增益值 GAIN22</td></tr></table>


3.3.42. GxAutoEntry


<table><tr><td>定义</td><td>值</td><td>解释</td></tr><tr><td>OFF</td><td>0</td><td>关闭</td></tr><tr><td>CONTINUOUS</td><td>1</td><td>连续</td></tr><tr><td>ONCE</td><td>2</td><td>单次</td></tr></table>


3.3.43. GxSwitchEntry


<table><tr><td>定义</td><td>值</td><td>解释</td></tr><tr><td>OFF</td><td>0</td><td>关闭</td></tr><tr><td>ON</td><td>1</td><td>开启</td></tr></table>


3.3.44. GxRegionSendModeEntry


<table><tr><td>定义</td><td>值</td><td>解释</td></tr><tr><td>SINGLE_ROI</td><td>0</td><td>单 ROI</td></tr><tr><td>MULTI_ROI</td><td>1</td><td>多 ROI</td></tr></table>

## 3.3.45. GxRegionSelectorEntry

<table><tr><td>定义</td><td>值</td><td>解释</td></tr><tr><td>REGION0</td><td>0</td><td>区域 0</td></tr><tr><td>REGION1</td><td>1</td><td>区域 1</td></tr><tr><td>REGION2</td><td>2</td><td>区域 2</td></tr><tr><td>REGION3</td><td>3</td><td>区域 3</td></tr><tr><td>REGION4</td><td>4</td><td>区域 4</td></tr><tr><td>REGION5</td><td>5</td><td>区域 5</td></tr><tr><td>REGION6</td><td>6</td><td>区域 6</td></tr><tr><td>REGION7</td><td>7</td><td>区域 7</td></tr></table>

## 3.3.46. GxTimerSelectorEntry

<table><tr><td>定义</td><td>值</td><td>解释</td></tr><tr><td>TIMER1</td><td>1</td><td>定时器 1</td></tr></table>

## 3.3.47. GxTimerTriggerSourceEntry

<table><tr><td>定义</td><td>值</td><td>解释</td></tr><tr><td>EXPOSURE_START</td><td>1</td><td>曝光开始信号开始计时</td></tr><tr><td>LINE10</td><td>10</td><td>接收引脚 10 信号开始计时</td></tr><tr><td>STROBE</td><td>16</td><td>接收闪光灯信号开始计时</td></tr></table>

## 3.3.48. GxCounterSelectorEntry

<table><tr><td>定义</td><td>值</td><td>解释</td></tr><tr><td>COUNTER1</td><td>1</td><td>计数器 1</td></tr><tr><td>COUNTER2</td><td>2</td><td>计数器 2</td></tr></table>


3.3.49. GxCounterEventSourceEntry


<table><tr><td>定义</td><td>值</td><td>解释</td></tr><tr><td>FRAME_START</td><td>1</td><td>统计 &quot;帧开始&quot; 事件的数量</td></tr><tr><td>FRAME_TRIGGER</td><td>2</td><td>统计 &quot;帧触发&quot; 事件的数量</td></tr><tr><td>ACQUISITION_TRIGGER</td><td>3</td><td>统计 &quot;采集触发&quot; 事件的数量</td></tr><tr><td>OFF</td><td>4</td><td>关闭</td></tr><tr><td>SOFTWARE</td><td>5</td><td>统计 &quot;软触发&quot; 事件的数量</td></tr><tr><td>LINE0</td><td>6</td><td>统计 &quot;Line 0 触发&quot; 事件的数量</td></tr><tr><td>LINE1</td><td>7</td><td>统计 &quot;Line 1 触发&quot; 事件的数量</td></tr><tr><td>LINE2</td><td>8</td><td>统计 &quot;Line 2 触发&quot; 事件的数量</td></tr><tr><td>LINE3</td><td>9</td><td>统计 &quot;Line 3 触发&quot; 事件的数量</td></tr></table>

## 3.3.50. GxCounterResetSourceEntry

<table><tr><td>定义</td><td>值</td><td>解释</td></tr><tr><td>OFF</td><td>0</td><td>无复位源</td></tr><tr><td>SOFTWARE</td><td>1</td><td>软触发</td></tr><tr><td>LINE0</td><td>2</td><td>引脚 0</td></tr><tr><td>LINE1</td><td>3</td><td>引脚 1</td></tr><tr><td>LINE2</td><td>4</td><td>引脚 2</td></tr><tr><td>LINE3</td><td>5</td><td>引脚 3</td></tr><tr><td>COUNTER2END</td><td>6</td><td>COUNTER2END 触发信号</td></tr></table>

## 3.3.51. GxCounterResetActivationEntry

<table><tr><td>定义</td><td>值</td><td>解释</td></tr><tr><td>RISINGEDGE</td><td>1</td><td>上升沿触发</td></tr></table>

## 3.3.52. GxCounterTriggerSourceEntry

<table><tr><td>定义</td><td>值</td><td>解释</td></tr><tr><td>OFF</td><td>0</td><td>无触发源</td></tr><tr><td>SOFTWARE</td><td>1</td><td>软触发</td></tr><tr><td>LINE0</td><td>2</td><td>引脚 0</td></tr><tr><td>LINE1</td><td>3</td><td>引脚 1</td></tr><tr><td>LINE2</td><td>4</td><td>引脚 2</td></tr><tr><td>LINE3</td><td>5</td><td>引脚 3</td></tr></table>

## 3.3.53. GxTimerTriggerActivationEntry

<table><tr><td>定义</td><td>值</td><td>解释</td></tr><tr><td>RISINGEDGE</td><td>1</td><td>上升沿触发</td></tr></table>


3.3.54. GxStopAcquisitionModeEntry


<table><tr><td>定义</td><td>值</td><td>解释</td></tr><tr><td>GENERAL</td><td>0</td><td>普通停采</td></tr><tr><td>LIGHT</td><td>1</td><td>轻量级停采</td></tr></table>


3.3.55. GxDSStreamBufferHandlingModeEntry


<table><tr><td>定义</td><td>值</td><td>解释</td></tr><tr><td>OLDEST_FIRST</td><td>1</td><td>OldestFirst 模式</td></tr><tr><td>OLDEST_FIRST_OVERWRITE</td><td>2</td><td>OldestFirstOverwrite 模式</td></tr><tr><td>NEWEST_ONLY</td><td>3</td><td>NewestOnly 模式</td></tr></table>

## 3.3.56. GxResetDeviceModeEntry

<table><tr><td>定义</td><td>值</td><td>解释</td></tr><tr><td>RECONNECT</td><td>1</td><td>重连设备</td></tr><tr><td>RESET</td><td>2</td><td>复位设备</td></tr></table>


3.3.57. DxBayerConvertType


<table><tr><td>定义</td><td>值</td><td>解释</td></tr><tr><td>NEIGHBOUR</td><td>0</td><td>邻域平均插值算法</td></tr><tr><td>ADAPTIVE</td><td>1</td><td>边缘自适应插值算法</td></tr><tr><td>NEIGHBOUR3</td><td>2</td><td>更大区域的邻域平均插值算法</td></tr></table>


3.3.58. DxValidBit


<table><tr><td>定义</td><td>值</td><td>解释</td></tr><tr><td>BIT0_7</td><td>0</td><td>0-7 位</td></tr><tr><td>BIT1_8</td><td>1</td><td>1-8 位</td></tr><tr><td>BIT2_9</td><td>2</td><td>2-9 位</td></tr><tr><td>BIT3_10</td><td>3</td><td>3-10 位</td></tr></table>


3.3.59. DxImageMirrorMode


<table><tr><td>定义</td><td>值</td><td>解释</td></tr><tr><td>HORIZONTAL_MIRROR</td><td>0</td><td>水平镜像</td></tr><tr><td>VERTICAL_MIRROR</td><td>1</td><td>垂直镜像</td></tr></table>

## 3.3.60. DxRGBChannelOrder

<table><tr><td>定义</td><td>值</td><td>解释</td></tr><tr><td>ORDER_RGB</td><td>0</td><td>RGB 通道</td></tr><tr><td>ORDER_BGR</td><td>1</td><td>BGR 通道</td></tr></table>

## 3.4. 模块接口定义

## 3.4.1. DeviceManager

负责相机设备的管理，包括枚举设备、打开设备、获取设备数量信息等。

接口列表：

<table><tr><td>update_device_list (timeout=200)</td><td>枚举同一网段中的设备</td></tr><tr><td>update_all_device_list (timeout=200)</td><td>枚举不同网段中的设备</td></tr><tr><td>get_device_number()</td><td>获取设备数量</td></tr><tr><td>get_device_info()</td><td>获取设备信息</td></tr><tr><td>open_device_by_sn (sn,access_mode=GxAccessMode.CONTROL)</td><td>通过序列号打开设备</td></tr><tr><td>open_device_by_user_id (user_id,access_mode=GxAccessMode.CONTROL)</td><td>通过用户ID号打开设备</td></tr><tr><td>open_device_by_index (index,access_mode=GxAccessMode.CONTROL)</td><td>通过设备索引打开设备</td></tr><tr><td>open_device_by_ip (ip,access_mode=GxAccessMode.CONTROL)</td><td>通过IP地址打开设备</td></tr><tr><td>open_device_by_mac (mac,access_mode=GxAccessMode.CONTROL)</td><td>通过mac地址打开设备</td></tr><tr><td>gige_reset_device (mac_address, reset_device_mode)</td><td>执行设备重连或复位操作</td></tr></table>

##  接口说明

##  update_device_list

声明：

DeviceManager.update_device_list (timeout=200) 

意义：

对于非千兆网相机，枚举所有设备；对于千兆网相机，枚举同一网段设备。

形参：

[in]timeout 枚举超时[0，0xffffffff]，缺省值为 200（ms）

返回值：

枚举得到设备数量和记录枚举设备信息的列表（list）。设备信息列表的元素个数为枚举到的设备个数，列表中元素的数据类型字典，字典中的键名称详见枚举设备

异常处理：

1) 如果输入参数不是整型值，则抛出 ParameterTypeError 异常。

2) 如果输入参数小于 0 或大于无符号整型的最大值，则打印”DeviceManager.update_device_list: Outof bounds, timeout:minimum=0, maximum= 0xffffffff”，函数返回 None。

3) 如果枚举同一网段中设备不成功，则抛出异常，异常类型详见错误处理。

4) 如果获取所有设备基本信息不成功，则抛出异常，异常类型详见错误处理。

##  update_all_device_list

声明：

```txt
DeviceManager.update_all_device_list (timeout=200) 
```

意义：

对于非千兆网相机，枚举所有设备；对于千兆网相机，枚举全网设备。

形参：

[in]timeout 枚举超时[0，0xffffffff]，缺省值为 200（ms）

返回值：

枚举得到设备数量和记录枚举设备信息的列表（list）。设备信息列表的元素个数为枚举到的设备个数，列表中元素的数据类型字典，字典中的键名称详见枚举设备

异常处理：

1) 如果输入参数不是整型值，则抛出 ParameterTypeError 异常。

2) 如果输入参数小于 0 或大于无符号整型的最大值，则打印”DeviceManager.update_all_device_list:Out of bounds, timeout:minimum=0, maximum= 0xffffffff”，函数返回 None。

3) 如果枚举不同网段中设备不成功，则抛出异常，异常类型详见错误处理。

4) 如果获取所有设备基本信息不成功，则抛出异常，异常类型详见错误处理。

##  get_device_number

声明：

```txt
DeviceManager.get_device_number () 
```

意义：

获取设备数量

返回值：

设备数量

```ignorefile
get_device_info 
```

声明：

意义：

获取设备信息

返回值：

设备信息列表。设备信息列表的元素个数为枚举到的设备个数，列表中元素的数据类型为字典，字典的键详见枚举设备

```txt
open_device_by_sn 
```

声明：

DeviceManager.open_device_by_sn (sn, access_mode=GxAccessMode.CONTROL) 

意义：

通过序列号打开设备

形参：

[in]sn 序列号[字符串类型]

[in]access_mode 打开设备模式，缺省值为 GxAccessMode.CONTROL，查看 GxAccessMode返回值：

设备对象

异常处理：

1) 如果输入参数 1 不是字符串型，则抛出 ParameterTypeError 异常。

2) 如果输入参数 2 不是整型，则抛出 ParameterTypeError 异常。

3) 如果输入参数2 不在打开设备模式 GxAccessMode中，则打印接口名称、打开设备方式不在范围内和当前参数所支持的枚举值信息，函数返回None。

4) 如果重复获取设备类不成功，则抛出NotFoundDevice 异常。

5) 如果打开设备不成功，则抛出异常，异常类型详见错误处理。

6) 如果打开获取的设备不是 U3V/USB2/GEV 类中的一种，则抛出 NotFoundDevice 异常。

##  open_device_by_user_id

声明：

DeviceManager.open_device_by_user_id (user_id, access_mode=GxAccessMode.CONTROL)意义：

通过用户ID 号打开设备

形参：

[in]user_id 用户 ID 号[字符串类型]

[in]access_mode 打开设备模式，缺省值为 GxAccessMode.CONTROL，查看 GxAccessMode返回值：

设备对象

异常处理：

1) 如果输入参数 1 不是字符串型，则抛出 ParameterTypeError 异常。

2) 如果输入参数 2 不是整型，则抛出 ParameterTypeError 异常。

3) 如果输入参数2 不在打开设备模式 GxAccessMode中，则打印接口名称、打开设备方式不在范围内和当前参数所支持的枚举值的信息，函数返回None。

4) 如果重复获取设备类不成功，则抛出NotFoundDevice 异常。

5) 如果打开设备不成功，则抛出异常，异常类型详见错误处理。

6) 如果打开获取的设备不是 U3V/USB2/GEV 类中的一种，则抛出 NotFoundDevice 异常。

##  open_device_by_index

声明：

DeviceManager.open_device_by_index ( index, access_mode=GxAccessMode.CONTROL) 

意义：

通过设备索引打开设备

形参：

[in]index 设备索引[1,2,3...0xffffffff]

返回值： 回值：

[in]access_mode 打开设备方式，缺省值为 GxAccessMode.CONTROL，查看 GxAccessMode

## 设备对象

异常处理：

1) 如果输入参数 1 或 2 不是整型值，则抛出 ParameterTypeError 异常。

2) 如果输入参数 1 小于 0 或大于无符号整型的最大值，则打印”DeviceManager.open_device_by_index:index out of bounds, index: minimum=1, maximum= 0xffffffff”，函数返回 None。

3) 如果输入参数2 不在打开设备模式 GxAccessMode中，则打印接口名称、打开设备方式不在范围内和当前参数所支持的枚举值的信息，函数返回None。

4) 如果设备数量小于输入参数 1索引，则抛出NotFoundDevice异常。

5) 如果打开设备不成功，则抛出异常，异常类型详见错误处理。

6) 如果打开获取的设备不是 U3V/USB2/GEV 类中的一种，则抛出 NotFoundDevice 异常。

##  open_device_by_ip

声明：

DeviceManager.open_device_by_ip (ip, access_mode=GxAccessMode.CONTROL) 

意义：

通过设备ip地址打开千兆网相机设备

形参：

[in]ip 设备 ip 地址[字符串类型]

[in]access_mode 打开设备模式，缺省值为 GxAccessMode.CONTROL，查看 GxAccessMode返回值：

## 设备对象

异常处理：

1) 如果输入参数 1 不是字符串型，则抛出 ParameterTypeError 异常。

2) 如果输入参数 2 不是整型，则抛出 ParameterTypeError 异常。

3) 如果输入参数2 不在打开设备模式 GxAccessMode中，则打印接口名称、打开设备方式不在范围内和当前参数所支持的枚举值的信息，函数返回None。

4) 如果打开设备不成功，则抛出异常，异常类型详见错误处理。

##  open_device_by_mac

声明：

DeviceManager.open_device_by_mac (mac, access_mode=GxAccessMode.CONTROL) 

意义：

通过设备mac 地址打开千兆网相机设备

形参：

[in]mac 设备mac地址[字符串类型]

[in]access_mode 打开设备模式，缺省值为 GxAccessMode.CONTROL，查看 GxAccessMode返回值：

设备对象

异常处理：

1) 如果输入参数 1 不是字符串型，则抛出 ParameterTypeError 异常。

2) 如果输入参数 2 不是整型，则抛出 ParameterTypeError 异常。

3) 如果输入参数2 不在打开设备模式 GxAccessMode中，则打印接口名称、打开设备方式不在范围内和当前参数所支持的枚举值的信息，函数返回None。

4) 如果打开设备不成功，则抛出异常，异常类型详见错误处理。

##  gige_reset_device

声明：

DeviceManager.gige_reset_device (mac_address, reset_device_mode) 

意义：

执行设备重连或复位操作。

设备重连通常应用在调试千兆网相机时，设备已经被打开，此时程序异常，而后立即重新打开设备报错（因为调试心跳5 分钟，设备依然处于打开状态），这时可通过设备重连功能，使设备处于未打开状态，而后再次打开设备即可成功。

设备复位通常应用在相机状态异常，此时设备重连功能也无法生效，也不具备给设备重新上电的条件，可尝试使用设备复位功能，使设备掉电再上电。设备复位后，需要重新执行枚举、打开设备操作。

注意：

1) 设备复位时间需要1s 左右，因此需要确保1s后再调用枚举接口。

2) 如果设备正在正常采集，禁止使用设备复位和重连功能，否则会导致设备掉线。形参：

```json
[in]mac_address 设备 mac 地址[字符串类型]
```

[in]reset_device_mode 重置设备模式，参考 GxResetDeviceModeEntry。

返回值：

None 

异常处理：

1) 如果执行命令失败，则抛出异常，异常类型详见错误处理。

## 3.4.2. Device

负责相机设备的采集控制、设备关闭、配置文件导入导出和获取设备句柄等。

接口列表：

<table><tr><td>get_stream_channel_num()</td><td>获取当前设备支持的流通道个数</td></tr><tr><td>stream_on ()</td><td>发送开始命令,相机开始传送图像数据</td></tr><tr><td>stream_off ()</td><td>发送结束命令,相机结束传送图像数据</td></tr><tr><td>export_config_file (file_path)</td><td>导出当前配置文件</td></tr><tr><td>import_config_file (file_path, verify=False)</td><td>导入配置文件</td></tr><tr><td>close_device ()</td><td>关闭设备,销毁设备句柄,将句柄置为空</td></tr></table>

##  接口说明

```txt
get_stream_channel_num 
```

声明：

```python
Device.get_stream_channel_num() 
```

意义：

获取当前设备支持的流通道个数

返回值：

流通道个数

注：目前千兆网相机、USB3.0、USB2.0相机均不支持多流通道

```txt
stream_on 
```

声明：

```txt
Device.stream_on() 
```

意义：

发送开始命令，相机开始传送图像数据

返回值：

None 

异常处理：

1) 如果发送开始命令不成功，则抛出异常，异常类型详见错误处理。

意义：

发送停止命令，相机停止传送图像数据

异常处理：

1) 如果发送停止命令不成功，则抛出异常，异常类型详见错误处理。

```txt
> export_config_file
声明：
Device.export_config_file(file_path)
```

意义：

1) 如果输入参数不是字符串型，则抛出 ParameterTypeError 异常。

2) 如果导出当前配置文件不成功，则抛出异常，异常类型详见错误处理。

##  import_config_file

```txt
声明：Device.import_config_file(file_path, verify=False)
```

意义：

导入配置文件

形参：

[in]file_path 文件路径

[in]verify 是否所有导入值将被验证一致性，缺省值为 False

返回值：

异常处理：

1) 如果输入参数 1 不是字符串型，则抛出 ParameterTypeError 异常。

2) 如果输入参数 2 不是布尔型，则抛出 ParameterTypeError 异常。

3) 如果导入配置文件不成功，则抛出异常，异常类型详见错误处理。

```txt
> close_device
声明：
Device.close_device()
```

意义：

关闭设备，销毁设备句柄，将句柄置为空

```txt
返回值：
None
```

异常处理：

1) 如果关闭设备不成功，则抛出异常，异常类型详见错误处理。注意：

当执行关闭设备后，如果还想使用此相机，请重新打开后再操作。

```txt
register_device_offline_callback
声明：
Device.register_device_offline_callback(callback_func)
```

意义：

注册掉线回调函数。

形参：

[in]callback_func 掉线回调函数

返回值：

异常处理：

1) 如果输入参数不是函数类型，则抛出 ParameterTypeError 异常。

2) 如果注册掉线回调函数失败，则抛出异常，异常类型详见错误处理。

```txt
unregister_device_offline_callback
声明：
Device.unregister_device_offline_callback()
```

意义：

注销设备掉线回调函数。

返回值：

异常处理：

1) 如果注销掉线回调函数失败，则抛出异常，异常类型详见错误处理。

## 3.4.3. DataStream

负责相机设备的数据流设置、控制，获取图像等。

接口列表：

<table><tr><td>set_acquisition_buffer_number(buf_num)</td><td>设置采集缓冲的大小</td></tr><tr><td>get_image(timeout=1000)</td><td>获取图像,成功创建图像类对象</td></tr><tr><td>flush_queue()</td><td>清除相机采集缓冲队列</td></tr></table>

##  接口说明

##  set_acquisition_buffer_number

声明：

```txt
DataStream.set_acquisition_buffer_number(buf_num) 
```

意义：

设置采集缓冲的大小

形参：

[in]buf_num 缓冲区地址的长度[1，0xffffffff]

返回值：

None 

异常处理：

1) 如果输入参数不是整型，则抛出 ParameterTypeError 异常。

```txt
2) 如果输入参数小于 1 或大于无符号整型的最大值，则打印
“DataStream.set_acquisition_buffer_number: buf_num out of bounds, minimum=1, maximum=0xffffff”，函数返回 None。
```

3) 如果设置采集缓冲大小不成功，则抛出异常，异常类型详见错误处理。

##  get_image

声明：

```txt
DataStream.get_image(timeout=1000) 
```

意义：

获取图像，成功创建图像类对象

形参：

[in]timeout 获取超时[0，0xffffffff]，缺省值为 1000（ms）

返回值：

图像对象： 获取成功

None: 超时

抛出异常： 其他错误

异常处理：

1) 如果输入参数不是整型，则抛出 ParameterTypeError 异常。

2) 如果输入参数小于 0 或大于 0xffffffff，则打印”DataStream.get_image: timeout out of bounds,minimum=0, maximum=0xffffffff”，函数返回 None。

3) 如果获取数据大小不成功，则抛出异常，异常类型详见错误处理。

4) 如果超时导致未获取图像成功，则函数返回None。

5) 如果未超时但获取图像失败，则打印“status，DataStream，get_image”，函数返回 None。

##  flush_queue

声明：

DataStream.flush_queue() 

意义：

清除相机采集缓冲队列

异常处理：

1) 如果清除相机采集缓冲队列不成功，则抛出异常，异常类型详见错误处理

##  register_capture_callback

声明：

DataStream.register_capture_callback(callback_func) 

意义：

注册采集回调函数，回调方式采集使用方式见：采集控制-回调方式

形参：

[in]callback_func 采集回调函数

返回值：

None 

异常处理：

1) 如果输入参数不是函数类型，则抛出 ParameterTypeError 异常。

2) 如果注册采集回调函数失败，则抛出异常，异常类型详见错误处理。注意：

需要先注册采集回调函数，再执行stream_on 开采。

 unregister_capture_callback 

声明：

DataStream.unregister_capture_callback() 

意义：

注销采集回调函数

返回值：

None 

异常处理：

1) 如果注销采集回调函数失败，则抛出异常，异常类型详见错误处理。

## 3.4.4. RGBImage

负责RGB 图像操作。

接口列表：

```txt
image_improvement(color_correction_param=0, contrast_lut=None, gamma_lut=None, channel_order=DxRGBChannelOrder.ORDER_RGB) 图像质量提高
brightness(factor) 图像进行亮度调节
contrast(factor) 图像进行对比度调节
saturation(factor) 图像进行饱和度调节
sharpen(factor) 图像进行锐化处理
get_white_balance_ratio() 获取图像白平衡系数
get_numpy_array() 将 RGB 数据转换为 numpy 对象
get_image_size() 获取 RGB 数据大小
```

##  接口说明

##  image_improvement

声明：

RGBImage.image_improvement( color_correction_param=0, contrast_lut=None, gamma_lut=None, channel_order=DxRGBChannelOrder.ORDER_RGB) 

意义：

图像质量提高

形参：

```txt
[in]contrast_lut 对比度 LUT
[in]gamma_lut gamma LUT
[in]color_collect 颜色校正
[in] channel_order 图像通道顺序，缺省值为 DxRGBChannelOrder.ORDER_RGB，参考 DxRGBChannelOrder
```

异常处理：

1) 如果参数 1、2 不是 Buffer 类型或 None，则抛异常 ParameterTypeError。

2) 如果参数 3 不是整型或 None，则抛异常 ParameterTypeError。

3) 如果参数 4 不是整型，则抛异常 ParameterTypeError。

4) 如果提高图像质量不成功，则抛出异常UnexpectedError。

5) 如果参数1、2、3都是默认缺省值，则不进行图像质量提升处理，函数退出。

##  brightness

声明：

RGBImage. brightness(factor) 

意义：

对RGB24 图像进行亮度调节

## 形参：

[in]factor 亮度调节因子，值范围-150~150。

其中： 0：亮度没有变化；

大于 0：增加亮度；

小于 0：减小亮度；

返回值：

None 

异常处理：

1) 如果输入参数不是整型，则抛出 ParameterTypeError 异常。

2) 如果亮度调节失败，则抛出 UnexpetedError 异常。

##  contrast

声明：

RGBImage.contrast(factor) 

意义：

对RGB24 图像进行对比度调节

## 形参：

[in]factor 对比度调节因子，值范围-50~150。

其中: 0：对比度没有变化；

大于0：增加对比度；

小于0：减小对比度；

返回值：

None 

异常处理：

1) 如果输入参数不是整型，则抛出 ParameterTypeError 异常。

2) 如果对比度调节失败，则抛出 UnexpetedError 异常。

##  saturation

声明：

RGBImage.saturation(factor) 

```txt
对 RGB 图像进行饱和度调节
形参：
[in]factor 饱和度调节参数，范围：0 ~ 128，
其中：64：饱和度没有变化；
大于 64：增加饱和度；
小于 64：减小饱和度；
128：饱和度为当前两倍；
0：黑白图像
返回值：
None
异常处理：
1）如果图像饱和度调节失败，则抛出异常 UnexpectedError。
> sharpen
声明：
RGBImage.sharpen()
意义：
对 RGB 图像进行锐化处理
形参：
[in]factor 锐化调节参数，范围：0.1 ~ 5.0
返回值：
None
异常处理：
1）如果图像锐化处理失败，则抛出异常 UnexpectedError。
> get_white_balance_ratio
声明：
RGBImage.get_white_balance_ratio()
意义：
获取白平衡系数
返回值：
返回 RGB 分量系数元组
> get_numpy_array
声明：
RGBImage.get_numpy_array()
```

## 意义：

意义：

获取 numpy 对象

返回值：

numpy 对象

 get_image_size 

声明：

RGBImage.get_image_size() 

意义：

获取RGB 数据大小

返回值：

RGB 图的大小

## 3.4.5. RawImage

负责Raw图像操作。

接口列表：

<table><tr><td>convert(mode, flip=False, valid_bits=DxValidBit.BIT4_11, convert_type=DxBayerConvertType.NEIGHBOUR, channel_order=DxRGBChannelOrder.ORDER_RGB)</td><td>图像格式转换</td></tr><tr><td>defective_pixel_correct()</td><td>图像坏点校正</td></tr><tr><td>raw8_rotate_90_cw()</td><td>将8位图像顺时针旋转90度</td></tr><tr><td>raw8_rotate_90_ccw()</td><td>将8位图像逆时针旋转90度</td></tr><tr><td>brightness(factor)</td><td>对8位灰度图像进行亮度调节</td></tr><tr><td>contrast(factor)</td><td>对8位灰度图像进行对比度调节</td></tr><tr><td>mirror(mirror_mode)</td><td>对8位图像进行镜像处理</td></tr><tr><td>get_ffc_coefficients(dark_img=None, target_value=None)</td><td>计算图像平场校正系数</td></tr><tr><td>flat_field_correction(ffc_coefficients)</td><td>对图像进行平场校正处理</td></tr><tr><td>get_numpy_array()</td><td>将raw数据转换为numpy对象</td></tr><tr><td>get_data()</td><td>获取raw数据</td></tr><tr><td>save_raw(file_path)</td><td>保存raw图数据</td></tr><tr><td>get_status()</td><td>获取raw图状态</td></tr><tr><td>get_width()</td><td>获取raw图宽度</td></tr><tr><td>get_height()</td><td>获取raw图高度</td></tr><tr><td>get_pixel_format()</td><td>获取图像像素格式</td></tr><tr><td>get_image_size()</td><td>获取raw图数据大小</td></tr><tr><td>get_frame_id()</td><td>获取帧ID</td></tr><tr><td>get_timestamp()</td><td>获取时间戳</td></tr></table>

##  接口说明

##  convert

声明：

RawImage.convert(mode, flip=False, valid_bits=DxValidBit.BIT4_11, 

convert_type=DxBayerConvertType.NEIGHBOUR, 

channel_order=DxRGBChannelOrder.ORDER_RGB) 

意义：

图像格式转换。

1) 当mode = ’RAW8’模式时，将 16 位 raw图转换为8 位raw图，截取的有效位默认为当前像素格式的高8位。用户也可通过参数 valid_bits 手动设置有效位。仅支持10/12 位的 Raw图。

2) 当 mode = ’RGB’模式时，将 raw 图转换为 RGB 图。如果输入为 10/12 位 raw 图，先转换为 8 位raw图，再转换为RGB 图。

形参：

[in]mode ‘RAW8’: 将 16 位 raw 图转换为 8 位 raw 图‘RGB’: 将 raw 图转换为 RGB24 图

[in]flip 输出的RGB图像是否上下翻转，缺省值为False，该功能仅支持mode = ’RGB’模式[in]valid_bits 有效位数，缺省值为当前像素格式的高 8位，参考DxValidBit

[in]convert_type 转换类型，缺省值为 DxBayerConvertType.NEIGHBOUR，参考 DxBayerConvertType，仅对 mode = ’RGB’模式有效

[in]channel_order 图像通道顺序，缺省值为 DxRGBChannelOrder.ORDER_RGB，参考DxRGBChannelOrder

## 返回值：

## RGB图像对象

## 异常处理：

1) 如果帧信息状态不成功，则打印错误信息”RawImage.convert:This is a incomplete image”，函数返回 None。

2) 如果参数 1 不是字符串型，则抛异常 ParameterTypeError。

3) 如果参数 2 不是布尔型，则抛异常 ParameterTypeError。

4) 如果参数 3、4 不是整型，则抛异常 ParameterTypeError。

5) 如果参数 4 不在 DxBayerConvertType 中，则打印：提示参数越界、当前参数所支持的枚举值，函数返回None。

6) 如果参数4不在DxValidBit中，则打印：提示参数越界、当前参数所支持的枚举值，函数返回None。

7) 如果像素不是 8/10/12 位，则打印错误信息”RawImage.convert:This pixel format is not support”，函数返回None。

```txt
raw8_rotate_90_cw
声明：
RawImage.raw8_rotate_90_cw()
```

8) 如果参数 1 为‘RAW8’且参数 2 为 True，则打印错误信息”RawImage.convert:mode = ‘RAW8’don’t support flip = True”，函数返回 None。

9) mode =‘RAW8’，位深不是 10/12 位，则打印错误信息”RawImage.convert: mode="RAW8" onlysupport 10bit and 12bit”，函数返回 None。

10) 如果参数1 不为‘RAW8’或’RGB’，则打印接口名称和输入的mode不在范围内的信息，函数返回 None。

##  defective_pixel_correct

声明：

RawImage.defective_pixel_correct() 

意义：

对raw数据进行坏点校正

返回值：

```txt
None 
```

异常处理：

1) 如果坏点校正不成功，则抛出异常 UnexpetedError。

意义：

对8位图像顺时针旋转90度

返回值：

旋转后的 RawImage 图像对象

异常处理：

1) 如果旋转失败，则抛出异常 UnexpetedError。

```txt
raw8_rotate_90_ccw 
```

声明：

```txt
RawImage.raw8_rotate_90_ccw() 
```

意义：

对8位图像逆时针旋转90度

返回值：

旋转后的 RawImage 图像对象

异常处理：

1) 如果旋转失败，则抛出异常 UnexpetedError。

##  brightness

声明：

RawImage. brightness(factor) 

意义：

对8位灰度图像进行亮度调节

## 形参：

[in]factor 亮度调节因子，值范围-150~150。

其中： 0：亮度没有变化；

大于 0：增加亮度；

小于 0：减小亮度；

返回值：

None 

异常处理：

1) 如果输入参数不是整型，则抛出 ParameterTypeError 异常。

2) 如果图像类型不是 Mono8，则打印“RawImage.brightness only support mono8 image”，抛出InvalidParameter 异常。

3) 如果亮度调节失败，则抛出 UnexpetedError 异常。

##  contrast

声明：

RawImage.contrast(factor) 

意义：

对8位灰度图像进行对比度调节

形参：

[in]factor 对比度调节因子，值范围-50~150。

其中： 0：对比度没有变化；

大于0：增加对比度；

小于0：减小对比度；

返回值：

None 

异常处理：

1) 如果输入参数不是整型，则抛出 ParameterTypeError 异常。

2) 如果图像类型不是 MONO8，则打印“RawImage.brightness only support mono8 image”，抛出InvalidParameter 异常。

3) 如果对比度调节失败，则抛出 UnexpetedError 异常。

##  mirror

声明：

RawImage.mirror(mirror_mode) 

意义：

为8位图像产生一个与原图像在水平方向或者垂直方向相对称的镜像图像

形参：

[in]mirror_mode 图像镜像翻转方式，参考 DxImageMirrorMode。

返回值：

镜像后的 RawImage 图像对象

异常处理：

1) 如果输入参数不是整型，则抛出 ParameterTypeError 异常。

2) 如果不是 8 位图像格式，则抛出 InvalidParameter 异常。

3) 如果图像镜像失败，则抛出 UnexpetedError 异常。

##  get_ffc_coefficients

声明：

RawImage. get_ffc_coefficients(dark=None, target_value=None) 

意义：

计算图像平场校正系数，仅支持 8~12 位Raw图

形参：

[in]dark 暗场图像

[in]target_value 期望灰度值

返回值：

平场校正系数

异常处理：

1) 如果 dark 类型不是 RawImage，则抛出 ParameterTypeError 异常。

2) 如果 target_value 不是 INT 类型，则抛出 ParameterTypeError 异常。

3) 如果图像格式不是 8/10/12 位，则抛出 InvalidParameter 异常。

4) 如果获取平场校正系数调用失败，则抛出UnexpetedError 异常。

##  flat_field_correction

声明：

RawImage. flat_field_correction(ffc_coefficients) 

意义：

为8~12 位的Raw图进行平场校正操作

形参：

[in] ffc_coefficients 平场校正系数。

返回值：

None 

异常处理：

1) 如果 ffc_coefficients 不是 Buffer 类型，则抛出 ParameterTypeError 异常。

2) 如果图像格式不是 8/10/12 位图像格式，则抛出 InvalidParameter 异常。

3) 如果图像平场校正失败，则抛出 UnexpetedError 异常。

##  get_numpy_array

声明：

RawImage.get_numpy_array() 

意义：

将 raw 数据转换为 numpy 对象

返回值：

numpy 对象： 成功

None： 失败

异常处理：

1) 如果帧信息状态不成功，则打印错误信息”RawImage.get_numpy_array:This is a incompleteimage”，函数返回 None。

2) 如果像素格式不为8 位或16 位，则返回None。

 get_data 

声明：RawImage.get_data()

意义：获取raw数据

返回值：

raw数据[字符串型]

 save_raw 

声明：RawImage.save_raw( file_path)

意义：

保存raw图数据

形参：

```txt
获取 raw 图宽度
```

```txt
get_width 
```

```txt
None 
```

[in]file_path 文件路径。

例如：file_path = ’raw_image.raw’，则将 raw 图保存到当前工程路径下；

```txt
file_path = 'E://python_gxiapi/raw_image.raw'，则将 raw 图保存到绝对路径'E://python_gxiapi/下。
```

返回值：

异常处理：

1) 如果参数不是字符串格式，则抛出异常 ParameterTypeError。

2) 如果保存 raw 图数据未成功，则抛异常 UnexpectedError。

声明：

意义：

获取raw图状态

返回值：

```txt
raw 图状态，数据类型参考 GxFrameStatusList
```

声明：

意义：

返回值：

声明：

意义：

返回值：

声明：

意义：

获取图像像素格式

返回值：

像素格式

声明：RawImage.get_image_size()

意义：获取raw图数据大小

返回值：Raw图的大小

 get_frame_id声明：RawImage.get_frame_id()

意义：获取帧 ID

返回值：帧 ID

 get_timestamp声明：RawImage.get_timestamp()

意义：获取时间戳

返回值：

时间戳

## 3.4.6. Buffer

负责 Buffer 类的操作。Buffer 类将在图像质量提升的部分使用，Utility.get_gamma_lut(gamma)和Utility.get_contrast_lut(contrast)接口返回的 Buffer 类型对象将作为参数传给

RGBImage.image_improvement(color_correction_param=0, contrast_lut=None, gamma_lut=None)接口。

接口列表：

<table><tr><td>from_file(file_name)</td><td>从文件获取 Buffer 对象</td></tr><tr><td>from_string(string_data)</td><td>从字符串获取 Buffer 对象</td></tr><tr><td>get_data()</td><td>返回 Buffer 对象的字符串数据</td></tr><tr><td>get_ctype_array()</td><td>返回 Buffer 对象的数据数组</td></tr><tr><td>get_numpy_array()</td><td>返回 Buffer 对象的 numpy 数组</td></tr><tr><td>get_length()</td><td>返回 Buffer 对象的数据数组长度</td></tr></table>

```txt
from_file（静态函数）
声明：
Buffer.from_file(file_name)
意义：
从文件获取 Buffer 对象
形参：
[in]file_name 文件路径
返回值：
Buffer 对象
> from_string（静态函数）
声明：
Buffer.from_string(string_data)
意义：
从字符串获取 Buffer 对象
形参：
[in]string_data 字符串
返回值：
Buffer 对象
> get_data
声明：
Buffer.get_data()
意义：
返回 Buffer 对象的字符串数据
返回值：
string_data 字符串数据
注：python2.7：返回字符串类型；python3.5：返回 bytes 类型
> get_ctype_array
声明：
Buffer.get_ctype_array()
意义：
返回 Buffer 对象的数据数组
```

##  接口说明

返回值：

Buffer 对象的数据数组[ctype 类型]

```txt
get_numpy_array 
```

声明：

```txt
Buffer.get_numpy_array() 
```

意义：

返回 Buffer 对象的 numpy 数组

返回值：

Buffer 对象的数据数组[numpy 类型]

```txt
get_length 
```

声明：

```txt
Buffer.get_length() 
```

意义：

返回Buffer对象的数据数组长度

返回值：

数据数组长度

## 3.4.7. Utility

负责参数 gamma 和 contrast 的操作。

接口列表：

<table><tr><td>get_gamma_lut(gamma=1)</td><td>通过 gamma 值获取 gamma 查找表的 Buffer 类型对象</td></tr><tr><td>get_contrast_lut(contrast=0)</td><td>通过对比度值获取对比度查找表的 Buffer 类型对象</td></tr><tr><td>get_lut(contrast=0, gamma=1, lightness=0)</td><td>计算图像处理 8 位查找表</td></tr><tr><td>calc_cc_param(color_correction_param,saturation=64)</td><td>计算图像处理色彩调节数组</td></tr><tr><td>calc_user_set_cc_param(color_transform_factor, saturation=64)</td><td>根据用户设置计算图像处理色彩调节数组</td></tr></table>

##  接口说明

##  get_gamma_lut（静态函数）

声明：

Utility.get_gamma_lut(gamma=1)（静态函数）

意义：

通过 gamma 值获取 gamma 查找表的 Buffer 类型对象

形参：

[in]gamma 整型或浮点型，范围[0.1, 10.0]，缺省值为 1

返回值：

gamma 查找表的 Buffer 类型对象

异常处理：

1) 如果输入参数不是整型或浮点型，则抛出ParameterTypeError异常。

2) 如果输入参数不在 0.1~10.0 范围内，则打印错误信息”Utility.get_gamma_lut:gamma out of bounds,range:[0.1, 10.0]”，函数返回 None。

3) 如果获取 gamma 查找表失败，则打印接口名称、获取 gamma lut 失败和错误码的信息，函数返回 None。

##  get_contrast_lut（静态函数）

声明：

Utility.get_contrast_lut(contrast=0)（静态函数）

意义：

通过对比度值获取对比度查找表的Buffer类型对象

形参：

[in]contrast 整型，范围[-50, 100]，缺省值为 0

返回值：

对比度查找表的Buffer类型对象

异常处理：

1) 如果输入参数不是整型，则抛出 ParameterTypeError 异常。

2) 如果输入参数不在-50~100 范围内，则打印错误信息”Utility.get_contrast_lut:contrast out of bounds,range:[-50, 100]”，函数返回 None。

3) 如果获取对比度查找表失败，则打印接口名称、获取constrast lut失败和错误码的信息，函数返回None。

##  get_lut（静态函数）

声明：

Utility.get_lut(contrast=0, gamma=1, lightness=0)（静态函数）

意义：

计算图像处理8 位查找表

形参：

[in]contrast 对比度调节参数，整型，范围[-50, 100]，缺省值为 0

[in]gamma 调节参数，浮点型，范围[0.1, 10]，缺省值为 1

[in]lightness 亮度调节参数，整型，范围[-150, 150]，缺省值为 0

返回值：

查找表的Buffer类型对象

异常处理：

1) 如果 contrast/gamma/lightness 不是整型/浮点型/整型，则抛出 ParameterTypeError 异常。

2) 如果获取查找表失败，则打印接口名称、获取 lut 失败和错误码的信息，函数返回 None。

 calc_cc_param（静态函数）

声明：

Utility.calc_cc_param(color_correction_param, saturation=64)（静态函数）

意义：

计算图像处理色彩调节数组

形参：

[in]color_correction_param 颜色校正值，整型，可以从相机获取或者设为 0

[in]saturation 饱和度调节参数，整型，范围[0, 128]，缺省值为64

返回值：

色彩调节参数的Buffer类型对象

异常处理：

1) 如果输入参数不是整型，则抛出 ParameterTypeError 异常。

2) 如果计算色彩调节参数失败，则打印接口名称、获取调节参数失败和错误码的信息，函数返回None。

 calc_user_set_cc_param（静态函数）

声明：

Utility.calc_user_set_cc_param(color_transform_factor, saturation=64)（静态函数）

意义：

根据用户设置计算图像处理色彩调节数组

形参：

[in]color_transform_factor 颜色校正/颜色转换数组，列表或者元组类型

[in]saturation 饱和度调节参数，整型，范围[0, 128]，缺省值为64

返回值：

色彩调节参数的Buffer类型对象

异常处理：

1) 如果输入参数不是列表（元组）、整型，则抛出ParameterTypeError异常。

2) 如果计算色彩调节参数失败，则打印接口名称、获取调节参数失败和错误码的信息，函数返回None

## 4. 常见问题解答

<table><tr><td>序号</td><td>常见问题</td><td>解决办法</td></tr><tr><td>1</td><td>程序运行中出现如下错误”NotInitApi:DeviceManager.update_device_list:{-13}{Not init API}”</td><td>1) 请检查并删除程序中调用 DeviceManager 类对象的 __del__()函数的语句。因为 Python 的垃圾回收机制会自动调用 __del__()函数销毁对象,所以不需要、不允许用户显示调用 __del__()函数,如:“ device_manager.__del__()”。</td></tr></table>

## 5. 版本说明

<table><tr><td>序号</td><td>修订版本号</td><td>所做改动</td><td>发布日期</td></tr><tr><td>1</td><td>V1.0.0</td><td>初始发布</td><td>2018-08-10</td></tr><tr><td>2</td><td>V1.0.1</td><td>添加水星二代相机新增功能说明</td><td>2018-10-31</td></tr><tr><td>3</td><td>V1.0.2</td><td>修改部分标题,更正了部分不准确的描述</td><td>2019-04-12</td></tr><tr><td>4</td><td>V1.0.3</td><td>补充了部分描述</td><td>2019-05-07</td></tr><tr><td>5</td><td>V1.0.4</td><td>添加了掉线回调和采集回调的注册和注销接口,同步新的功能码和对应的数据类型</td><td>2021-03-04</td></tr><tr><td>6</td><td>V1.0.5</td><td>修改部分描述的问题,去掉不准确的描述</td><td>2021-03-08</td></tr><tr><td>7</td><td>V1.0.6</td><td>添加设备重连接口以及调节亮度、镜像等图像处理库接口</td><td>2021-05-27</td></tr></table>