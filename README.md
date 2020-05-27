# 14-days-CV-team-learning 🎨

2020年5月Datawhale第十三期组队学习挑战    

学习内容为：零基础入门CV赛事之[**街景字符识别**](https://tianchi.aliyun.com/competition/entrance/531795/introduction)，来自 Datawhale与天池联合发起的“0基础入门系列”赛事第二场。   

**Star⭐ me if you find useful🤣**

## 学习内容与进度 📙

详细的个人笔记和注解见[notebook](/nbs/)中批注，以下是每个task的总结 👇

### Task01 - [Baseline](/nbs/Task00-Baseline.ipynb) 🎈

- 基本思路：CNN 定长字符识别 / 数据读取 (Dataset 和 DataLoder) → 模型构建 (ResNet18) → 训练验证 → 结果预测
- 实验：15个epoch (成绩 0.57) / 
- 如何提高精度：增加 epoch / 其他数据扩增方法 (模糊、像素噪音等) / 多折预测结果集成 / 其他模型 / 修改损失函数 / 增加字数预测模型对结果修正

### Task01 - [赛题理解](/nbs/Task01-赛题理解.ipynb) 🎈

- 赛题任务：街道字符识别 / CV
- 数据集：公开数据集 [SVHN](http://ufldl.stanford.edu/housenumbers/) / .png / train (3w) / val (1w) / test A (4w) / test B (4w)
- 数据标签：.json / 字符框位置 - top, height, left, width / 标签 - label / 同一张图片可包含多个字符
- 评测指标：Score = 编码识别正确的数量 / 测试集图片数量
- 解题思路：**定长字符识别 (全部填充至最长字符)** / 不定长字符识别 / 先检测再识别

### Task02 - [数据读取与数据扩增](/nbs/Task02-数据读取与数据扩增.ipynb) 🎈

- 图像读取：**Pillow (与 notebook 无缝集成)** / OpenCV (功能更强大) / matplotlib.image / scipy.misc / skimage
- Pytorch读取数据：常用数据集 `torchvision.datasets.CIFAR10(...)` / 自定义数据集用`Dataset`进行封装 (实现 `_getitem_` 和 `_len_  `方法)， `DataLoader`批量读取 / 读取的数据格式 - 图像 (batchsize * chanel * height * wigth)，标签 (batchsize * max_len)
- 数据扩增：为什么有用？(模型参数多，训练样本少) / 不同任务的数据扩增有相应区别，如本任务不能进行翻转操作 (6 和 9)
<<<<<<< HEAD
  > 基于图像处理的数据扩增方法:
  >
=======

  > 基于图像处理的数据扩增方法:
  > 
>>>>>>> d8dbdc2dbc84becadbd031f5f19e634530d67f7d
  > - 几何变换：旋转 / 缩放 / 翻转 / 裁剪 / 平移 / 仿射变换
  > - 灰度和彩色空间变换：亮度调整 / 对比度、饱和度调整 / 颜色空间转换 / 色彩调整
  > - 添加噪声和滤波：注入高斯噪声、椒盐噪声 / 滤波 - 模糊、锐化等
  > - 图像混合：多用于目标检测
  > - 随机擦除
  > 
  > 基于深度学习的数据扩增方法：GAN数据增强 / 神经风格转换 / AutoAugment
- 常用数据扩增库：**torchvision (与 torch 集成)** / imgaug / albumentations
- `torchvision.transforms`数据扩增方法：Resize / ColorJitter / RandomRotation / Normalize / ToTensor 等，具体见 [PPT](/PPT/天池直播-2_数据读取和数据扩增.pdf)

### Task03 - [字符识别模型](/nbs/Task03-字符识别模型.ipynb) 🎈

- 早期图像分类模型：近邻算法 (K近邻) / 线性分类 / 多层感知机
- CNN：卷积操作原理 (卷积核、感受野) / 一般由卷积 (convolution)、池化 (pooling)、非线性激活函数 (non-linear activation function)、全连接层 (fully connected layer) 构成 / 端到端 (End to End)
- CNN的发展：
  - **LeNet (1998)**：input=32 * 32 * 1 / conv - filter=5 * 5，stride=1 / pooling - filter=2 * 2，stride=2
  - **AlexNet (2012)**：input=227  *  227  *  3 / conv1 - filter=11 * 11，stride=4，96个卷积核
  - **VGG (2014)**：全部使用 3 * 3 卷积核 / 参数较多
  - **GoogLeNet (2014)**：22层 / Inception模块，并行4路操作，3卷积1池化，输出长和宽相同，1 * 1卷积的使用 / 无全连接层，使用全局平均池化
  - **ResNet (2015)**：基本单元 - 残差块 (Residual block)，2个3 * 3卷积 / 周期性2倍增卷积核组数 / 只有一个全连接层
- pytorch中定义模型：继承`nn.Module` / 定义模型参数 / 定义`forward`函数
- 模型训练：大循环 epoch → 小循环 batch → 前向传播 → 计算 loss → 梯度清零 → 反向传播 → 更新参数
- 使用预训练模型：ImageNet数据集上的预训练模型 / `torchvision.models.resnet18(pretrained=True)` 

### Task04 - [模型训练与验证](/nbs/Task04-模型训练与验证.ipynb) 🎈

- 过拟合现象：训练误差降低，测试误差上升 / 常见原因：模型复杂度太高 / 解决方法：构建验证集验证模型精度和调参
- 验证集划分：本赛题以给出验证集，可以用训练集训练，验证集验证；也可以合并训练集验证集自行划分
  - 留出法 (Hold-Out)：直接将训练集划分为两部分 / 优点：简单直接 / 缺点：可能在验证集上过拟合 / 适用于数据量较大情况
  - 交叉验证法 (Cross Validation)：训练集分为K份，K-1份为训练集，1份为验证集，循环训练K次 / 优点：精度较可靠 / 缺点：训练K次 / 适用于数据量不大的情况
  - 自助采样法 (BootStrap)：有放回的采样方式得到新的训练集验证集 / 适用于数据量较小情况
- pytorch训练和验证：定义train模块 / 定义validate模块 / 循环训练epochs次
- 模型保存加载：`torch.save(model_object.state_dict(),'model.pt')` / `model.load_state_dict(torch.load(' model.pt'))`
- 本任务推荐的提升模型效果流程：构建简单CNN模型，跑通训练、验证和预测流程 → 增加模型复杂度 → 增加数据扩增方法

## 直播和答疑

- 第一次直播：[比赛介绍和baseline](https://tianchi.aliyun.com/course/video?liveId=41167) ([PPT](/PPT/天池直播-1_比赛介绍和baseline.pdf))
- 第二次直播：[图像数据读取，数据扩增方法和图像识别模型介绍](https://tianchi.aliyun.com/course/live?spm=5176.12586971.1001.1.11be6956fkKgJ8&liveId=41168) ([PPT1](/PPT/天池直播-2_数据读取和数据扩增.pdf)，[PPT2](/PPT/天池直播-2_分类模型介绍.pdf))
- 第三次直播：
- 答疑汇总：https://shimo.im/docs/5zAZVYlY4RF5FgAo

## 参考资料

Datawhale开源学习资料：https://github.com/datawhalechina/team-learning 

天池比赛论坛：https://tianchi.aliyun.com/competition/entrance/531795/forum

