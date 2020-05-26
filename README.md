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

- 图像读取：**Pillow (与 notebook 无缝集成)** / OpenCV (功能更强大)
- 数据扩增：为什么有用？(模型参数多，训练样本少) / 数据扩增方法 (颜色空间、尺度空间、样本空间) / 不同任务的数据扩增有相应区别，如本任务不能进行翻转操作 (6 和 9)
- 常用数据扩增库：**torchvision (与 torch 集成)** / imgaug / albumentations
- `torchvision.transforms`数据扩增方法：Resize / ColorJitter / RandomRotation / Normalize / ToTensor 等
- Pytorch读取数据：`Dataset`进行封装 / `DataLoader`并行读取 / 读取的数据格式 - 图像 (batchsize * chanel * height * wigth)，标签 (batchsize * max_len)

### Task03 - [字符识别模型](/nbs/Task03-字符识别模型.ipynb) 🎈

- CNN：一类模型 / CV领域常用 / 卷积操作的原理 / 一般由卷积 (convolution)、池化 (pooling)、非线性激活函数 (non-linear activation function)、全连接层 (fully connected layer) 构成 / 端到端 (End to End)
- CNN的发展：LeNet (1998) / AlexNet (2012) / VGG (2014) / Inception (2014) / ResNet (2015)
- pytorch中定义模型：继承`nn.Module` / 定义模型参数 / 定义`forward`函数
- 模型训练：大循环 epoch → 小循环 batch → 前向传播 → 计算 loss → 梯度清零 → 反向传播 → 更新参数
- 使用预训练模型：ImageNet数据集上的预训练模型 / `torchvision.models.resnet18(pretrained=True)` 

## 直播和答疑

- 第一次直播：[比赛介绍和baseline](https://tianchi.aliyun.com/course/video?liveId=41167) ([PPT](/PPT/天池直播-1_比赛介绍和baseline.pdf))
- 第二次直播：[图像数据读取，数据扩增方法和图像识别模型介绍](https://tianchi.aliyun.com/course/live?spm=5176.12586971.1001.1.11be6956fkKgJ8&liveId=41168) (PPT)
- 第三次直播：
- 答疑汇总：https://shimo.im/docs/5zAZVYlY4RF5FgAo

## 参考资料

Datawhale开源学习资料：https://github.com/datawhalechina/team-learning 

天池比赛论坛：https://tianchi.aliyun.com/competition/entrance/531795/forum

