# 14-days-CV-team-learning 🎨

2020年5月Datawhale第十三期组队学习挑战    

学习内容为：零基础入门CV赛事之[**街景字符识别**](https://tianchi.aliyun.com/competition/entrance/531795/introduction)，来自 Datawhale与天池联合发起的“0基础入门系列”赛事第二场。   

**Star⭐ me if you find useful🤣**

## 学习内容与进度 📙

详细的个人笔记和注解见[notebook](/nbs/)中批注，以下是每个task的总结 👇

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

## 参考资料

Datawhale开源学习资料：https://github.com/datawhalechina/team-learning 

天池比赛论坛：https://tianchi.aliyun.com/competition/entrance/531795/forum

