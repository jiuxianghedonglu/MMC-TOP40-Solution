# 瑞金医院MMC人工智能辅助构建知识图谱大赛解决方案（TOP 40/1629）

## [大赛连接](https://tianchi.aliyun.com/competition/entrance/231687/introduction)

## 初赛 (TOP 47/1629)

- 将文章按照中文逗号，句号，问号，感叹号切分为句子
- 标注使用了BIO标注格式
- Embedding使用的是字的Embedding，没有进行分词处理
- 模型主体是Embedding之后接 Bi-LSTM+CRF

## 复赛 (TOP 40/1629)

- 将问题转化为二分类问题：即两个实体之间是否有关系
- 训练集的构造方式：遍历每个实体，对于当前遍历的实体在一定的窗口范围内取多个后面的实体，对于每个实体对，两个实体之间的部分构成一个句子，为了考虑到更多的上下文关系，左边实体的左侧多取一部分，右边实体的右侧多取一部分拼接起来作为x1，左侧实体作为x2,右侧实体作为x3，两个实体之间是否有关系作为y
- 经过上一步即可得到训练集:X（x1,x2,x3);Y 0或1
- 测试集只需构造X，与训练集构造方式相同
- 模型方面。由于时间原因，只测试了比较简单的方法：将x1,x2,x3分别输入同一个Encoder模型,对三个Encoder之后的向量做了dot,concat等一些操作，再接全连接进行分类


## 环境

- 系统：Windows 7

- GPU：1080Ti+cuda:9.0+cudnn:7.0.5

- Python:3.6.5

- pickle:0.7.4

- keras:2.1.6

- keras_contrib:2.0.8

- tensorflow:1.9

- numpy:1.14.3

- pandas:0.23.0

- sklearn:0.19.1

## 运行方式

1. 在data文件夹下新建文件夹train,test_b,将解压后的训练集数据放入./data/train/,测试集b榜数据放入./data/test_b/
此时data的文件目录为./data/train/.txt  ./data/train/.ann  ./data/test_b/.txt

2. 在code文件夹下执行命令：python main.py

3. 预测结果保存在submit文件夹下