# Baseline细节说明
### 文档说明
(dir)data:存放比赛的所有数据

(dir)model:存放神经网络的模型文件

config:配置文件

class_idx.py:本次比赛的类别是离散程度很大的值，class_idx中class_idx类可以实现由把目标类映射到输出空间或者把输出还原为目标类

word_idx:给所有的词生成onehot形式的编号

accurancy:其中的F1类通过save_data保存batch中生成的数据，caculate_f1计算f1值

Dataloader:加载文件

main:包含train()0和val()

test:加载模型并生成提交文件
### 使用说明
在seedcup目录下：python3 main.py即可开始训练，生成模型之后，用python3 test.py即可生成对应的提交文件
### 本次baseline的编写环境
python:3.6.6

torch 0.4.1
