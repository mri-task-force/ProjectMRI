# 更新记录

## 16/5/2019

> by wu

0. 添加了 `settings.py` ，全局变量包括文件名、文件路径、log对象，都放在这里

1. 数据集的 loader 添加了 `num_workers=4`，加快速度

2. tensorboard 画混淆矩阵

   安装 `tbplot`

   `pip install tensorflow-plot`


## 10/5/2019

> add by peng
0. 添加了updates.md 记录每次更新；
1. 将load_dataset_v3改名为laod_dataset；
2. models库添加模型alexnet、densenet、inception、mobilenet、resnet、squeezenet、vggnet；
3. 尝试用pre-trained模型进行训练；
4. 为了使用预训练模型，将图片resize到224*224
5. 
