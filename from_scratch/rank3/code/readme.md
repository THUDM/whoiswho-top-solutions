# 复现说明
## 运行库
transformers==4.2.1
scipy==1.7.3
datasets==1.2.1
pandas==1.1.5
scikit-learn==0.24.0
prettytable==2.1.0
setuptools==63.3.0
gensim==3.4.0
numpy==1.21.6
cogdl==0.5.3
torch==1.12.0
tqdm==4.36.1

## 运行步骤
1. 数据集准备：竞赛官方数据：https://www.aminer.cn/whoiswho
在一个路径下存放下载好的数据并解压：data/train, data/sna-valid, data/sna-test; 
并修改train.py line 142, predict line 138为对应路径，如 base = '/home/data'
2. 训练模型，调参：执行train.py
3. 验证数据消歧，并保存结果：执行predict.py, 在./saveResult/文件夹下得到提交文件

## 文件结构
```
├── save         #(初始为空)保存各种中间文件或临时文件      
├── saveName     #(初始为空)存放每个消歧名字对应的论文集合
├── saveResult   #(初始为空)生成的测试集的结果
├── data 
      ├── test_data  #(初始为空)测试集
      ├── sna_data   #(初始为空)验证集
      └── train      #(初始为空)训练集
├── utils.py     #导入运行所需的库，类及函数等工具
├── train.py     #对训练集中的数据进行消歧并调参
├── predict.py   #对验证集中的数据进行消歧并得到结果（保存在genetest中）
└── 数据分析.ipynb   #数据分析notebook

```
## 提交版本用到的代码
* utils.py
* train.py
* predict.py

## 未用到
* 数据分析.ipynb