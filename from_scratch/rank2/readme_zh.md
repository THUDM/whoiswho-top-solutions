# 复现说明
## 运行环境
Win10
Anaconda3(64bit)
python3.7
jupyter notebook环境

## 运行库
python==3.7.0 
genism==3.4.0 
numpy==1.20.3 
scikit-learn==0.19.2

## 运行步骤
1. 准备环境：Win10 + Anaconda3(64bit) + python3.7
2. 数据集准备：在一个路径下存放下载好的数据并解压：
   /train, /sna-valid, /sna-test; 并修改word2vec下cell为对应路径
   如base = '/home/data'
3. 在 sna.ipynb 中自上而下按顺序运行每个cell 
4. 运行完毕，在genetest 文件夹中得到消歧结果的文件：result_test.json 
注：name disambiguation (train)可不运行，该部分用于调参测试。

## 文件结构
```
├── gene       #（生成）存放各种中间文件或临时文件      
├── genename   #（生成）存放每个消歧名字对应的论文集合
├── genetest   #（生成）存放生成的消歧结果
├── word2vec    #（生成）存放训练得到的word2vec模型
├── data 
      ├── test_data  #测试集
      ├── sna_data   #验证集
      └── train      #训练集
├── sna.ipynb
      ├── utils      #导入库、函数等，包括基于元路径的随机游走、关系提取和匹配函数
      ├── word2vec   #训练并保存word2vec模型
      ├── name disambiguation(test)   #运行得到测试结果
      └── name disambiguation(train)  #运行得到训练结果

```
### 提交版本用到的代码
* sna.ipynb

 
 
 