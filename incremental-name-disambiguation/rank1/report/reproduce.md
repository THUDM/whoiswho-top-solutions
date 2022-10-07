# IJCAI 2021 - WhoIsWho Task1-kingsundad 复现说明

## 代码执行环境说明

代码运行环境为

> CUDA Version: 10.1  
>
> Ubuntu 16.04 LTS
>
> Python 3.6.12

需要安装如下python库(在requirements.txt中)

> xgboost==0.90
> numpy==1.19.2
> pypinyin==0.42.0
> tqdm==4.36.1
> Unidecode==1.2.0
> pyjarowinkler==1.8
> lightgbm==2.3.0
> scipy==1.5.3
> torch==1.7.1
> gensim==3.8.3
> cogdl==0.4.0
> catboost==0.19.1
> scikit_learn==0.24.2

## 代码执行前配置

在执行代码之前，需要配置`whole_config.py`中以下路径为合适的值

```
data_root = './data/'
feat_root = './feat/'
paper_idf_dir = './paper_idf/'
```

`paper_idf_dir` 为下载的 IDF 文件存放目录路径。（可以在 https://pan.baidu.com/s/1g1w2m20V4WPj0YNGYyF8Tw  passwd: y2ws 下载）

`data_root`为存放数据集的目录，其结构如下

> data/
> ├── processed_data
> └── raw_data
>  ├── cna-test
>  │   ├── cna_test_unass.json
>  │   └── cna_test_unass_pub.json
>  ├── cna-valid
>  │   ├── cna_valid_example.json
>  │   ├── cna_valid_unass.json
>  │   ├── cna_valid_unass_pub.json
>  │   ├── whole_author_profiles.json
>  │   └── whole_author_profiles_pub.json
>  └── train
>  │   ├──  train_author.json
>  │   ├── train_pub.json

`feat_root`为自定义的特征文件存放路径。

## 快速复现结果

由于特征生成等步骤执行较慢，这里放出我们当时产生的特征及训练文件，可通过如下方式下载

- 百度网盘

  链接：https://pan.baidu.com/s/18PltLZ0XVnr_cel8_fkR5Q 
  提取码：tj7p

解压好下载的 `submit_data.zip` 后得到`submit_data`文件夹，用`submit_data`文件夹里的`data`和`feat`两个文件夹分别将项目中的`data`和`feat`文件夹进行替换。

```
sh script/ml_main.sh
```

`ml_main.sh` 负责的工作是训练模型和利用模型来预测测试集的结果。其中的 `final_cell_list_config` 即为最终使用的模型结构配置。

就可以在当前路径下看见对应的结果文件。

- `result.v1.json` 为 cna-valid 对应的结果
- `result.v2.json` 为 cna-test 对应的结果

训练好的模型及一些中间结果存放在`save_model`文件夹下。

## 代码执行步骤

提交的代码大体上分为如下三个部分

1. 数据预处理
2. 特征生成
3. 训练及预测

每一步执行的详细描述如下。

### 下载OAG-BERT预训练语言模型

处理特征的时候需要用到OAG-BERT，因此需先下载。

- 百度网盘

链接: https://pan.baidu.com/s/11L3wOSBn2HfHrvNbOJQdoA  
密码: 1snm  

解压好下载的 `oagbert-v2-sim.zip` 后得到`oagbert-v2-sim`文件夹，将`oagbert-v2-sim`文件夹放入项目根目录下的`saved`文件夹下。

### 数据预处理

```
sh script/data_process.sh
```

`data_process.sh` 负责的工作是处理原始数据集，产生多折的训练集和验证集及其他各个需要的中间文件。

### 特征生成

依次执行如下脚本

```
sh script/get_hand_feat.sh
sh script/get_graph_simi_feat.sh
sh script/get_paper_emb.sh
sh script/get_bert_simi_feat.sh
```

`get_hand_feat.sh` 负责的工作是为各个待分配论文及其对应的候选者产生合适的手工特征。

`get_graph_simi_feat.sh` 负责的工作是为各个待分配论文及其对应的候选者产生基于论文网络嵌入的相似度特征。  

`get_paper_emb.sh` 负责的工作是按照作者名称，对同名的候选作者发过的论文用OAG-BERT计算论文的表征且暂存，目的是为了后续计算bert_simi feature时，避免对同一篇论文重复计算表征。  

`get_bert_simi_feat.sh` 负责的工作是为各个待分配论文及其对应的候选者发过的论文集合计算相关性特征，该相关性特征称为bert_simi feature。  

### 训练及预测

```
sh script/ml_main.sh
```

就可以在当前路径下看见对应的结果文件。

- `result.v1.json` 为 cna-valid 对应的结果
- `result.v2.json` 为 cna-test 对应的结果

训练好的模型及一些中间结果存放在`save_model`文件夹下。

