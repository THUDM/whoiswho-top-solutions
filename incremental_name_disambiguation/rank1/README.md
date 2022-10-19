# WhoIsWho Task1-kingsundad Top-1 Solution

## Prerequisites


> CUDA Version: 10.1  
>
> Python 3.7

Please install the following Python packages (in requirements.txt)

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

## Setup

`paper_idf_dir` is the directory for *token IDF* values. (It can be downloaded from https://pan.baidu.com/s/1g1w2m20V4WPj0YNGYyF8Tw  password: y2ws)

`data_root` reprensents the data directory and its structure follows:

```
data/
├── processed_data
├── raw_data
│   ├── cna-test
│   │   ├── cna_test_unass.json
│   │   └── cna_test_unass_pub.json
│   ├── cna-valid
│   │   ├── cna_valid_example.json
│   │   ├── cna_valid_ground_truth.json
│   │   ├── cna_valid_unass.json
│   │   ├── cna_valid_unass_pub.json
│   │   ├── whole_author_profiles.json
│   │   └── whole_author_profiles_pub.json
│   ├── readme.md
│   └── train
│       ├── train_author.json
│       └── train_pub.json
├── cna-test
├── cna-valid
└── train

```

`feat_root` is the directory to save customized features.

## Reproduce the results quickly

Please download the processed data from https://pan.baidu.com/s/18PltLZ0XVnr_cel8_fkR5Q with password tj7p.

Unzip `submit_data.zip`. Put two directories `data` and `feat` directories in `submit_data` into `whoiswho-top-solutions/incremental_name_disambiguation/rank1/
`.

```
sh script/ml_main.sh
```

Output files:

- `result.v1.json` for cna-valid 
- `result.v2.json` for cna-test 

Pretrained models and intermediate results are in the `save_model` directory.

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

