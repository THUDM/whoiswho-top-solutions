# WhoIsWho Task1 Top-1 Solution (Team Name: kingsundad)

## Prerequisites


> Python 3.7
> 
> torch 1.10.0 + cu111

Please install the following Python packages (```pip install -r requirements.txt```)

> xgboost==0.90
> numpy==1.19.2
> pypinyin==0.42.0
> tqdm==4.36.1
> Unidecode==1.2.0
> pyjarowinkler==1.8
> lightgbm==2.3.0
> scipy==1.5.3
> gensim==3.8.3
> cogdl==0.4.0
> catboost==0.19.1
> scikit_learn==0.24.2

## Setup

`paper_idf_dir` in current folder is the directory for *token IDF* values. (It can be downloaded from https://pan.baidu.com/s/1g1w2m20V4WPj0YNGYyF8Tw  password: y2ws)

`data` reprensents the data directory.

`feat` is the directory to save customized features.

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

## Complete Running Steps

The submitted code is roughly divided into the following three parts:

1. Data preprocessing
2. Feature Generation
3. Training and prediction

A detailed description of the execution of each step is as follows.

### Download OAG-BERT Pretraining Model

Please download the OAG-BERT model from BaiduPan with url https://pan.baidu.com/s/11L3wOSBn2HfHrvNbOJQdoA and password **1snm**. Unzip `oagbert-v2-sim.zip` and put `oagbert-v2-sim` into the `saved` directory of the project root directory.

Make sure that the `data` directory in current folder is like follows:

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


### Preprocess

```bash
sh script/data_process.sh
```

`data_process.sh` is to process the original data set, generate multi-fold training set and validation set and other required intermediate files.

### Feature Generation

Execute the following scripts one by one.

```bash
sh script/get_hand_feat.sh
sh script/get_graph_simi_feat.sh
cp data/raw_data/train/train_pub.json data/processed_data/train/
sh script/get_paper_emb.sh
sh script/get_bert_simi_feat.sh
```


### Training and Inference

```bash
sh script/ml_main.sh
```

Output files:

- `result.v1.json` for cna-valid 
- `result.v2.json` for cna-test 

Pretrained models and intermediate results are in the `save_model` directory.

