# Rank 2 Solution (Team Name: AlexNE)

## Environments
- Python 3.7
- ``` pip install -r requirements.txt```
- torch 1.10.0 + cu111

## Data preparation
Create `datas` folder in current directory and download raw data from http://whoiswho.biendata.xyz/#/data. The data organization is:

```
├── datas
│   ├── Task1
│   │   ├── cna-test
│   │   │   ├── cna_test_unass.json
│   │   │   └── cna_test_unass_pub.json
│   │   ├── cna-valid
│   │   │   ├── cna_valid_example.json
│   │   │   ├── cna_valid_ground_truth.json
│   │   │   ├── cna_valid_unass.json
│   │   │   ├── cna_valid_unass_pub.json
│   │   │   ├── whole_author_profiles.json
│   │   │   └── whole_author_profiles_pub.json
│   │   └── train
│   │       ├── train_author.json
│   │       └── train_pub.json
```

## Running Steps
```bash 
python baseline/splitProUnass.py
python baseline/evaluation/processCandidate.py
```

Download files from https://pan.baidu.com/s/1g1w2m20V4WPj0YNGYyF8Tw with passwd: y2ws and put them into `paper_idf` directory.
Download files from https://pan.baidu.com/s/11L3wOSBn2HfHrvNbOJQdoA with passwd: 1snm and put `oagbert-v2-sim` into `path_to_oagbert` directory.  
Download GloVe embeddings from https://www.kaggle.com/datasets/takuok/glove840b300dtxt and put it into current directory.
```bash
cd ...
export PYTHONPATH="`pwd`:$PYTHONPATH"
cd incremental_name_disambiguation/rank2
python baseline/pre_gen_data.py
python baseline/evaluation/get_oag_embeddings.py
python build_glove_embeds.py
python name_utils.py
python get_unsupervised_features.py
python node2vec/build_graph.py
python node2vec/Node2Vec.py
python ml_methods.py
python k_fold_pineline.py
```

Results are saved in `results` folder.

#### Code Structure
```
.
├── datas
│   └── Task1
│       ├── cna-test
│       ├── cna-valid
│       └── train
├── baseline
│   ├── character
│   │   ├── feature_config.py
│   │   ├── feature_process.py
│   │   └── name_match
│   │       ├── data
│   │       │   └── test_data.json
│   │       ├── main.py
│   │       ├── test.py
│   │       └── tool
│   │           ├── const.py
│   │           ├── __init__.py
│   │           ├── interface.py
│   │           ├── is_chinese.py
│   │           ├── match_name.py
│   │           ├── token.py
│   │           └── util.py
│   ├── compData
│   │   ├── prepared_test_data_1.pkl
│   │   └── prepared_train_data_1.pkl
│   ├── datas
│   │   ├── {}_proNameAuthorPubs.jon (train/test)
│   │   ├── proNameAuthorPubs.json
│   │   ├── {}_data.pkl (train/test)
│   │   ├── {}_author_profile.json (train/test)
│   │   ├── {}_author_unass.json (train/test)
│   │   ├── {}_feature_data.pkl (train/test)
│   │   ├── {}_embedding_data.pkl (train/test)
│   │   ├── {}_oag_embeddings.pkl (train/test/whole/valid_unass/test_unass)
│   │   ├── {}_sim_data.pkl (train/test)
│   │   ├── {}_unass_CandiAuthor_add_sim.pkl (valid/test)
│   │   ├── {}_unassCandi.json (valid/test)
│   │   ├── {}_unass_featData_add_sim.pkl (valid/test)
│   │   └── {}_unass_simData_add_sim.pkl (valid/test) 
│   ├── evaluation
│   │   ├── evalFeatureMain.py
│   │   ├── get_oag_embeddings.py
│   │   └── processCandidate.py
│   ├── paper_idf
│   │   ├── name_uniq_dict.json
│   │   ├── new_org_idf.json
│   │   ├── title_idf.json
│   │   └── venue_idf.json
│   ├── semantic
│   │   ├── config.py
│   │   └── model.py
│   ├── README.md
│   ├── data_process.py
│   ├── dl_main.py
│   ├── ml_main.py
│   ├── pre_gen_data.py
│   ├── splitProUnass.py
│   └── whole_config.py
├── models
│   ├── 5_fold
│   │   ├── caboost_fold_{}.json (0/1/2/3/4)
│   │   └── xgboost_fold_{}.json (0/1/2/3/4)
│   ├── caboost.json
│   └── xgboost.json
├── node2vec
│   ├── build_graph.py
│   ├── __init__.py
│   ├── Node2Vec.py
│   └── Node2Vec_whoiswho.py
├── resource
│   ├── add_features_to_official_features
│   │   ├── {}_aid2len.pkl (abstract/coauthor/org/title/keywords)
│   │   ├── {}_ngram_cnt.pkl (abstract/coauthor/org/title/keywords)
│   │   ├── {}_author_{}_ngram_weights.pkl (train/test/valid_unass/test_unass) (abstract/coauthor/org/title/keywords)
│   │   ├── {}_author_{}_weights.pkl (train/test/valid_unass/test_unass) (org/keywords/coauthor)
│   │   ├── {}_glove_similarity.pkl (train/test/valid_unass/test_unass)
│   │   └── {}_predict_features.pkl (train/valid/valid_unass/test_unass)
│   ├── glove_embeddings
│   │   ├── authors
│   │   └── papers
│   ├── node2vec
│   │   ├── {}_paper_keywords.pkl (train/test/whole/valid_unass/test_unass)
│   │   ├── {}_author_top_keywords.pkl (train/test/whole)
│   │   ├── {}_graph.txt (train/test/valid_unass/test_unass/all)
│   │   ├── node2vec_fast_all_50_walklen.bin
│   │   ├── node2vec_fast_test_20_walklen.bin
│   │   ├── node2vec_fast_test_unass_20_walklen.bin
│   │   ├── node2vec_fast_train_20_walklen.bin
│   │   └── node2vec_fast_valid_unass_20_walklen.bin
│   ├── {}_author_coauthors.json (train/whole)
│   ├── {}_author_coorgs.json (train/whole)
│   ├── {}_author_keywords.pkl (train/whole)
│   ├── {}_author_keywords_tfidf.pkl (train/whole)
│   ├── {}_author_orgs.pkl (train/whole)
│   ├── {}_author_orgs_tfidf.pkl (train/whole)
│   ├── test_unassCandi.json
│   ├── train_pub_keywords.json
│   ├── unassCandi.json
│   └── valid_unassCandi.json
├── results
│   └── task1
│       ├── test
│       └── valid
├── name_utils.py
├── build_glove_embeds.py
├── data_utils.py
├── get_unsupervised_features.py
├── k_fold_pineline.py
├── ml_methods.py
├── requirements.txt
├── path-to-oagbert
└── readme.md
```
