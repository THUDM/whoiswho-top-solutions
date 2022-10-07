## IJCAI2021 - WhoIsWHo Task1
##### 运行说明
1. 运行IJCAI-21-WhoIsWho-baseline-main/splitProUnass.py划分训练和测试集
2. 运行IJCAI-21-WhoIsWho-baseline-main/evaluation/processCandidate.py获取每个名字对应的作者和论文
3. 运行IJCAI-21-WhoIsWho-baseline-main/pre_gen_data.py得到训练集和测试集的正负样本、手工特征以及oag embedding
   在运行这一步前，请在https://pan.baidu.com/s/1g1w2m20V4WPj0YNGYyF8Tw  passwd: y2ws 下载数据，放入paper_idf中
4. 运行IJCAI-21-WhoIsWho-baseline-main/evaluation/get_oag_embedding.py得到valid_unass和test_unass的oag embedding，以及每个数据的oag matching similarity
    在运行这一步前，请将oagbert-v2-sim放置于IJCAI-21-WhoIsWho-baseline-main/path_to_oagbert下
    - 百度网盘
    链接: https://pan.baidu.com/s/11L3wOSBn2HfHrvNbOJQdoA  
    密码: 1snm

5. 运行build_glove_embeds.py获取所有数据集论文的glove embedding特征
   在运行这一步前，从https://www.kaggle.com/datasets/takuok/glove840b300dtxt
   把下载的文件放在主文件夹下
6. 运行name_utils.py获取所有数据集的作者（机构/关键词/共同作者）特征
<!-- TOHERE -->
7. 运行get_unsupervised_features.py获取额外构建的无监督特征（机构词交集、关键词交集、共同作者交集、glove余弦相似度，以及各种ngram tfidf特征）
<!-- TOHERE -->
8. 运行node2vec/build_graph.py构建图
9. 运行node2vec/Node2vec.py训练node2vec并得到node2vec向量
10. 运行ml_methods.py进行单个模型训练
11. 运行k_fold_pineline.py进行k_fold训练并得到最终结果
（注：以上1、3、9步骤由于没有固定随机种子，不保证每次运行结果一致。如有需要，我们可以后续提供）


#### 运行环境
见 requirements.txt


#### 代码结构
```
.
├── datas
│   └── Task1
│       ├── cna-test
│       ├── cna-valid
│       └── train
├── IJCAI-21-WhoIsWho-baseline-main
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
│   ├── path_to_oagbert
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
└── readme.md
```

注：代码结构中展示的是完整复现出结果所需的所有文件。除了模型文件和结果文件外，所有中间结果的json、pkl、bin格式的文件因为较大，暂不提供，可按照代码运行步骤重新生成。
