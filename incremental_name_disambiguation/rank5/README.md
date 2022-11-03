# Rank5 Solution (Team Name: 数据魔术师)

## Environment
- Python 3.7
- pip install -r requirements.txt

## Data preparation
Create `data` folder in current directory and download raw data from http://whoiswho.biendata.xyz/#/data. The data organization is:

```
├── data
│   └── na_v3
│       ├── cna-test
│       │   ├── cna_test_unass.json
│       │   └── cna_test_unass_pub.json
│       ├── cna-valid
│       │   ├── cna_valid_example.json
│       │   ├── cna_valid_ground_truth.json
│       │   ├── cna_valid_unass.json
│       │   ├── cna_valid_unass_pub.json
│       │   ├── whole_author_profiles.json
│       │   └── whole_author_profiles_pub.json
│       └── train
│           ├── train_author.json
│           └── train_pub.json
```

## Running steps
```bash
python code/step1_data_prepare.py 
python code/step2_split_data.py 
python code/step3_extract_base_feature.py 
python code/step4_extract_coauthors_feature.py 
python code/step5_extract_org_feature.py 
python code/step6_extract_word2vec_sim_feature.py 
python code/step7_extract_weight_word2vec_sim_feature.py 
python code/step8_extract_text_countvec_sim_feature.py 

python code/step9_extract_weight_countvec_sim_feature.py 
python code/step10_extract_text_tfidfvec_sim_feature.py 
python code/step11_extract_weight_tfidfvec_sim_feature.py 
python code/step12_extract_keywords_abstract_feature.py 
python code/step13_extract_year_feature.py 
python code/step14_extract_set_feature.py 
python code/run_lightgbm.py 
python code/run_xgboost.py 
python code/merge.py
```

Results are saved in `work/submit/online/tree/XGBoost/lgb_testa.json`
