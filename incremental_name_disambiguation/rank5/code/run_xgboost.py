 # -*- coding: utf-8 -*-
import os
import gc
import pandas as pd

# 自定义库
from args import get_parser
from utils import timer,reduce_mem_usage
from treeModels.lightgbm import kfold_lightgbm
from treeModels.xgboost import kfold_xgboost

def run_model(): 
    featpath = args.featpath+"/data_features_without_tfidfvec_time.pkl"
    if not os.path.exists(featpath):
        # In[]
        # 构造训练集特征
        all_train_feats = pd.DataFrame()
        for mode in ['na_v3']:
            print("Merge {} train data".format(mode))
            train_df = pd.read_pickle(args.featpath+"/{}_train.pkl".format(mode))
            
            # 构造与合作者相关的特征
            train_coauthors_feats = pd.read_pickle(args.featpath+"/{}_train_accurate_coauthors_feature.pkl".format(mode))
            train_feats = pd.merge(train_df,train_coauthors_feats,on=['idx'],how='left')
            
            # 构造与论文机构相关的特征
            train_org_feats = pd.read_pickle(args.featpath+"/{}_train_org_feature.pkl".format(mode))
            train_feats = pd.merge(train_feats,train_org_feats,on=['idx'],how='left')
            
            # 构造与论文title相关的特征
            train_title_feats = pd.read_pickle(args.featpath+"/{}_train_title_word2vec_sim_feature.pkl".format(mode))
            train_feats = pd.merge(train_feats,train_title_feats,on=['idx'],how='left')
            
            train_title_feats = pd.read_pickle(args.featpath+"/{}_train_title_countvec_sim_feature.pkl".format(mode))
            train_feats = pd.merge(train_feats,train_title_feats,on=['idx'],how='left')
                        
            train_title_feats = pd.read_pickle(args.featpath+"/{}_train_title_set_feature.pkl".format(mode))
            train_feats = pd.merge(train_feats,train_title_feats,on=['idx'],how='left')
            
            # 构造与论文abstract相关的特征
            train_abstract_feats = pd.read_pickle(args.featpath+"/{}_train_abstract_word2vec_sim_feature.pkl".format(mode))
            train_feats = pd.merge(train_feats,train_abstract_feats,on=['idx'],how='left')
            
            train_abstract_feats = pd.read_pickle(args.featpath+"/{}_train_abstract_countvec_sim_feature.pkl".format(mode))
            train_feats = pd.merge(train_feats,train_abstract_feats,on=['idx'],how='left')
            
            
            # 构造与论文keywords相关的特征
            train_keywords_feats = pd.read_pickle(args.featpath+"/{}_train_keywords_word2vec_sim_feature.pkl".format(mode))
            train_feats = pd.merge(train_feats,train_keywords_feats,on=['idx'],how='left')
            
            train_keywords_feats = pd.read_pickle(args.featpath+"/{}_train_keywords_countvec_sim_feature.pkl".format(mode))
            train_feats = pd.merge(train_feats,train_keywords_feats,on=['idx'],how='left')
                        
            train_keywords_feats = pd.read_pickle(args.featpath+"/{}_train_keywords_set_feature.pkl".format(mode))
            train_feats = pd.merge(train_feats,train_keywords_feats,on=['idx'],how='left')
            
            # 构造与论文期刊相关的特征
            train_venue_feats = pd.read_pickle(args.featpath+"/{}_train_venue_word2vec_sim_feature.pkl".format(mode))
            train_feats = pd.merge(train_feats,train_venue_feats,on=['idx'],how='left')
            
            train_venue_feats = pd.read_pickle(args.featpath+"/{}_train_venue_countvec_sim_feature.pkl".format(mode))
            train_feats = pd.merge(train_feats,train_venue_feats,on=['idx'],how='left')
                        
            train_venue_feats = pd.read_pickle(args.featpath+"/{}_train_venue_set_feature.pkl".format(mode))
            train_feats = pd.merge(train_feats,train_venue_feats,on=['idx'],how='left')
            
            # 构造title_abstract的相似度特征
            train_title_feats = pd.read_pickle(args.featpath+"/{}_train_title_abstract_word2vec_sim_feature.pkl".format(mode))
            train_feats = pd.merge(train_feats,train_title_feats,on=['idx'],how='left')
            
            train_title_feats = pd.read_pickle(args.featpath+"/{}_train_title_abstract_countvec_sim_feature.pkl".format(mode))
            train_feats = pd.merge(train_feats,train_title_feats,on=['idx'],how='left')
                        
            # 时间加权相似度特征
            train_sim_feats = pd.read_pickle(args.featpath + "/{}_train_weight_word2vec_sim_feature.pkl".format(mode))
            train_feats = pd.merge(train_feats,train_sim_feats,on=['idx'],how='left')
            
            train_sim_feats = pd.read_pickle(args.featpath + "/{}_train_weight_countvec_sim_feature.pkl".format(mode))
            train_feats = pd.merge(train_feats,train_sim_feats,on=['idx'],how='left')
            
            
            # 统计keywords在abstract中出现的次数
            train_keywords_feats = pd.read_pickle(args.featpath+"/{}_train_word_keywords_appear_abstract_feature.pkl".format(mode))
            train_feats = pd.merge(train_feats,train_keywords_feats,on=['idx'],how='left')
            
            train_feats['mode'] = mode
            all_train_feats = pd.concat([all_train_feats,train_feats],axis=0).reset_index(drop=True)
            
        all_train_feats0 = all_train_feats[all_train_feats['label']==0].reset_index(drop=True) 
        all_train_feats1 = all_train_feats[all_train_feats['label']==1].reset_index(drop=True)
        all_train_feats1 = all_train_feats1.sample(frac=0.9,random_state=2021).reset_index(drop=True)
        
        all_train_feats  = pd.concat([all_train_feats0,all_train_feats1],axis=0).reset_index(drop=True)
        all_train_feats  = all_train_feats.sample(frac=1,random_state=2021).reset_index(drop=True)
            
        # 构造测试集特征
        all_test_feats = pd.DataFrame()
        for mode in ['testa','testb']:
            print("Merge na_v3 {} data".format(mode))
            test_df = pd.read_pickle(args.featpath+"/na_v3_{}.pkl".format(mode))
            
            # 构造与合作者相关的特征
            test_coauthors_feats = pd.read_pickle(args.featpath+"/na_v3_{}_accurate_coauthors_feature.pkl".format(mode))
            test_feats = pd.merge(test_df,test_coauthors_feats,on=['idx'],how='left')
            
            # 构造与论文机构相关的特征
            test_org_feats = pd.read_pickle(args.featpath+"/na_v3_{}_org_feature.pkl".format(mode))
            test_feats = pd.merge(test_feats,test_org_feats,on=['idx'],how='left')
            
            # 构造与论文title相关的特征
            test_title_feats = pd.read_pickle(args.featpath+"/na_v3_{}_title_word2vec_sim_feature.pkl".format(mode))
            test_feats = pd.merge(test_feats,test_title_feats,on=['idx'],how='left')
            
            test_title_feats = pd.read_pickle(args.featpath+"/na_v3_{}_title_countvec_sim_feature.pkl".format(mode))
            test_feats = pd.merge(test_feats,test_title_feats,on=['idx'],how='left')
                        
            test_title_feats = pd.read_pickle(args.featpath+"/na_v3_{}_title_set_feature.pkl".format(mode))
            test_feats = pd.merge(test_feats,test_title_feats,on=['idx'],how='left')
            
            # 构造与论文abstract相关的特征
            test_abstract_feats = pd.read_pickle(args.featpath+"/na_v3_{}_abstract_word2vec_sim_feature.pkl".format(mode))
            test_feats = pd.merge(test_feats,test_abstract_feats,on=['idx'],how='left')
            
            test_abstract_feats = pd.read_pickle(args.featpath+"/na_v3_{}_abstract_countvec_sim_feature.pkl".format(mode))
            test_feats = pd.merge(test_feats,test_abstract_feats,on=['idx'],how='left')
                        
            
            # 构造与论文keywords相关的特征
            test_keywords_feats = pd.read_pickle(args.featpath+"/na_v3_{}_keywords_word2vec_sim_feature.pkl".format(mode))
            test_feats = pd.merge(test_feats,test_keywords_feats,on=['idx'],how='left')
            
            test_keywords_feats = pd.read_pickle(args.featpath+"/na_v3_{}_keywords_countvec_sim_feature.pkl".format(mode))
            test_feats = pd.merge(test_feats,test_keywords_feats,on=['idx'],how='left')
                        
            test_keywords_feats = pd.read_pickle(args.featpath+"/na_v3_{}_keywords_set_feature.pkl".format(mode))
            test_feats = pd.merge(test_feats,test_keywords_feats,on=['idx'],how='left')
            
            # 构造与论文期刊相关的特征
            test_venue_feats = pd.read_pickle(args.featpath+"/na_v3_{}_venue_word2vec_sim_feature.pkl".format(mode))
            test_feats = pd.merge(test_feats,test_venue_feats,on=['idx'],how='left')
            
            test_venue_feats = pd.read_pickle(args.featpath+"/na_v3_{}_venue_countvec_sim_feature.pkl".format(mode))
            test_feats = pd.merge(test_feats,test_venue_feats,on=['idx'],how='left')
                        
            test_venue_feats = pd.read_pickle(args.featpath+"/na_v3_{}_venue_set_feature.pkl".format(mode))
            test_feats = pd.merge(test_feats,test_venue_feats,on=['idx'],how='left')
            
            # 构造title_abstract的相似度特征
            test_title_feats = pd.read_pickle(args.featpath+"/na_v3_{}_title_abstract_word2vec_sim_feature.pkl".format(mode))
            test_feats = pd.merge(test_feats,test_title_feats,on=['idx'],how='left')
            
            test_title_feats = pd.read_pickle(args.featpath+"/na_v3_{}_title_abstract_countvec_sim_feature.pkl".format(mode))
            test_feats = pd.merge(test_feats,test_title_feats,on=['idx'],how='left')
            
            
            # 时间加权相似度特征
            test_sim_feats = pd.read_pickle(args.featpath + "/na_v3_{}_weight_word2vec_sim_feature.pkl".format(mode))
            test_feats = pd.merge(test_feats,test_sim_feats,on=['idx'],how='left')
            
            test_sim_feats = pd.read_pickle(args.featpath + "/na_v3_{}_weight_countvec_sim_feature.pkl".format(mode))
            test_feats = pd.merge(test_feats,test_sim_feats,on=['idx'],how='left')
                        
            
            # 统计keywords在abstract中出现的次数
            test_keywords_feats = pd.read_pickle(args.featpath+"/na_v3_{}_word_keywords_appear_abstract_feature.pkl".format(mode))
            test_feats = pd.merge(test_feats,test_keywords_feats,on=['idx'],how='left')
            
            test_feats['mode'] = mode
            all_test_feats = pd.concat([all_test_feats,test_feats],axis=0).reset_index(drop=True)
        
        print([f for f in all_train_feats.columns if f not in all_test_feats.columns])
        print([f for f in all_test_feats.columns if f not in all_train_feats.columns])
        # 合并特征
        all_data_feats = pd.concat([all_train_feats,all_test_feats],axis=0).reset_index(drop=True)
        all_data_feats.drop(['author_idx','paper_abstract','paper_author','paper_author_list', 
                             'paper_authors','paper_id_list','paper_keywords','paper_org',
                             'paper_org_list','paper_title','paper_venue','paper_year',
                             'original_paper_author'],axis=1,inplace=True)
        
        all_data_feats = reduce_mem_usage(all_data_feats)
        all_data_feats.to_pickle(featpath)
        
        del all_data_feats, all_train_feats, all_test_feats
        gc.collect()
        
    # In[]
    # 训练模型 
    args.model = "XGBoost"
    with timer("Run {}".format(args.model)):
        target               = "label"
        args.savename        = "xgb"
        args.seed            = 2019
        args.learning_rate   = 0.02
        args.num_boost_round = 10000
        args.cat_features    = []
        args.FEATS_EXCLUDED  = ['paper_id','author_id','label','isTrain','idx','mode']
        args.subpath = os.path.join(args.subpath,"tree/{}".format(args.model))
        os.makedirs(args.subpath ,exist_ok=True)
        
        if args.model == "LightGBM":
            kfold_lightgbm(args,target,featpath) 
        else:
            kfold_xgboost(args,target,featpath)
 

if __name__=="__main__":
    args = get_parser()
    run_model()
    
