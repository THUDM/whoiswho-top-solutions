# -*- coding: utf-8 -*-
import gc
import json
import numpy as np 
import pandas as pd 
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

# Display/plot feature importance
def display_importances(feature_importance_df_,args):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)
    cols = cols[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]    
    best_features = best_features.sort_values(by="importance", ascending=False)

    plt.figure(figsize=(8, 6))
    sns.barplot(best_features["importance"],best_features["feature"])
    
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig(args.outpath+'/lgbScore.png')
    

def gen_dict(df, label):
    df = df[['paper_id', 'author_id', label]]
    res = df.groupby(['paper_id'])[label].apply(np.argmax).reset_index()
    res.columns = ['paper_id', 'index']
    idx_name = df[['author_id']].reset_index()
    res = res.merge(idx_name, 'left', 'index')
    from collections import defaultdict
    res_dict = defaultdict(list)
    for pid, aid in res[['paper_id', 'author_id']].values:
        res_dict[aid].append(pid)
    return res_dict


def f1_score(pred_dict, true_dict):
    total_unassigned_paper = np.sum([len(l) for l in true_dict.values()])
    print('total_unassigned_paper: ', total_unassigned_paper)
    print('true author num: ', len(true_dict))
    author_weight = dict((k, len(v) / total_unassigned_paper) for k, v in true_dict.items())
    author_precision = {}
    author_recall = {}
    for author in author_weight.keys():
        # total pred, total belong, correct pred
        total_belong = len(true_dict[author])
        total_pred = (len(pred_dict[author]) if author in pred_dict else 0)
        correct_pred = len(set(true_dict[author]) & (set(pred_dict[author]) if author in pred_dict else set()))
        author_precision[author] = (correct_pred/total_pred) if total_pred > 0 else 0
        author_recall[author] = correct_pred / total_belong
        
    weighted_precision = 0
    weighted_recall = 0
    for author, weight in author_weight.items():
        weighted_precision += weight * author_precision[author]
        weighted_recall += weight * author_recall[author]
    weighted_f1 = 2 * weighted_precision * weighted_recall / (weighted_precision + weighted_recall)
    
    return weighted_precision, weighted_recall, weighted_f1
    

# LightGBM GBDT with KFold or Stratified KFold
def kfold_lightgbm(args,target,featpath):
    data_df  = pd.read_pickle(featpath)
    data_df[target] = data_df[target].astype(int)
    
    test_df  = data_df[data_df['isTrain']==0].reset_index(drop=True)
    testa_df = test_df[test_df['mode']=="testa"].reset_index(drop=True)
    testb_df = test_df[test_df['mode']=="testb"].reset_index(drop=True)
    data_df  = data_df[data_df['isTrain']==1].reset_index(drop=True)
    data_df  = data_df.sample(frac=1,random_state=args.seed).reset_index(drop=True)
    print("Data's shape: {},Testa's shape: {},Testb's shape: {}".format(data_df.shape,testa_df.shape,testb_df.shape))
    
    # Create arrays and dataframes to store results
    oof_dfs = pd.DataFrame()
    testa_sub_preds = np.zeros(testa_df.shape[0])
    testb_sub_preds = np.zeros(testb_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in data_df.columns if f not in args.FEATS_EXCLUDED]
   
    params = {'objective': 'binary','boosting_type' : 'gbdt','learning_rate': args.learning_rate,
              'max_depth' :args.max_depth,'num_leaves' : args.num_leaves,'min_child_weight' : 10,'min_data_in_leaf' : 40,
              'feature_fraction' : 0.80,'subsample' : 0.85,'seed' : 114,'bagging_freq' : 1,'metric': {'auc'}
              }
    
    print(data_df['paper_id'].unique())
    f1_list = []
    all_paper_id = data_df['paper_id'].unique()
    for i in range(5):
        print("------------------------ fold_{} ------------------------".format(i+1))
        val_paper_id = all_paper_id[int(len(all_paper_id)*0.2*i):int(len(all_paper_id)*0.2*(i+1))]
        trn_paper_id = [paperid for paperid in all_paper_id if paperid not in val_paper_id]
        
        train_df = data_df[data_df['paper_id'].isin(trn_paper_id)].reset_index(drop=True)
        valid_df = data_df[data_df['paper_id'].isin(val_paper_id)].reset_index(drop=True)
            
        train_x, train_y = train_df[feats], train_df[target]
        valid_x, valid_y = valid_df[feats], valid_df[target]

        # set data structure
        lgb_train = lgb.Dataset(train_x,label=train_y,free_raw_data=False)
        lgb_test  = lgb.Dataset(valid_x,label=valid_y,free_raw_data=False)

        model = lgb.train(params,lgb_train,
                          valid_sets=[lgb_train, lgb_test],
                          valid_names=['train', 'valid'],
                          num_boost_round=args.num_boost_round,
                          early_stopping_rounds=200,
                          verbose_eval=100)
        # 验证集预测结果及评分
        val_pred = model.predict(valid_x, num_iteration=model.best_iteration)
        
        oof_df = valid_df[['idx','paper_id','author_id','label']]
        val_true_dict = gen_dict(oof_df, 'label')
        
        oof_df['pred'] = val_pred
        oof_df = oof_df[oof_df['pred']>=0.2].reset_index(drop=True)
        val_pred_dict = gen_dict(oof_df, 'pred')
        
        precision, recall, f1 = f1_score(val_pred_dict, val_true_dict)
        print('weighted_precision: %0.4f, weighted_recall: %0.4f, weighted_f1: %0.4f' %(precision, recall, f1))
        f1_list.append(f1)
        oof_dfs = pd.concat([oof_dfs,oof_df],axis=0).reset_index(drop=True)
        
        # 测试集预测结果及评分
        testa_pred = model.predict(testa_df[feats], num_iteration=model.best_iteration)
        testa_sub_preds += testa_pred / 5
        
        testa_sub_df_tmp  = testa_df[['idx','paper_id','author_id']]
        testa_sub_df_tmp['pred'] = testa_pred
        testa_sub_df_tmp['author_id'] = testa_sub_df_tmp['author_id'].fillna(0).astype(str)
        
        testb_pred = model.predict(testb_df[feats], num_iteration=model.best_iteration)
        testb_sub_preds += testb_pred / 5
        
        testb_sub_df_tmp  = testb_df[['idx','paper_id','author_id']]
        testb_sub_df_tmp['pred'] = testb_pred
        testb_sub_df_tmp['author_id'] = testb_sub_df_tmp['author_id'].fillna(0).astype(str)
        
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = np.log1p(model.feature_importance(importance_type='gain', iteration=model.best_iteration))
        fold_importance_df["fold"] = i + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        del model, train_x, train_y, valid_x, valid_y
        gc.collect()
     
    display_importances(feature_importance_df,args)
    
    # 验证集得分
    print('weighted_f1: %0.4f' %(np.mean(f1_list)))
    
    # 生成A榜提交结果
    testa_sub_df         = testa_df[['idx','paper_id','author_id']]
    testa_sub_df['pred'] = testa_sub_preds
    testa_sub_df['author_id'] = testa_sub_df['author_id'].fillna(0).astype(str)
    testa_sub_df.to_csv(args.subpath+"/%s_testa.csv"%(args.savename),index=False)
    testa_sub_df.drop(['idx'],axis=1,inplace=True)
    testa_sub_df = testa_sub_df[testa_sub_df['author_id']!='0'].reset_index(drop=True)
    testa_sub_df = testa_sub_df[testa_sub_df['pred']>=0.2].reset_index(drop=True)
    testa_sub_df = testa_sub_df.sort_values(by=['paper_id','pred'],ascending=False).reset_index(drop=True)
    
    testa_result_dict = gen_dict(testa_sub_df, 'pred')    
    with open(args.subpath+"/%s_testa_%0.4f.json"%(args.savename,np.mean(f1_list)), 'w') as file:
        file.write(json.dumps(testa_result_dict))
        
    del testa_sub_df, testa_result_dict
    gc.collect()
        
    # 生成B榜提交结果
    testb_sub_df         = testb_df[['idx','paper_id','author_id']]
    testb_sub_df['pred'] = testb_sub_preds
    testb_sub_df['author_id'] = testb_sub_df['author_id'].fillna(0).astype(str)
    testb_sub_df.to_csv(args.subpath+"/%s_testb.csv"%(args.savename),index=False)
    testb_sub_df.drop(['idx'],axis=1,inplace=True)
    testb_sub_df = testb_sub_df[testb_sub_df['author_id']!='0'].reset_index(drop=True)
    testb_sub_df = testb_sub_df[testb_sub_df['pred']>=0.2].reset_index(drop=True)
    testb_sub_df = testb_sub_df.sort_values(by=['paper_id','pred'],ascending=False).reset_index(drop=True)
    
    testb_result_dict = gen_dict(testb_sub_df, 'pred')    
    with open(args.subpath+"/%s_testb_%0.4f.json"%(args.savename,np.mean(f1_list)), 'w') as file:
        file.write(json.dumps(testb_result_dict))
        
    del testb_sub_df, testb_result_dict
    gc.collect()
        