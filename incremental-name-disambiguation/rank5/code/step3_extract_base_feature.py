# -*- coding: utf-8 -*-
import os
import re
import json
import pickle
import pandas as pd
import numpy as np
from args import get_parser
from utils import reduce_mem_usage,convert_name,distance_score

def fix_name(s):
    s = s.lower().strip()
    x = re.split(r'[ \.\-\_]', s)
    set_x = set()
    for a in x:
        if len(a) > 0:
            set_x.add(a)
    x = list(set_x)
    x.sort()
    s = ''.join(x)
    return s


def read_test_base_feature(args,data_type,mode="na_v3"):
    savepath = args.featpath + "/{}_{}.pkl".format(mode,data_type)
    if not os.path.exists(savepath):
        if data_type == "testa":
            # 读取测试集 cna_valid_unass_pub.json
            with open(args.datapath+"/{}/cna-valid/cna_valid_unass_pub.json".format(mode), "r",encoding='utf-8') as f:
                test_paper = json.load(f) 
        else:
            # 读取测试集 cna_test_unass_pub.json
            with open(args.datapath+"/{}/cna-test/cna_test_unass_pub.json".format(mode), "r",encoding='utf-8') as f:
                test_paper = json.load(f) 
                
        with open(args.datapath+"/{}/cna-valid/whole_author_profiles.json".format(mode), "r") as f:
            test_author = json.load(f)
        
        test_author_df = []    
        for author_id in test_author:
            name = test_author[author_id]['name']
            pubs = test_author[author_id]['pubs']
            test_author_df.append([author_id,name,pubs])
        test_author_df = pd.DataFrame(test_author_df,columns=['author_id','original_paper_author','paper_id_list'])
            
        # 读取测试集 cna_valid_unass.json
        if data_type == "testa":
            testpath = args.datapath+"/{}/cna-valid/cna_valid_unass.json".format(mode)
        else:
            testpath = args.datapath+"/{}/cna-test/cna_test_unass.json".format(mode)
        
        test_df = []
        with open(testpath, "r") as f:
            paper_id_list = json.load(f)
            for paper_id in paper_id_list:
                author_idx = int(paper_id.split("-")[-1])
                paper_id   = paper_id.split("-")[0]
                test_df.append([paper_id,author_idx])
        test_df = pd.DataFrame(test_df,columns=['paper_id','author_idx'])
        
        for col in ['title','abstract','keywords','authors','venue','year']:
            test_df['paper_{}'.format(col)] = test_df['paper_id'].apply(lambda x:test_paper[x][col])
    
        test_df['paper_org_list'] = test_df['paper_authors'].apply(lambda xs:[x['org'] for x in xs])   
        test_df['paper_author_list'] = test_df['paper_authors'].apply(lambda xs:[x['name'] for x in xs])    
        test_df['original_paper_author'] = test_df[['paper_author_list','author_idx']].apply(lambda x:x[0][x[1]],axis=1)
        test_df['original_paper_author'] = test_df['original_paper_author'].apply(lambda x:convert_name(x))
        test_df['paper_org'] = test_df[['paper_org_list','author_idx']].apply(lambda x:x[0][x[1]],axis=1)
        
        
        test_df['paper_author'] = test_df['original_paper_author'].apply(fix_name)
        test_author_df['paper_author'] = test_author_df['original_paper_author'].apply(fix_name)
        test_df = pd.merge(test_df,test_author_df[['author_id','paper_author','paper_id_list']],on=['paper_author'],how='left')
        test_df['author_id'] = test_df['author_id'].fillna(0).astype(str)
        
        # In[]
        test_df1 = test_df[test_df['author_id']!='0'].reset_index(drop=True)
        test_df0 = test_df[test_df['author_id']=='0'].reset_index(drop=True)
        test_df0.drop(['author_id','paper_id_list'],axis=1,inplace=True)

        all_tmp = pd.DataFrame()
        for original_paper_author in test_df0['original_paper_author'].unique():
            tmp = []
            for original_paper_author2 in test_author_df['original_paper_author'].unique():
                score = distance_score(original_paper_author,original_paper_author2)
                tmp.append([original_paper_author,original_paper_author2,score])
            tmp = pd.DataFrame(tmp,columns=['original_paper_author','original_paper_author2','score'])
            tmp = tmp.sort_values(by=['score']).reset_index(drop=True)[:20]
            
            if np.min(tmp['score'].values)<2:
                tmp = tmp[['original_paper_author','original_paper_author2']][:1]
            else:
                tmp = tmp[['original_paper_author','original_paper_author2']][:2]
            all_tmp = pd.concat([all_tmp,tmp],axis=0).reset_index(drop=True)
            
        test_df0 = pd.merge(test_df0,all_tmp,on=['original_paper_author'],how='left')
        test_df0['original_paper_author'] = test_df0['original_paper_author2']
        test_df0.drop(['original_paper_author2'],axis=1,inplace=True)
        test_df0['paper_author'] = test_df0['original_paper_author'].apply(fix_name)
        test_df0 = pd.merge(test_df0,test_author_df[['author_id','paper_author','paper_id_list']],on=['paper_author'],how='left')
        
        test_df  = pd.concat([test_df0,test_df1],axis=0).reset_index(drop=True)  
        test_df['author_id'] = test_df['author_id'].fillna(0).astype(str)
        # In[]    
        test_df['isTrain'] = 0
        test_df['label']   = -1
        test_df['idx'] = test_df.index
        test_df = reduce_mem_usage(test_df)
        test_df.to_pickle(savepath)
        
    test_df = pd.read_pickle(savepath)
    
    return test_df

def extract_train_base_feature(args,mode):
    savepath = args.featpath + "/{}_train.pkl".format(mode)
    if not os.path.exists(savepath):
        # 读取训练集 data_pub.json
        with open(args.datapath + "/prepare/{}_data_pub.json".format(mode), "rb") as f:
            train_paper = pickle.load(f)
            
        with open(args.datapath + "/prepare/{}_data_author.json".format(mode), "rb") as f:
            train_author = pickle.load(f)
        
        train_author_df = [] 
        for name in train_author:
            for author_id in train_author[name]:
                pubs = train_author[name][author_id]
                train_author_df.append([author_id,name,pubs])
        train_author_df = pd.DataFrame(train_author_df,columns=['author_id','paper_author','paper_id_list'])
            
        # 读取训练集 train_unass_data.json
        with open(args.outpath+"/{}_train_unass_data.json".format(mode), "r",encoding='utf-8') as f:
            train_df = json.load(f)  
            
        train_df = pd.DataFrame(train_df,columns=['paper_id','label']) 
        train_df['author_idx'] = train_df['paper_id'].apply(lambda x:x.split("-")[-1]).astype(int)
        train_df['paper_id'] = train_df['paper_id'].apply(lambda x:x.split("-")[0])
        
        for col in ['title','abstract','keywords','authors','venue','year']:
            train_df['paper_{}'.format(col)] = train_df['paper_id'].apply(lambda x:train_paper[x][col])
    
        train_df['paper_org_list'] = train_df['paper_authors'].apply(lambda xs:[x['org'] for x in xs])   
        train_df['paper_author_list'] = train_df['paper_authors'].apply(lambda xs:[x['name'] for x in xs])    
        train_df['paper_author'] = train_df[['paper_author_list','author_idx']].apply(lambda x:x[0][x[1]],axis=1)
        train_df['paper_org'] = train_df[['paper_org_list','author_idx']].apply(lambda x:x[0][x[1]],axis=1)
       
        train_df['paper_author'] = train_df['paper_author'].apply(fix_name)
        train_author_df['paper_author'] = train_author_df['paper_author'].apply(fix_name)
        print(train_df)
        train_df = pd.merge(train_df,train_author_df,on=['paper_author'],how='left')
        print(train_df[['label','paper_id','author_id']])
        train_df['label'] = train_df[['label','author_id']].apply(lambda x:1 if x[0]==x[1] else 0,axis=1)
        train_df['idx'] = train_df.index
        
        train_df0 = train_df[train_df['label']==0].reset_index(drop=True)
        train_df1 = train_df[train_df['label']==1].reset_index(drop=True)
        train_df1['paper_id_list'] = train_df1[['paper_id','paper_id_list']].apply(lambda x:[k for k in x[1] if k != x[0]],axis=1)
        
        # 合并
        train_df = pd.concat([train_df0,train_df1],axis=0).reset_index(drop=True)
        train_df = train_df.sort_values(by=['idx']).reset_index(drop=True)
        print(train_df['idx'])
        
        train_df['isTrain'] = 1
        train_df = reduce_mem_usage(train_df)
        train_df.to_pickle(savepath)
        
    train_df = pd.read_pickle(savepath)
    
    
    return train_df


if __name__=="__main__":
    args = get_parser()
    # 构造训练集
    for mode in ['na_v3']: 
        extract_train_base_feature(args,mode)    
    
    # 构建测试集
    for data_type in ['testa','testb']:
        read_test_base_feature(args,data_type,mode="na_v3")
    

