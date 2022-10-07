# -*- coding: utf-8 -*-
import os
import re
import json
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
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

def get_coauthors_count_dict(items):    
    all_paper_author_df = []
    for item in items:
        all_paper_author_df.extend(item)
    all_paper_author_df = pd.DataFrame(all_paper_author_df,columns=['name'])
    all_paper_author_df = all_paper_author_df.groupby(['name'],as_index=False)['name'].agg({"count":"count"})
    
    all_paper_author_num   = len(all_paper_author_df)
    all_paper_author_count = np.sum(all_paper_author_df['count'].values)
    
    dit = {}
    for name,count in zip(all_paper_author_df['name'].values,all_paper_author_df['count'].values):
        dit[name] = count
    
    return [dit,all_paper_author_num,all_paper_author_count]

def get_count_list(x):
    paper_author_list     = x[0]
    all_paper_author_dict = x[1]
    
    count_list = []
    for x in paper_author_list:
        if x in all_paper_author_dict:
            count_list.append(all_paper_author_dict[x])
        else:
            count_list.append(0)
            
    return count_list

def get_max_min(x,mode='max'):
    if mode == "max":
        try:
            return np.max(x)
        except:
            return 0
    if mode == "min":
        try:
            return np.min(x)
        except:
            return 0

def extract_accurate_coauthors_feature(args,data,mode='na_v1',sign="train"):
    savepath = args.featpath + "/{}_{}_accurate_coauthors_feature.pkl".format(mode,sign)
    if not os.path.exists(savepath):
        print("Making Feature")
        
        if sign == "train":
            with open(args.datapath + "/prepare/{}_data_pub.json".format(mode), "rb") as f:
                pubs = pickle.load(f)
        else:       
            with open(args.datapath+"/{}/cna-valid/whole_author_profiles_pub.json".format(mode), "r",encoding='utf-8') as f:
                pubs = json.load(f)
            
        paperid_author_dict = {}
        for paper_id in tqdm(pubs):
            authors_orgs = pubs[paper_id]['authors']
            authors = [fix_name(convert_name(dit['name'])) for dit in authors_orgs]
            paperid_author_dict[paper_id] = authors
            
        data['author_id'] = data['author_id'].fillna(0).astype(str)
        data = data[data['author_id']!='0'].reset_index(drop=True)
        
        all_feat_df = pd.DataFrame()
        for i in tqdm(range(100)):
            df = data[int(len(data)*(i/100)):int(len(data)*((i+1)/100))].reset_index(drop=True)
            
            # 构造特征
            # 所有共同作者的数量
            df['paper_author_list'] = df['paper_author_list'].apply(lambda x:[fix_name(convert_name(k)) for k in x])
            df['paper_author_list'] = df[['paper_author','paper_author_list']].apply(lambda x:[k for k in x[1] if k != x[0]],axis=1)
            
            if sign in ["train"]:
                df_1 = df[df['label']==1].reset_index(drop=True)
                df_1['all_paper_author_list']  = df_1['paper_id_list'].apply(lambda xs:[paperid_author_dict[x] for x in xs])
                df_1['all_paper_author_dict']  = df_1['all_paper_author_list'].apply(lambda x:get_coauthors_count_dict(x)[0])
                df_1['all_paper_author_num']   = df_1['all_paper_author_list'].apply(lambda x:get_coauthors_count_dict(x)[1])
                df_1['all_paper_author_count'] = df_1['all_paper_author_list'].apply(lambda x:get_coauthors_count_dict(x)[2])
                df_1.drop(['all_paper_author_list'],axis=1,inplace=True)
                df_1['paper_author_count_list'] = df_1[['paper_author_list','all_paper_author_dict']].apply(get_count_list,axis=1)
                
                df_2 = df[df['label']==0].reset_index(drop=True)
                tmp = df_2.drop_duplicates(subset=['author_id'])[['author_id','paper_id_list']].reset_index(drop=True)
                tmp['all_paper_author_list']  = tmp['paper_id_list'].apply(lambda xs:[paperid_author_dict[x] for x in xs])
                tmp['all_paper_author_dict']  = tmp['all_paper_author_list'].apply(lambda x:get_coauthors_count_dict(x)[0])
                tmp['all_paper_author_num']   = tmp['all_paper_author_list'].apply(lambda x:get_coauthors_count_dict(x)[1])
                tmp['all_paper_author_count'] = tmp['all_paper_author_list'].apply(lambda x:get_coauthors_count_dict(x)[2])
                tmp.drop(['paper_id_list','all_paper_author_list'],axis=1,inplace=True)
                df_2 = pd.merge(df_2,tmp,on=['author_id'],how='left')
                df_2['paper_author_count_list'] = df_2[['paper_author_list','all_paper_author_dict']].apply(get_count_list,axis=1)
                
                df = pd.concat([df_1,df_2],axis=0).reset_index(drop=True)
                
            else:
                tmp = df.drop_duplicates(subset=['author_id'])[['author_id','paper_id_list']].reset_index(drop=True)
                tmp['all_paper_author_list']  = tmp['paper_id_list'].apply(lambda xs:[paperid_author_dict[x] for x in xs])
                tmp['all_paper_author_dict']  = tmp['all_paper_author_list'].apply(lambda x:get_coauthors_count_dict(x)[0])
                tmp['all_paper_author_num']   = tmp['all_paper_author_list'].apply(lambda x:get_coauthors_count_dict(x)[1])
                tmp['all_paper_author_count'] = tmp['all_paper_author_list'].apply(lambda x:get_coauthors_count_dict(x)[2])
                tmp.drop(['paper_id_list','all_paper_author_list'],axis=1,inplace=True)
                df = pd.merge(df,tmp,on=['author_id'],how='left')
                df['paper_author_count_list'] = df[['paper_author_list','all_paper_author_dict']].apply(get_count_list,axis=1)
            
            print(df['paper_author_count_list'])
            drop_columns = [f for f in df.columns if f not in ['idx']]   
            df['coauthors_count']       = df['paper_author_count_list'].apply(lambda x:np.sum(x))
            df['coauthors_count_ratio'] = df['coauthors_count'] / df['all_paper_author_count']
            df['coauthors_count_mean']  = df['paper_author_count_list'].apply(lambda x:np.mean(x))
            df['coauthors_count_max']   = df['paper_author_count_list'].apply(lambda x:get_max_min(x,mode='max'))
            df['coauthors_count_std']   = df['paper_author_count_list'].apply(lambda x:np.std(x))
            df['coauthors_num']         = df['paper_author_count_list'].apply(lambda x:len(x) - x.count(0))
            df['coauthors_num_ratio']   = df['coauthors_num'] / df['all_paper_author_num']    
            print(df[['coauthors_count_ratio','coauthors_num_ratio']])
            
            df.drop(drop_columns,axis=1,inplace=True)
            df = reduce_mem_usage(df)
            all_feat_df = pd.concat([all_feat_df,df],axis=0).reset_index(drop=True)
            
        all_feat_df = reduce_mem_usage(all_feat_df)
        all_feat_df.to_pickle(savepath)
        
    all_feat_df = pd.read_pickle(savepath)
    
    return all_feat_df


if __name__=="__main__":
    args = get_parser()
    # 构建训练集特征
    for mode in ['na_v3']:
        train_df = pd.read_pickle(args.featpath + "/{}_train.pkl".format(mode))
        # 精准匹配
        extract_accurate_coauthors_feature(args,train_df,mode=mode,sign="train")
    
    # 构建测试集特征
    for mode in ['na_v3']:
        test_df = pd.read_pickle(args.featpath + "/{}_testa.pkl".format(mode))
        extract_accurate_coauthors_feature(args,test_df,mode=mode,sign="testa")
        
        test_df = pd.read_pickle(args.featpath + "/{}_testb.pkl".format(mode))
        extract_accurate_coauthors_feature(args,test_df,mode=mode,sign="testb")
