# -*- coding: utf-8 -*-
import os
import json
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from args import get_parser
from utils import reduce_mem_usage


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

def extract_year_feature(args,data,mode='na_v1',sign="train"):
    savepath = args.featpath + "/{}_{}_year_feature.pkl".format(mode,sign)
    if not os.path.exists(savepath):
        print("Making Feature")
        
        if sign == "train":
            with open(args.datapath + "/prepare/{}_data_pub.json".format(mode), "rb") as f:
                pubs = pickle.load(f)
        else:       
            with open(args.datapath+"/{}/cna-valid/whole_author_profiles_pub.json".format(mode), "r",encoding='utf-8') as f:
                pubs = json.load(f)
            
        paperid_year_dict = {}
        for paper_id in tqdm(pubs):
            year = pubs[paper_id]['year']
            paperid_year_dict[paper_id] = year
            
        data['author_id'] = data['author_id'].fillna(0).astype(str)
        data = data[data['author_id']!='0'].reset_index(drop=True)
        
        all_feat_df = pd.DataFrame()
        for i in tqdm(range(100)):
            df = data[int(len(data)*(i/100)):int(len(data)*((i+1)/100))].reset_index(drop=True)
            
            all_year_gap_list = []
            for paper_year,paper_id_list in zip(df['paper_year'].values,df['paper_id_list'].values):
                year_gap_list = []
                for paper_id in paper_id_list:
                    paper_year2 = paperid_year_dict[paper_id]
                    try:
                        gap = abs(int(paper_year) - int(paper_year2))
                        if gap > 20:
                            gap = 20
                        year_gap_list.append(gap)
                    except:
                        pass
                all_year_gap_list.append(year_gap_list)
                
            df['year_gap_list'] = all_year_gap_list
            drop_columns = [f for f in df.columns if f not in ['idx']]  
            for col in ['year_gap_list']:
                print(df[col])
                df[col+"_mean"] = df[col].apply(lambda x:np.mean(x) if x!=[] else np.nan)
                df[col+"_max"]  = df[col].apply(lambda x:np.max(x) if x!=[] else np.nan)
                df[col+"_min"]  = df[col].apply(lambda x:np.min(x) if x!=[] else np.nan)
                df[col+"_max_min"] = df[col+"_max"] - df[col+"_min"]
                df[col+"_median"]  = df[col].apply(lambda x:np.median(x) if x!=[] else np.nan)
                
                df[col+"_0_count"] = df[col].apply(lambda x:x.count(0))
                df[col+"_1_count"] = df[col].apply(lambda x:x.count(1))
                df[col+"_2_count"] = df[col].apply(lambda x:x.count(2))
                df[col+"_3_count"] = df[col].apply(lambda x:x.count(3))
                
            df.drop(drop_columns,axis=1,inplace=True)
            df = reduce_mem_usage(df)
            all_feat_df = pd.concat([all_feat_df,df],axis=0).reset_index(drop=True)
            
        all_feat_df = reduce_mem_usage(all_feat_df)
        all_feat_df.to_pickle(savepath)
        
    all_feat_df = pd.read_pickle(savepath)
    print(all_feat_df.columns)
    return all_feat_df


if __name__=="__main__":
    args = get_parser()
    # 构建训练集特征
    for mode in ['na_v3']:
        train_df = pd.read_pickle(args.featpath + "/{}_train.pkl".format(mode))
        # 精准匹配
        extract_year_feature(args,train_df,mode=mode,sign="train")
    
    # 构建测试集特征
    for mode in ['na_v3']:
        test_df = pd.read_pickle(args.featpath + "/{}_testa.pkl".format(mode))
        extract_year_feature(args,test_df,mode=mode,sign="testa")
        
        test_df = pd.read_pickle(args.featpath + "/{}_testb.pkl".format(mode))
        extract_year_feature(args,test_df,mode=mode,sign="testb")
