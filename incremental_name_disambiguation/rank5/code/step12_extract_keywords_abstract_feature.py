# -*- coding: utf-8 -*-
import os
import re
import json
import pickle
import pandas as pd
from tqdm import tqdm
from args import get_parser
from utils import reduce_mem_usage

    
#正则去标点
def etl(content):
    content = re.sub("[\s+\.\!\/,;$%^*(+\"\')]+|[+——_()?【】“”！，。？、~@#￥%……&*（）-]+", " ", content)
    content = re.sub(r" {2,}", " ", content)
    return content

def deal_keywords(x,wordtype):
    if wordtype == "word":
        new_kws = []
        for keyword in x:
            keyword = etl(keyword).lower()
            if keyword != '':
                new_kws.append(keyword)
        return new_kws
    else:
        new_kws = []
        for keyword in x:
            keyword = etl(keyword).lower()
            if keyword != '':
                keyword = keyword.split()
                new_kws.extend(keyword)
        return new_kws

# 统计关键词在摘要中出现的次数
def get_keywords_appear_abstract_count(x):
    paper_keywords = x[0]
    all_paper_abstract_list = x[1] 
    
    count = 0
    for paper_abstract in all_paper_abstract_list:
        for kw in paper_keywords:
            count = count + len(paper_abstract.split("{}".format(kw))) - 1
        
    return count

def extract_keywords_abstract_feature(args,data,mode="na_v1",sign="train",wordtype="word"):
    savepath = args.featpath + "/{}_{}_{}_keywords_appear_abstract_feature.pkl".format(mode,sign,wordtype)
    if not os.path.exists(savepath):
        if sign == "train":
            with open(args.datapath + "/prepare/{}_data_pub.json".format(mode), "rb") as f:
                pubs = pickle.load(f)
        else:       
            with open(args.datapath+"/{}/cna-valid/whole_author_profiles_pub.json".format(mode), "r",encoding='utf-8') as f:
                pubs = json.load(f)
            
        paperid_abstract_dict = {}
        for paper_id in tqdm(pubs):
            abstract = etl(pubs[paper_id]['abstract']).lower()
            paperid_abstract_dict[paper_id] = abstract
            
        data['author_id'] = data['author_id'].fillna(0).astype(str)
        data = data[data['author_id']!='0'].reset_index(drop=True)
        
        all_feat_df = pd.DataFrame()
        for i in tqdm(range(100)):
            df = data[int(len(data)*(i/100)):int(len(data)*((i+1)/100))].reset_index(drop=True)
            df['paper_keywords'] = df['paper_keywords'].apply(lambda x:deal_keywords(x,wordtype))
            df['all_paper_abstract_list'] = df['paper_id_list'].apply(lambda xs:[paperid_abstract_dict[x] for x in xs])
            
            drop_columns = [f for f in df.columns if f not in ['idx']] 
            df['{}_keywords_appear_abstract_count'.format(wordtype)] = df[['paper_keywords','all_paper_abstract_list']].apply(get_keywords_appear_abstract_count,axis=1)
            df['{}_keywords_appear_abstract_ratio'.format(wordtype)] = df[['{}_keywords_appear_abstract_count'.format(wordtype),'paper_id_list']].apply(
                    lambda x:x[0]/len(x[1]),axis=1)
            print(df['{}_keywords_appear_abstract_ratio'.format(wordtype)])
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
        extract_keywords_abstract_feature(args,train_df,mode=mode,sign="train",wordtype="word")
    
    # 构建测试集特征
    for mode in ['na_v3']:
        test_df = pd.read_pickle(args.featpath + "/{}_testa.pkl".format(mode))
        extract_keywords_abstract_feature(args,test_df,mode=mode,sign="testa",wordtype="word")
        
        test_df = pd.read_pickle(args.featpath + "/{}_testb.pkl".format(mode))
        extract_keywords_abstract_feature(args,test_df,mode=mode,sign="testb",wordtype="word")
