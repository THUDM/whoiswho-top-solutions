# -*- coding: utf-8 -*-
import os
import re
import json
import pickle
import Levenshtein
import pandas as pd
import numpy as np
from tqdm import tqdm
from args import get_parser
from utils import reduce_mem_usage


#正则去标点
def etl(content):
    content = re.sub("[\s+\.\!\/,;$%^*(+\"\')]+|[+——_()?【】“”！，。？、~@#￥%……&*（）-]+", " ", content)
    content = re.sub(r" {2,}", " ", content)
    return content

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

def extract_year_feature(args,data,mode='na_v1',sign="train",text='venue'):
    savepath = args.featpath + "/{}_{}_{}_set_feature.pkl".format(mode,sign,text)
    if not os.path.exists(savepath):
        print("Making Feature")
        
        if sign == "train":
            with open(args.datapath + "/prepare/{}_data_pub.json".format(mode), "rb") as f:
                pubs = pickle.load(f)
        else:       
            with open(args.datapath+"/{}/cna-valid/whole_author_profiles_pub.json".format(mode), "r",encoding='utf-8') as f:
                pubs = json.load(f)
            
        paperid_text_dict = {}
        for paper_id in tqdm(pubs):
            seq = pubs[paper_id][text]
            if text != "keywords":
                new_seq = etl(seq).lower()
            else:
                new_seq = []
                for s in seq:
                    s = etl(s).lower()
                    if s!='':
                        new_seq.append(s)
            paperid_text_dict[paper_id] = new_seq
            
        data['author_id'] = data['author_id'].fillna(0).astype(str)
        data = data[data['author_id']!='0'].reset_index(drop=True)
        
        all_feat_df = pd.DataFrame()
        for i in tqdm(range(100)):
            df = data[int(len(data)*(i/100)):int(len(data)*((i+1)/100))].reset_index(drop=True)
            
            all_jw_list = []
            all_intersection_list = []
            for seq,paper_id_list in zip(df['paper_{}'.format(text)].values,df['paper_id_list'].values):
                jw_list = []
                intersection_list = []
                if text != "keywords":
                    seq = str(etl(seq).lower())
                    if seq != '':
                        for paper_id in paper_id_list:
                            seq2 = paperid_text_dict[paper_id]
                            jw = Levenshtein.jaro_winkler(seq,seq2)
                            jw_list.append(jw)
                        
                            seq_list  = seq.split()
                            seq2_list = seq2.split()
                            intersection_num = len(set(seq_list)&set(seq2_list))
                            intersection_list.append(intersection_num)
                            
                            
                else:
                    new_seq = []
                    for s in seq:
                        s = etl(s).lower()
                        if s!='':
                            new_seq.append(s)
                    for paper_id in paper_id_list:
                        new_seq2 = paperid_text_dict[paper_id]        
                        intersection_num = len(set(new_seq)&set(new_seq2))
                        intersection_list.append(intersection_num)
                        
                        jw = Levenshtein.jaro_winkler(' '.join(new_seq),' '.join(new_seq2))
                        jw_list.append(jw)
                            
                all_jw_list.append(jw_list)
                all_intersection_list.append(intersection_list) 
                
            df['{}_jaro_winkler'.format(text)] = all_jw_list
            df['{}_set_num'.format(text)] = all_intersection_list
            
            drop_columns = [f for f in df.columns if f not in ['idx']] 
            for col in ['{}_jaro_winkler'.format(text)]:
                print(df[col])
                df[col+"_mean"] = df[col].apply(lambda x:np.mean(x) if x!=[] else np.nan)
                df[col+"_max"]  = df[col].apply(lambda x:np.max(x) if x!=[] else np.nan)
                df[col+"_min"]  = df[col].apply(lambda x:np.min(x) if x!=[] else np.nan)
                df[col+"_std"]  = df[col].apply(lambda x:np.std(x) if x!=[] else np.nan)
                df[col+"_1_count"] = df[col].apply(lambda x:x.count(1) if x!=[] else np.nan)
                df[col+"_1_ratio"] = df[col].apply(lambda x:x.count(1)/len(x) if x!=[] else np.nan)
                print(df[col+"_1_ratio"])
                
            for col in ['{}_set_num'.format(text)]:
                df[col+"_sum"]   = df[col].apply(lambda x:np.sum(x) if x!=[] else np.nan)
                df[col+"_ratio"] = df[col].apply(lambda x:np.sum(x)/len(x) if x!=[] else np.nan)
                print(df[col+"_ratio"])
                
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
        for text in ['title','keywords','venue']:
            train_df = pd.read_pickle(args.featpath + "/{}_train.pkl".format(mode))
            # 精准匹配
            extract_year_feature(args,train_df,mode=mode,sign="train",text=text)
    
    # 构建测试集特征
    for mode in ['na_v3']:
        for text in ['title','keywords','venue']:
            test_df = pd.read_pickle(args.featpath + "/{}_testa.pkl".format(mode))
            extract_year_feature(args,test_df,mode=mode,sign="testa",text=text)
            
            test_df = pd.read_pickle(args.featpath + "/{}_testb.pkl".format(mode))
            extract_year_feature(args,test_df,mode=mode,sign="testb",text=text)
