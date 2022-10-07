# -*- coding: utf-8 -*-
import os
import gc
import re
import json
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.spatial import distance
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
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
        
def gen_countvec_features(df, value,n=10):
    df.columns = ['list']
    df['list'] = df['list'].apply(lambda x: ','.join(x))
    enc_vec = CountVectorizer()
    tfidf_vec = enc_vec.fit_transform(df['list'])
    svd_enc = TruncatedSVD(n_components=n, n_iter=20, random_state=2020)
    vec_svd = svd_enc.fit_transform(tfidf_vec)
    vec_svd = pd.DataFrame(vec_svd)
    vec_svd.columns = ['svd_countvec_{}_{}'.format(value, i) for i in range(n)]
    df = pd.concat([df, vec_svd], axis=1)
    del df['list']
    return df

def extract_text_countvec_sim_feature(args,data,mode="na_v1",sign="train",text='title'):
    if text in ["abstract","title_abstract"]:
        n_components = 128
    else:
        n_components = 64
    savepath = args.featpath + "/{}_{}_{}_countvec_sim_feature.pkl".format(mode,sign,text)
    if not os.path.exists(savepath):
        # In[]
        # 构建字典，便于查询
        print("Bulding dict")
        text2emb_path = args.outpath + "/{}_countvec_{}.pkl".format(mode,text)
        if not os.path.exists(text2emb_path): 
            text_df = []
            with open(args.datapath + "/prepare/{}_data_pub.json".format(mode), "rb") as f:
                train_pubs = pickle.load(f)
            for paper_id in tqdm(train_pubs):
                if text in ['title','abstract','venue']:
                    seq = etl(train_pubs[paper_id][text]).lower()
                    if seq == "":
                        seq = "null"
                    seq = seq.split()
                    text_df.append([paper_id,seq]) 
                if text in ['title_abstract']:
                    seq1 = etl(train_pubs[paper_id]['title']).lower()
                    seq2 = etl(train_pubs[paper_id]['abstract']).lower()
                    seq  = seq1 + " " + seq2
                    if seq == '':
                        seq = 'null'
                    seq = seq.split()
                    text_df.append([paper_id,seq])
                if text in ['keywords']:
                    new_keywords = []
                    for keyword in train_pubs[paper_id][text]:
                        keyword = etl(keyword).lower()
                        if keyword == "":
                            keyword = "null"
                        keyword = keyword.split()
                        new_keywords.extend(keyword)
                    text_df.append([paper_id,new_keywords]) 
            
            del train_pubs
            gc.collect()
            
            if mode == "na_v3":
                for pub_path in [args.datapath+"/{}/cna-valid/whole_author_profiles_pub.json".format(mode),
                                 args.datapath+"/{}/cna-valid/cna_valid_unass_pub.json".format(mode),
                                 args.datapath+"/{}/cna-test/cna_test_unass_pub.json".format(mode)]:
                    with open(pub_path, "r",encoding='utf-8') as f:
                        pubs = json.load(f)
                  
                    for paper_id in tqdm(pubs):
                        if text in ['title','abstract','venue']:
                            seq = etl(pubs[paper_id][text]).lower()
                            if seq == "":
                                seq = "null"
                            seq = seq.split()
                            text_df.append([paper_id,seq]) 
                        if text in ['title_abstract']:
                            seq1 = etl(pubs[paper_id]['title']).lower()
                            seq2 = etl(pubs[paper_id]['abstract']).lower()
                            seq  = seq1 + " " + seq2
                            if seq == '':
                                seq = 'null'
                            seq = seq.split()
                            text_df.append([paper_id,seq])
                        if text in ['keywords']:
                            new_keywords = []
                            for keyword in pubs[paper_id][text]:
                                keyword = etl(keyword).lower()
                                if keyword == "":
                                    keyword = "null"
                                keyword = keyword.split()
                                new_keywords.extend(keyword)
                            text_df.append([paper_id,new_keywords]) 
                    del pubs
                    gc.collect()
                
            text_df = pd.DataFrame(text_df,columns=['paper_id',text])
            text_df = text_df.drop_duplicates(subset=['paper_id'],keep='last').reset_index(drop=True)
            cntvec_feat = gen_countvec_features(text_df[[text]],text,n=n_components)
            cntvec_feat['svd_countvec_{}'.format(text)] = cntvec_feat.apply(lambda xs:[x for x in xs],axis=1)
            text_df = pd.concat([text_df[['paper_id']],cntvec_feat[['svd_countvec_{}'.format(text)]]],axis=1)
            print(text_df['svd_countvec_{}'.format(text)])
                
            text2emb = {}
            for paper_id,vec in tqdm(zip(text_df['paper_id'].values,text_df['svd_countvec_{}'.format(text)].values)):
                text2emb[paper_id] = vec
                
            with open(text2emb_path, 'wb') as f:
                pickle.dump(text2emb, f)
                
        # In[]
        print("Making Feature")            
        with open(text2emb_path, "rb") as f:
            text2emb = pickle.load(f)
            
        data['author_id'] = data['author_id'].fillna(0).astype(str)
        data = data[data['author_id']!='0'].reset_index(drop=True)
        all_feat_df = pd.DataFrame()
        for i in tqdm(range(100)):
            df = data[int(len(data)*(i/100)):int(len(data)*((i+1)/100))].reset_index(drop=True)
            if text == "title_abstract":
                df['paper_{}'.format(text)] = df[['paper_title','paper_abstract']].apply(
                        lambda x:etl(x[0]).lower()+etl(x[1]).lower(),axis=1)
            
            all_w2v_cosine    = []
            all_w2v_cityblock = []
            for paper_id,seq,paper_id_list in zip(df['paper_id'].values,df['paper_{}'.format(text)].values,df['paper_id_list'].values):
                if text in ['title','abstract','venue','title_abstract']:
                    seq = etl(seq).lower()
                else:
                    seq = ' '.join(seq)
                    seq = etl(seq).lower()
                
                if seq == '':
                    w2v_cosine    = []
                    w2v_cityblock = []
                else:
                    w2v_cosine    = []
                    w2v_cityblock = []
                    
                    for paper_id2 in paper_id_list:        
                        w2v_cosine.append(distance.cosine(text2emb[paper_id],text2emb[paper_id2]))
                        w2v_cityblock.append(distance.cityblock(text2emb[paper_id],text2emb[paper_id2]))
               
                all_w2v_cosine.append(w2v_cosine)
                all_w2v_cityblock.append(w2v_cityblock)
                
            df['{}_countvec_cosine_list'.format(text)] = all_w2v_cosine
            df['{}_countvec_cityblock_list'.format(text)] = all_w2v_cityblock
         
            drop_columns = [f for f in df.columns if f not in ['idx']]   
    
            for col in ['{}_countvec_cosine_list'.format(text),'{}_countvec_cityblock_list'.format(text)]:
                print(df[col])
                df['{}_mean'.format(col)] = df[col].apply(lambda x:np.mean(x) if x!=[] else -1)
                df['{}_max'.format(col)]  = df[col].apply(lambda x:get_max_min(x,mode='max') if x!=[] else -1)
                df['{}_min'.format(col)]  = df[col].apply(lambda x:get_max_min(x,mode='min') if x!=[] else -1)
                df['{}_std'.format(col)]  = df[col].apply(lambda x:np.std(x) if x!=[] else -1)
                  
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
        for text in ['title_abstract','title','abstract','keywords','venue']:
            train_df = pd.read_pickle(args.featpath + "/{}_train.pkl".format(mode))
            extract_text_countvec_sim_feature(args,train_df,mode=mode,sign="train",text=text)
    
    # 构建测试集特征
    for mode in ['na_v3']:
        for text in ['title_abstract','title','abstract','keywords','venue']:
            test_df = pd.read_pickle(args.featpath + "/{}_testa.pkl".format(mode))
            extract_text_countvec_sim_feature(args,test_df,mode=mode,sign="testa",text=text)
            
            test_df = pd.read_pickle(args.featpath + "/{}_testb.pkl".format(mode))
            extract_text_countvec_sim_feature(args,test_df,mode=mode,sign="testb",text=text)
