# -*- coding: utf-8 -*-
import os
import gc
import re
import json
import pickle
import Levenshtein
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.spatial import distance
from gensim.models import Word2Vec
from args import get_parser
from utils import reduce_mem_usage,distance_score

def get_levenshtein_distance(x,distance_type='jaro_winkler'):
    if x[0] == '':
        return []
    elif x[1] == []:
        return []
    else:
        if distance_type == "jaro_winkler":
            return [Levenshtein.jaro_winkler(x[0],s) for s in x[1]]
        if distance_type == "distance":
            return [distance_score(x[0],s) for s in x[1]]
        if distance_type == "ratio":
            return [Levenshtein.ratio(x[0],s) for s in x[1]]
        if distance_type == "jaro":
            return [Levenshtein.jaro(x[0],s) for s in x[1]]
    
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

def extract_weight_word2vec_sim_feature(args,data,mode="na_v1",sign="train",text='title',all_text=['title','abstract','venue','keywords']):
    embsize = 128
    savepath = args.featpath + "/{}_{}_{}_weight_word2vec_sim_feature.pkl".format(mode,sign,text)
    if not os.path.exists(savepath):
        # 训练词向量
        # In[]
        w2vpath = args.w2vpath+'/{}_text_w2v_{}.model'.format(mode,embsize)
        if not os.path.exists(w2vpath):
            print("Training Word2Vec Model")
            all_sentences = []
            with open(args.datapath + "/prepare/{}_data_pub.json".format(mode), "rb") as f:
                train_pubs = pickle.load(f)
    
                for paper_id in tqdm(train_pubs):
                    for col in all_text:
                        if col != "keywords":
                            seq = etl(train_pubs[paper_id][col]).lower()
                            if seq == '':
                                seq = "null"
                            seq = seq.split()
                            all_sentences.append(seq)  
                        else:
                            new_keywords = []
                            for keyword in train_pubs[paper_id]['keywords']:
                                keyword = etl(keyword).lower()
                                if keyword == '':
                                    keyword = "null"
                                keyword = keyword.split()
                                new_keywords.extend(keyword)
                            all_sentences.append(new_keywords) 
                del train_pubs
                gc.collect()
                            
            if mode=="na_v3":    
                with open(args.datapath+"/{}/cna-valid/whole_author_profiles_pub.json".format(mode), "r",encoding='utf-8') as f:
                    whole_pubs = json.load(f)
                    
                    for paper_id in tqdm(whole_pubs):
                        for col in all_text:
                            if col != "keywords":
                                seq = etl(whole_pubs[paper_id][col]).lower()
                                if seq == '':
                                    seq = "null"
                                seq = seq.split()
                                all_sentences.append(seq)  
                            else:
                                new_keywords = []
                                for keyword in whole_pubs[paper_id]['keywords']:
                                    keyword = etl(keyword).lower()
                                    if keyword == '':
                                        keyword = "null"
                                    keyword = keyword.split()
                                    new_keywords.extend(keyword)
                                all_sentences.append(new_keywords) 
                                    
                    del whole_pubs
                    gc.collect()
                                
                with open(args.datapath+"/{}/cna-valid/cna_valid_unass_pub.json".format(mode), "r",encoding='utf-8') as f:
                    valid_pubs = json.load(f)
        
                    for paper_id in tqdm(valid_pubs):
                        for col in all_text:
                            if col != "keywords":
                                seq = etl(valid_pubs[paper_id][col]).lower()
                                if seq == '':
                                    seq = "null"
                                seq = seq.split()
                                all_sentences.append(seq)  
                            else:
                                new_keywords = []
                                for keyword in valid_pubs[paper_id]['keywords']:
                                    keyword = etl(keyword).lower()
                                    if keyword == '':
                                        keyword = "null"
                                    keyword = keyword.split()
                                    new_keywords.extend(keyword)
                                all_sentences.append(new_keywords) 
                    del valid_pubs
                    gc.collect()
                                
                with open(args.datapath+"/{}/cna-test/cna_test_unass_pub.json".format(mode), "r",encoding='utf-8') as f:
                    test_pubs = json.load(f)
                    
                    for paper_id in tqdm(test_pubs):
                        for col in all_text:
                            if col != "keywords":
                                seq = etl(test_pubs[paper_id][col]).lower()
                                if seq == '':
                                    seq = "null"
                                seq = seq.split()
                                all_sentences.append(seq)  
                            else:
                                new_keywords = []
                                for keyword in test_pubs[paper_id]['keywords']:
                                    keyword = etl(keyword).lower()
                                    if keyword == '':
                                        keyword = "null"
                                    keyword = keyword.split()
                                    new_keywords.extend(keyword)
                                all_sentences.append(new_keywords) 
                    del test_pubs
                    gc.collect()
            
            model = Word2Vec(all_sentences,size=embsize, window=10,min_count=5, sg=1, seed=42,iter=10)
            model.save(w2vpath)
        # In[]
        # 构建字典，便于查询
        text2emb_path = args.outpath + "/{}_{}_word2vec_{}.pkl".format(mode,sign,text)
        if not os.path.exists(text2emb_path): 
            print("Bulding dict")
            model = Word2Vec.load(w2vpath) 
            text_df = []
            if sign == "train":
                with open(args.datapath + "/prepare/{}_data_pub.json".format(mode), "rb") as f:
                    train_pubs = pickle.load(f)
                for paper_id in tqdm(train_pubs):
                    if text in ['title','abstract','venue']:
                        seq = etl(train_pubs[paper_id][text]).lower()
                        if seq == '':
                            seq = 'null'
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
                            if keyword == '':
                                keyword = 'null'
                            keyword = keyword.split()
                            new_keywords.extend(keyword)
                        text_df.append([paper_id,new_keywords]) 
                
                del train_pubs
                gc.collect()
                    
            if sign in ['testa','testb']:
                with open(args.datapath+"/{}/cna-valid/whole_author_profiles_pub.json".format(mode), "r",encoding='utf-8') as f:
                    whole_pubs = json.load(f)
              
                for paper_id in tqdm(whole_pubs):
                    if text in ['title','abstract','venue']:
                        seq = etl(whole_pubs[paper_id][text]).lower()
                        if seq == '':
                            seq = 'null'
                        seq = seq.split()
                        text_df.append([paper_id,seq])  
                    if text in ['title_abstract']:
                        seq1 = etl(whole_pubs[paper_id]['title']).lower()
                        seq2 = etl(whole_pubs[paper_id]['abstract']).lower()
                        seq  = seq1 + " " + seq2
                        if seq == '':
                            seq = 'null'
                        seq = seq.split()
                        text_df.append([paper_id,seq])
                    if text in ['keywords']:
                        new_keywords = []
                        for keyword in whole_pubs[paper_id][text]:
                            keyword = etl(keyword).lower()
                            if keyword == '':
                                keyword = 'null'
                            keyword = keyword.split()
                            new_keywords.extend(keyword)
                        text_df.append([paper_id,new_keywords]) 
                del whole_pubs
                gc.collect()
                    
                if sign == "testa":
                    with open(args.datapath+"/{}/cna-valid/cna_valid_unass_pub.json".format(mode), "r",encoding='utf-8') as f:
                        test_pubs = json.load(f)
                if sign == "testb":
                    with open(args.datapath+"/{}/cna-test/cna_test_unass_pub.json".format(mode), "r",encoding='utf-8') as f:
                        test_pubs = json.load(f)
                        
                for paper_id in tqdm(test_pubs):
                    if text in ['title','abstract','venue']:
                        seq = etl(test_pubs[paper_id][text]).lower()
                        if seq == '':
                            seq = 'null'
                        seq = seq.split()
                        text_df.append([paper_id,seq])  
                    if text in ['title_abstract']:
                        seq1 = etl(test_pubs[paper_id]['title']).lower()
                        seq2 = etl(test_pubs[paper_id]['abstract']).lower()
                        seq  = seq1 + " " + seq2
                        if seq == '':
                            seq = 'null'
                        seq = seq.split()
                        text_df.append([paper_id,seq])
                    if text in ['keywords']:
                        new_keywords = []
                        for keyword in test_pubs[paper_id][text]:
                            keyword = etl(keyword).lower()
                            if keyword == '':
                                keyword = 'null'
                            keyword = keyword.split()
                            new_keywords.extend(keyword)
                        text_df.append([paper_id,new_keywords]) 
                del test_pubs
                gc.collect()
                
            text_df = pd.DataFrame(text_df,columns=['paper_id',text])
                
            text2emb = {}
            for paper_id,sentences in tqdm(zip(text_df['paper_id'].values,text_df[text].values)):
                vec = [model[s] for s in sentences if s in model]
                vec = np.mean(vec, axis=0)
                text2emb[paper_id] = vec
                
            with open(text2emb_path, 'wb') as f:
                pickle.dump(text2emb, f)
        # In[]
        print("Making Feature")            
        with open(text2emb_path, "rb") as f:
            text2emb = pickle.load(f)
            
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
        print(data['paper_id_list'])
        
        all_feat_df = pd.DataFrame()
        for i in tqdm(range(100)):
            w2v_cosine_list = []
            w2v_cityblock_list   = []
            df = data[int(len(data)*(i/100)):int(len(data)*((i+1)/100))].reset_index(drop=True)
            if text == "title_abstract":
                df['paper_{}'.format(text)] = df[['paper_title','paper_abstract']].apply(
                        lambda x:etl(x[0]).lower()+etl(x[1]).lower(),axis=1)
                
            for paper_id,seq,paper_id_list,paper_year in zip(df['paper_id'].values,
                                                  df['paper_{}'.format(text)].values,
                                                  df['paper_id_list'].values,
                                                  df['paper_year']):
                if text in ['keywords']:
                    seq = ' '.join(seq)
                    seq = etl(seq).lower()
                else:
                    seq = etl(seq).lower()
                
                if seq == '':
                    w2v_cosine    = np.nan
                    w2v_cityblock = np.nan
                else:
                    weight_list = []
                    for paper_year2 in [paperid_year_dict[pid] for pid in paper_id_list]:
                        try:
                            gap = abs(int(paper_year)-int(paper_year2))+1
                            if gap>20:
                                gap = 20
                            weight = 1 / gap
                        except:
                            weight = 1 / 20
                        weight_list.append(weight)
                    
                    emb = np.array([0]*embsize)
                    for weight,paper_id2 in zip(weight_list,paper_id_list):
                        emb = emb + np.array(text2emb[paper_id2])*weight
                        
                    w2v_cosine = distance.cosine(text2emb[paper_id],emb/np.sum(weight_list))
                    w2v_cityblock = distance.cityblock(text2emb[paper_id],emb/np.sum(weight_list))
                
                w2v_cosine_list.append(w2v_cosine)
                w2v_cityblock_list.append(w2v_cityblock)
            
            drop_columns = [f for f in df.columns if f not in ['idx']]  
            df['{}_wgight_w2v_cosine'.format(text)] = w2v_cosine_list
            print(df['{}_wgight_w2v_cosine'.format(text)])
            df['{}_weight_w2v_cityblock'.format(text)] = w2v_cityblock_list
         
            df.drop(drop_columns,axis=1,inplace=True)
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
        train_df    = pd.read_pickle(args.featpath + "/{}_train.pkl".format(mode))
        train_feats = train_df[['idx']]
        for text in ['title_abstract','title','abstract','keywords','venue']:
            feat_df = extract_weight_word2vec_sim_feature(args,train_df,mode=mode,sign="train",text=text)
            train_feats = pd.merge(train_feats,feat_df,on=['idx'],how='left')
        train_feats = reduce_mem_usage(train_feats)
        train_feats.to_pickle(args.featpath + "/{}_train_weight_word2vec_sim_feature.pkl".format(mode))
    
    # 构建测试集特征
    for mode in ['na_v3']:
        test_df   = pd.read_pickle(args.featpath + "/{}_testa.pkl".format(mode))
        test_feats = test_df[['idx']]
        for text in ['title_abstract','title','abstract','keywords','venue']:
            feat_df = extract_weight_word2vec_sim_feature(args,test_df,mode=mode,sign="testa",text=text)
            test_feats = pd.merge(test_feats,feat_df,on=['idx'],how='left')
        test_feats = reduce_mem_usage(test_feats)
        test_feats.to_pickle(args.featpath + "/{}_testa_weight_word2vec_sim_feature.pkl".format(mode))
    
    for mode in ['na_v3']:
        test_df   = pd.read_pickle(args.featpath + "/{}_testb.pkl".format(mode))
        test_feats = test_df[['idx']]
        for text in ['title_abstract','title','abstract','keywords','venue']:
            feat_df = extract_weight_word2vec_sim_feature(args,test_df,mode=mode,sign="testb",text=text)
            test_feats = pd.merge(test_feats,feat_df,on=['idx'],how='left')
        test_feats = reduce_mem_usage(test_feats)
        test_feats.to_pickle(args.featpath + "/{}_testb_weight_word2vec_sim_feature.pkl".format(mode))