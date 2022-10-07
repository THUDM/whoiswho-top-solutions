# -*- coding: utf-8 -*-
import os
import json
import pickle
import pandas as pd
from tqdm import tqdm
from args import get_parser

def read_train_author(mode):
    with open(args.datapath+"/{}/train/train_author.json".format(mode), "r",encoding='utf-8') as f:
        json_df = json.load(f)  
    
    train_author = []
    for author_name in json_df:
        for author_id in json_df[author_name]:
            train_author.append([author_name,author_id,json_df[author_name][author_id]])
    train_author = pd.DataFrame(train_author)
    train_author.columns = ['author_name','author_id','paper_ids']
        
    return train_author

def read_whole_author(mode):
    with open(args.datapath+"/{}/cna-valid/whole_author_profiles.json".format(mode), "r",encoding='utf-8') as f:
        json_df = json.load(f)  
    
    whole_author = []
    for author_id in json_df:
        author_name = json_df[author_id]['name']
        if mode == "na_v3":
            paper_ids   = json_df[author_id]['pubs']
        else:
            paper_ids   = json_df[author_id]['papers']
        whole_author.append([author_name,author_id,paper_ids])
    whole_author = pd.DataFrame(whole_author,columns=['author_name','author_id','paper_ids'])
    
    return whole_author

def multidList_to_1dList(multi_list):
    newList = []
    for s in multi_list:
        newList.extend(s)
    newList = list(set(newList))
    return newList

def read_valid_author(mode):
    with open(args.datapath+"/{}/cna-valid/cna_valid_author_ground_truth.json".format(mode), "r",encoding='utf-8') as f:
        json_df = json.load(f) 
        
    valid_author = []
    for author_name in json_df:
        for author_id in json_df[author_name]:
            valid_author.append([author_name,author_id,json_df[author_name][author_id]])
    valid_author = pd.DataFrame(valid_author)
    valid_author.columns = ['author_name','author_id','paper_ids']
    
    return valid_author

def read_test_author(mode):
    with open(args.datapath+"/{}/cna-valid/cna_test_author_ground_truth.json".format(mode), "r",encoding='utf-8') as f:
        json_df = json.load(f) 
        
    valid_author = []
    for author_name in json_df:
        for author_id in json_df[author_name]:
            valid_author.append([author_name,author_id,json_df[author_name][author_id]])
    valid_author = pd.DataFrame(valid_author)
    valid_author.columns = ['author_name','author_id','paper_ids']
    
    return valid_author

def read_data_author(mode):
    savepath = args.datapath + "/prepare/{}_data_author.json".format(mode)
    if not os.path.exists(savepath):
        # 读取 na_v1 中的数据集
        # 读取训练集 train_author.json
        train_author = read_train_author(mode)
        # 读取候选文档集 whole_author_profile.json
        whole_author = read_whole_author(mode)
        
        if mode in ["na_v1","na_v2"]:
            # 读取验证集 cna_valid_author_ground_truth.json
            valid_author = read_valid_author(mode)
            # 读取验证集 cna_test_author_ground_truth.json
            test_author = read_test_author(mode)
            data_author = pd.concat([train_author,whole_author,valid_author,test_author],axis=0).reset_index(drop=True)
            
        else:
            data_author = pd.concat([train_author,whole_author],axis=0).reset_index(drop=True)
            
        data_author['author_id'] = data_author['author_id'].apply(lambda x:x.split("-")[0])
        print(data_author[['author_name','author_id']])
        print(data_author['paper_ids'])
        print(data_author.shape)
        data_author = data_author.groupby(['author_name','author_id'])['paper_ids'].apply(lambda x:list(x)).reset_index()
        data_author['paper_ids'] = data_author['paper_ids'].apply(multidList_to_1dList)
        print(data_author['paper_ids'])
        print(data_author.shape)
        data = data_author.drop_duplicates(subset=['author_name'])[['author_name']].reset_index(drop=True)
        for col in ['author_id','paper_ids']:
            tmp = data_author.groupby(['author_name'])[col].apply(lambda x:list(x)).reset_index()
            data = pd.merge(data,tmp,on=['author_name'],how='left')
        
        data_author_dict = {}
        for author_name,author_ids,paper_ids in zip(data['author_name'].values,data['author_id'].values,data['paper_ids'].values):
            tmp_dict = {}
            for author_id,paper_id in zip(author_ids,paper_ids):
                tmp_dict[author_id] = paper_id
            data_author_dict[author_name] = tmp_dict
        
        os.makedirs(args.datapath + "/prepare",exist_ok =True)
        with open(savepath, 'wb') as f:
            pickle.dump(data_author_dict, f)
                    
    with open(savepath, "rb") as f:
        data_author_dict = pickle.load(f)
        
    return data_author_dict

def get_pub_unit(data_pub,json_df):
    for paper_id in tqdm(json_df):
        title    = json_df[paper_id]['title']
        try:
            abstract = json_df[paper_id]['abstract']
        except:
            abstract = '' 
        try:
            keywords = json_df[paper_id]['keywords']
        except:
            keywords = ''
            
        authors  = []
        for author in json_df[paper_id]['authors']:
            name = author['name']
            try:
                org = author['org']
            except:
                org = ''
            tmp_dict = {"name":name,"org":org}
            authors.append(tmp_dict)
            
        venue    = json_df[paper_id]['venue']
        try:
            year = json_df[paper_id]['year']
        except:
            year = ''
        data_pub.append([paper_id,title,abstract,keywords,authors,venue,year])
        
    return data_pub

def read_data_pub(mode):
    savepath = args.datapath+"/prepare/{}_data_pub.json".format(mode)
    if not os.path.exists(savepath):
        data_pub = []
        with open(args.datapath+"/{}/train/train_pub.json".format(mode), "r",encoding='utf-8') as f:
            json_df = json.load(f) 
            data_pub = get_pub_unit(data_pub,json_df)
            
        with open(args.datapath+"/{}/cna-valid/whole_author_profiles_pub.json".format(mode), "r",encoding='utf-8') as f:
            json_df = json.load(f) 
            data_pub = get_pub_unit(data_pub,json_df)
            
        if mode in ["na_v1","na_v2"]:
            # 读取 cna_valid_pub.json
            with open(args.datapath+"/{}/cna-valid/cna_valid_pub.json".format(mode), "r",encoding='utf-8') as f:
                json_df = json.load(f) 
                data_pub = get_pub_unit(data_pub,json_df)
            
            # 读取 cna_test_pub.json
            with open(args.datapath+"/{}/cna-valid/cna_test_pub.json".format(mode), "r",encoding='utf-8') as f:
                json_df = json.load(f) 
                data_pub = get_pub_unit(data_pub,json_df)
            
        data_pub = pd.DataFrame(data_pub,columns=['paper_id','title','abstract','keywords','authors','venue','year'])
        print(data_pub.shape)
        data_pub = data_pub.drop_duplicates(subset=['paper_id','title'],keep='last').reset_index(drop=True)
        print(data_pub.shape)
        
        data_pub_dict = {}
        for paper_id,title,abstract,keywords,authors,venue,year in zip(data_pub['paper_id'].values,
                                                                       data_pub['title'].values,
                                                                       data_pub['abstract'].values,
                                                                       data_pub['keywords'].values,
                                                                       data_pub['authors'].values,
                                                                       data_pub['venue'].values,
                                                                       data_pub['year'].values
                                                                       ):
            tmp_dict = {}
            tmp_dict['title']    = title
            tmp_dict['abstract'] = abstract
            tmp_dict['keywords'] = keywords
            tmp_dict['authors']  = authors
            tmp_dict['venue']    = venue
            tmp_dict['year']     = year
            data_pub_dict[paper_id] = tmp_dict
        
        os.makedirs(args.datapath + "/prepare",exist_ok =True)
        with open(savepath, 'wb') as f:
            pickle.dump(data_pub_dict, f)
                    
    with open(savepath, "rb") as f:
        data_pub_dict = pickle.load(f)
        
    return data_pub_dict

if __name__=="__main__":
    args = get_parser()
    for mode in ['na_v3']:
        read_data_author(mode)
        read_data_pub(mode)