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

def get_all_paper_org_list(x,paperid_author_org_dict):
    paper_author  = x[0]
    paper_id_list = x[1]
    
    org_list = []
    for paper_id in paper_id_list:
        if paper_author in paperid_author_org_dict[paper_id]:
            org = paperid_author_org_dict[paper_id][paper_author]
            if org != '':
                org_list.append(org)         
        else:
            org = ''
            
    return org_list

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

# 预处理机构,简写替换，
def preprocessorg(org):
    if org != "":
        org = org.replace('sch.', 'school')
        org = org.replace('dept.', 'department')
        org = org.replace('coll.', 'college')
        org = org.replace('inst.', 'institute')
        org = org.replace('univ.', 'university')
        org = org.replace('lab ', 'laboratory ')
        org = org.replace('lab.', 'laboratory')
        org = org.replace('natl.', 'national')
        org = org.replace('comp.', 'computer')
        org = org.replace('sci.', 'science')
        org = org.replace('tech.', 'technology')
        org = org.replace('technol.','technology')
        org = org.replace('elec.', 'electronic')
        org = org.replace('engr.', 'engineering')
        org = org.replace('aca.', 'academy')
        org = org.replace('syst.', 'systems')
        org = org.replace('eng.', 'engineering')
        org = org.replace('res.', 'research')
        org = org.replace('appl.', 'applied')
        org = org.replace('chem.', 'chemistry')
        org = org.replace('prep.', 'petrochemical')
        org = org.replace('phys.', 'physics')
        org = org.replace('phys ', 'physics')
        org = org.replace('mech.', 'mechanics')
        org = org.replace('mat.', 'material')
        org = org.replace('cent.', 'center')
        org = org.replace('ctr.', 'center')
        org = org.replace('behav.', 'behavior')
        org = org.replace('atom.', 'atomic')
        org = org.replace('|', ' ')
        org = org.replace('-', ' ')
        org = org.replace('{', ' ')
        org = org.split(';')[0]  # 多个机构只取第一个
    return etl(org).strip()

def paper_org_set_count(x):
    if x[0]=='':
        return []
    elif x[1]==[]:
        return []
    else:
        paper_org = x[0].split(" ")
        
        len_list = []
        for paper_org2 in x[1]:
            paper_org2 = paper_org2.split(" ")
            len_list.append(len(set(paper_org)&set(paper_org2)))
        return len_list
    
    
def extract_org_distance_feature(args,data,mode="na_v1",sign="train"):
    savepath = args.featpath + "/{}_{}_org_feature.pkl".format(mode,sign)
    if not os.path.exists(savepath):
        print("Making Feature")
        
        if sign == "train":
            with open(args.datapath + "/prepare/{}_data_pub.json".format(mode), "rb") as f:
                pubs = pickle.load(f)
        else:       
            with open(args.datapath+"/{}/cna-valid/whole_author_profiles_pub.json".format(mode), "r",encoding='utf-8') as f:
                pubs = json.load(f)
        
        # 构建字典
        paperid_author_org_dict = {}
        for paper_id in tqdm(pubs):
            authors_orgs = pubs[paper_id]['authors']
            authors_list = [fix_name(convert_name(dit['name'])) for dit in authors_orgs]
            org_list = [dit['org'].lower() for dit in authors_orgs]
            
            dict_tmp = {}
            for author,org in zip(authors_list,org_list):
                dict_tmp[author] = preprocessorg(org)    
            paperid_author_org_dict[paper_id] = dict_tmp
            
        data['author_id'] = data['author_id'].fillna(0).astype(str)
        data = data[data['author_id']!='0'].reset_index(drop=True)
        
        all_feat_df = pd.DataFrame()
        for i in tqdm(range(100)):
            df = data[int(len(data)*(i/100)):int(len(data)*((i+1)/100))].reset_index(drop=True)
            
            # 构造特征
            df['paper_author'] = df['paper_author'].apply(lambda x:fix_name(convert_name(x)))
            df['paper_author_list'] = df['paper_author_list'].apply(lambda x:[fix_name(convert_name(k)) for k in x])
            df['paper_author_list'] = df['paper_author_list'].apply(lambda x:[fix_name(convert_name(k)) for k in x])
            
            # 去除同名作者
            df['paper_author_list'] = df[['paper_author','paper_author_list']].apply(lambda x:[k for k in x[1] if k != x[0]],axis=1)
            
            df['paper_org'] = df['paper_org'].apply(lambda x:str(x).lower())
            df['paper_org'] = df['paper_org'].apply(preprocessorg)
            df['all_paper_org_list'] = df[['paper_author','paper_id_list']].apply(lambda x:get_all_paper_org_list(x,paperid_author_org_dict),axis=1)
            print(df['all_paper_org_list'])
            drop_columns = [f for f in df.columns if f not in ['idx']]   
            for distance_type in ['jaro_winkler','ratio','jaro']:
                df['paper_org_{}_list'.format(distance_type)] = df[['paper_org','all_paper_org_list']].apply(
                        lambda x:get_levenshtein_distance(x,distance_type=distance_type),axis=1)
                df['paper_org_{}_sum'.format(distance_type)]  = df['paper_org_{}_list'.format(distance_type)].apply(
                        lambda x:np.sum(x) if x!=[] else -1)
                df['paper_org_{}_mean'.format(distance_type)] = df['paper_org_{}_list'.format(distance_type)].apply(
                        lambda x:np.mean(x) if x!=[] else -1)
                df['paper_org_{}_max'.format(distance_type)]  = df['paper_org_{}_list'.format(distance_type)].apply(
                        lambda x:np.max(x) if x!=[] else -1)
                df['paper_org_{}_min'.format(distance_type)]  = df['paper_org_{}_list'.format(distance_type)].apply(
                        lambda x:np.min(x) if x!=[] else -1)
                df['paper_org_{}_std'.format(distance_type)]  = df['paper_org_{}_list'.format(distance_type)].apply(
                        lambda x:np.std(x) if x!=[] else -1)
                df['paper_org_{}_std'.format(distance_type)]  = df['paper_org_{}_list'.format(distance_type)].apply(
                        lambda x:np.std(x) if x!=[] else -1)
                if distance_type == "jaro_winkler":
                    df['paper_org_{}_1_count'.format(distance_type)]  = df['paper_org_{}_list'.format(distance_type)].apply(
                            lambda x:x.count(1) if x!=[] else -1)
                    df['paper_org_{}_1_ratio'.format(distance_type)]  = df[['paper_org_{}_1_count'.format(distance_type),'paper_org_{}_list'.format(distance_type)]].apply(
                            lambda x:x[0]/len(x[1]) if x[0]!=-1 else -1,axis=1)
                print(df['paper_org_{}_list'.format(distance_type)])
                df.drop(['paper_org_{}_list'.format(distance_type)],axis=1,inplace=True)
            
            # 无用特征
            if False:
                for distance_type in ['distance']:
                    df['paper_org_{}_list'.format(distance_type)] = df[['paper_org','all_paper_org_list']].apply(
                            lambda x:get_levenshtein_distance(x,distance_type=distance_type),axis=1)
                    df['paper_org_{}2_list'.format(distance_type)] = df[['paper_org','paper_org_{}_list'.format(distance_type)]].apply(
                            lambda x:[k / len(fix_name(x[0])) for k in x[1]] if x[1]!=[] else [],axis=1)
                    
                    for col in ['paper_org_{}_list'.format(distance_type),'paper_org_{}2_list'.format(distance_type)]:
                        print(df[col])
                        df['{}_sum'.format(col)]  = df[col].apply(lambda x:np.sum(x) if x!=[] else -1)
                        df['{}_mean'.format(col)] = df[col].apply(lambda x:np.mean(x) if x!=[] else -1)
                        df['{}_max'.format(col)]  = df[col].apply(lambda x:np.max(x) if x!=[] else -1)
                        df['{}_min'.format(col)]  = df[col].apply(lambda x:np.min(x) if x!=[] else -1)
                        df['{}_std'.format(col)]  = df[col].apply(lambda x:np.std(x) if x!=[] else -1)
                       
                        df.drop([col],axis=1,inplace=True)
            
            # 集合的交集
            df['paper_org_set_count'] = df[['paper_org','all_paper_org_list']].apply(lambda x:paper_org_set_count(x),axis=1)
            print(df['paper_org_set_count'])
            df['paper_org_set_count_sum']  = df['paper_org_set_count'].apply(lambda x:-1 if x==[] else np.sum(x))
            df['paper_org_set_count_mean'] = df['paper_org_set_count'].apply(lambda x:-1 if x==[] else np.mean(x))
            df['paper_org_set_count_max']  = df['paper_org_set_count'].apply(lambda x:-1 if x==[] else np.max(x))
            df['paper_org_set_count_min']  = df['paper_org_set_count'].apply(lambda x:-1 if x==[] else np.min(x))
            df['paper_org_set_count_std']  = df['paper_org_set_count'].apply(lambda x:-1 if x==[] else np.std(x))
            
            df['paper_org_set_count_ratio'] = df[['paper_org_set_count','paper_org']].apply(lambda x:[k/(len(x[1].strip())+1) for k in x[0]] if x[0]!=[] else [],axis=1)
            print(df['paper_org_set_count_ratio'])
            df['paper_org_set_count_ratio_sum']  = df['paper_org_set_count_ratio'].apply(lambda x:-1 if x==[] else np.sum(x))
            df['paper_org_set_count_ratio_mean'] = df['paper_org_set_count_ratio'].apply(lambda x:-1 if x==[] else np.mean(x))
            df['paper_org_set_count_ratio_max']  = df['paper_org_set_count_ratio'].apply(lambda x:-1 if x==[] else np.max(x))
            df['paper_org_set_count_ratio_std']  = df['paper_org_set_count_ratio'].apply(lambda x:-1 if x==[] else np.std(x))
            
            df.drop(['paper_org_set_count','paper_org_set_count_ratio'],axis=1,inplace=True)
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
        extract_org_distance_feature(args,train_df,mode=mode,sign="train")
    
    # 构建测试集特征
    for mode in ['na_v3']:
        test_df = pd.read_pickle(args.featpath + "/{}_testa.pkl".format(mode))
        extract_org_distance_feature(args,test_df,mode=mode,sign="testa")
        
        test_df = pd.read_pickle(args.featpath + "/{}_testb.pkl".format(mode))
        extract_org_distance_feature(args,test_df,mode=mode,sign="testb")
