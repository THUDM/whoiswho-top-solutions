# -*- coding: utf-8 -*-
import os
import re
import pickle
import json
import random
from args import get_parser

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
        
def split_data(mode):
    savepath = args.outpath+'/{}_train_unass_data.json'.format(mode)
    if not os.path.exists(savepath):
         # 读取 训练集 train_author.json
        with open(args.datapath + "/prepare/{}_data_author.json".format(mode), "rb") as f:
            train_author = pickle.load(f)
        
        # 读取 训练集 
        with open(args.datapath + "/prepare/{}_data_pub.json".format(mode), "rb") as f:
            train_pub = pickle.load(f)
            
        count = 0
        unass_data = []
        existing_data = {}
        for author_name in train_author:
            if len(train_author[author_name]) > 2:
                for person_id in train_author[author_name]:
                    papers_of_person = len(train_author[author_name][person_id])
                    if papers_of_person >= 2:
                        for i in range(min(20,papers_of_person)):
                            if papers_of_person <= 20:
                                paper_rank = i
                            else:
                                paper_rank = random.randint(0, papers_of_person - 1) 
                                
                            the_paper_id = train_author[author_name][person_id][paper_rank]
                            authors_of_the_paper = train_pub[the_paper_id]['authors']
                            
                            for index, authors_info in enumerate(authors_of_the_paper):
                                if fix_name(authors_info['name']) == fix_name(author_name):
                                    count += 1
                                    unass_data.append((the_paper_id + '-' + str(index), person_id))
                                    existing_data[person_id] = {"name": author_name, "pubs": [paper_id for paper_id in train_author[author_name][person_id]]}
                                    
                                    
        print('len unass_data', len(unass_data))
        # unass_data -> 随机抽一篇
        # existing_data -> 其余的加入训练集
        with open(savepath, 'w') as w:
            w.write(json.dumps(unass_data))
        with open(args.outpath+'/{}_train_existing_data.json'.format(mode), 'w') as w:
            w.write(json.dumps(existing_data))
                            
if __name__=="__main__":
    args = get_parser()
    for mode in ['na_v3']:
        split_data(mode)
