# -*- coding: utf-8 -*-
import json
import pandas as pd
import numpy as np

mode = 'testa'
subpath = "./work/submit/online/tree/LightGBM"
lgb_sub = pd.read_csv(subpath+"/lgb_{}.csv".format(mode))
subpath = "./work/submit/online/tree/XGBoost"
xgb_sub = pd.read_csv(subpath+"/xgb_{}.csv".format(mode))

all_sub = lgb_sub.copy()
all_sub['pred'] = 0.6*lgb_sub['pred'].values+0.4*xgb_sub['pred'].values
allsub=all_sub.drop(['idx'],axis=1,inplace=True)
all_sub = all_sub[all_sub['author_id']!='0'].reset_index(drop=True)
all_sub = all_sub[all_sub['pred']>=0.2].reset_index(drop=True)
all_sub = all_sub.sort_values(by=['paper_id','pred'],ascending=False).reset_index(drop=True)

print(all_sub)

def gen_dict(df, label):
    df = df[['paper_id', 'author_id', label]]
    # print(df)
    # res = df.groupby(['paper_id'])[label].apply(np.argmax).reset_index()
    # print(res)
    # res.columns = ['paper_id', 'index']
    # idx_name = df[['author_id']].reset_index()
    # print(res[['paper_id', 'author_id']])
    # res = res.merge(idx_name, 'left', 'index')
    from collections import defaultdict
    res_dict = defaultdict(list)
    # print(res[['paper_id', 'author_id']])
    for pid, aid in df[['paper_id', 'author_id']].values:
        res_dict[aid].append(pid)
    return res_dict

result_dict = gen_dict(all_sub, 'pred')    
with open(subpath+"/lgb_%s.json"%(mode), 'w') as file:
    file.write(json.dumps(result_dict))
