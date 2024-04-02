import json as js
import numpy as np
import pickle as pk
from unidecode import unidecode
import torch
from torch_geometric.data.batch import Batch 
import multiprocessing as mp
import re
import argparse
puncs = '[!“”"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~—～’]+'
stopwords = ['at', 'based', 'in', 'of', 'for', 'on', 'and', 'to', 'an', 'using', 'with',
            'the', 'by', 'we', 'be', 'is', 'are', 'can']
stopwords_extend = ['university', 'univ', 'china', 'department', 'dept', 'laboratory', 'lab',
                    'school', 'al', 'et', 'institute', 'inst', 'college', 'chinese', 'beijing',
                    'journal', 'science', 'international', 'key', 'sciences', 'research',
                    'academy', 'state', 'center']
stopwords_check = ['a', 'was', 'were', 'that', '2', 'key', '1', 'technology', '0', 'sciences', 'as',
                    'from', 'r', '3', 'academy', 'this', 'nanjing', 'shanghai', 'state', 's', 'research',
                    'p', 'results', 'peoples', '4', 'which', '5', 'high', 'materials', 'study', 'control',
                    'method', 'group', 'c', 'between', 'or', 'it', 'than', 'analysis', 'system',  'sci',
                    'two', '6', 'has', 'h', 'after', 'different', 'n', 'national', 'japan', 'have', 'cell',
                    'time', 'zhejiang', 'used', 'data', 'these']
r = '[!“”"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~—～’]+'

def clean_name(name):
    # print(name)
    name = unidecode(name)
    name = name.lower()
    new_name = ""
    for a in name:
        if a.isalpha():
            new_name += a
        else:
            new_name = new_name.strip()
            new_name += " "
    return new_name.strip()
    
def simple_name_match(n1,n2):
    n1_set = set(n1.split())
    n2_set = set(n2.split())

    if(len(n1_set) != len(n2_set)):
        return False
    com_set = n1_set & n2_set
    if(len(com_set) == len(n1_set)):
        return True
    return False
    
def org_venue_features(n1_attr, n2_attr, score_dict, default_value):

    n1_attr_set = set(n1_attr.split())
    n2_attr_set = set(n2_attr.split())

    inter_words = set(n1_attr_set) & set(n2_attr_set)
    scores = 0.0
    
    if(len(inter_words) > 0):
        for inter in inter_words:
            scores += score_dict.get(inter, default_value)
        if len(n1_attr_set) > len(n2_attr_set):
            divide = n1_attr_set
        else:
            divide = n2_attr_set
        

        divide_score = 0.0
        for each in divide:
            divide_score += score_dict.get(each, default_value)
        
        if(divide_score * 1 <= scores):
            scores = scores / divide_score
        else:
            scores = 0.0
    return scores
    
def co_occurance(core_name, paper1, paper2):

    core_name = clean_name(core_name)
    coauthor_weight = 0
    coorg_weight = 0
    covenue_weight = 0
    ori_n1_authors = [clean_name(paper1["authors"][ins_index]["name"]).strip() for ins_index in range(min(len(paper1["authors"]), 50))]
    ori_n2_authors = [clean_name(paper2["authors"][ins_index]["name"]).strip() for ins_index in range(min(len(paper2["authors"]), 50))]
    

    #remove disambiguate author
    for name in ori_n1_authors:
        if simple_name_match(core_name,name):
            ori_n1_authors.remove(name)

    for name in ori_n2_authors:
        if simple_name_match(core_name,name):
            ori_n2_authors.remove(name)

    paper1_authors = ori_n1_authors
    paper2_authors = ori_n2_authors

    whole_authors = min(len(set(paper1_authors)), len(set(paper2_authors)))
    matched = []
    for per_n1 in paper1_authors:

        for per_n2 in paper2_authors:
            if(simple_name_match(per_n1, per_n2)):
                matched.append((per_n1, per_n2))
                coauthor_weight += 1
                break

    coauthor_weight = coauthor_weight/max(whole_authors, 1)
    
    def concat_str(list_strs):
        strs = ' '.join([clean_name(each).strip() for each in list_strs]).strip()
        return strs
    
    n1_org_str = ' '.join([clean_name(concat_str(each.get("orgs", ""))).strip() for each in paper1["authors"][:50] if each.get("orgs", "")!=None]).strip()
    n2_org_str = ' '.join([clean_name(concat_str(each.get("orgs", ""))).strip() for each in paper2["authors"][:50] if each.get("orgs", "")!=None]).strip()
    coorg_weight = org_venue_features(n1_org_str, n2_org_str, {}, 14.37)

    # co_venue
    n1_venue = paper1.get("venue", "")
    n2_venue = paper2.get("venue", "")
    if(n1_venue !=None) and (n2_venue!= None):
        covenue_weight = org_venue_features(clean_name(n1_venue).strip(), clean_name(n2_venue).strip(), {}, 10.42)
    return matched, coauthor_weight, coorg_weight, covenue_weight

def getdata(orcid):
    trainset = True 
    if "normal_data" in author_names[orcid]: # train set
        normal_papers_id = author_names[orcid]["normal_data"]
        outliers_id = author_names[orcid]["outliers"]
        all_pappers_id = normal_papers_id + outliers_id
    elif "papers" in author_names[orcid]: # test set
        all_pappers_id = author_names[orcid]["papers"]
        trainset = False

    total_matrix, total_weight = [], []
    for ii in range(len(all_pappers_id)):
        paper1_id = all_pappers_id[ii]
        for jj in range(len(all_pappers_id)):
            paper2_id = all_pappers_id[jj]
            if paper1_id == paper2_id:
                continue
            
            paper1_inf = papers_info[paper1_id]
            paper2_inf = papers_info[paper2_id]
            
            _, w_coauthor, w_coorg, w_covenue = co_occurance('', paper1_inf, paper2_inf)
            if w_coauthor + w_coorg + w_covenue == 0:
                continue
            if(w_coauthor > 0) or (w_coorg) >0 or (w_covenue)>0:
                total_matrix.append([paper1_id, paper2_id])
                total_weight.append([(w_coauthor + w_coorg + w_covenue) / 3])

    num_papers = len(all_pappers_id)


    #重新编号
    re_num = dict(zip(all_pappers_id, list(range(num_papers))))
    # edge_index
    if trainset:
        set_norm = set(normal_papers_id)
        set_out = set(outliers_id)
        list_edge_y = [0 if (i in set_out) or (j in set_out) else 1 for i,j in total_matrix]
        

    total_matrix = [[re_num[i],re_num[j]] for i,j in total_matrix]
    edge_index = np.array(total_matrix, dtype=np.int64).T
    
    # node labels
    if trainset:
        list_y = len(normal_papers_id) * [1] + len(outliers_id) * [0]
    else: 
        list_y =len(all_pappers_id) *[1]

    #batch
    batch = [0] * num_papers

    # edge weight
    total_weight = [x[0] for x in total_weight]

    if edge_index.size == 0: #if no edge, for rare cases, add default self loop with low weight
        e = [[],[]]
        for i in range(len(all_pappers_id)):
            for j in range(len(all_pappers_id)):
                if i != j:
                    e[0].append(i)
                    e[1].append(j)
        edge_index = e
        total_weight = [0.0001] * len(e[0])
        if trainset:
            list_edge_y =[]
            for i in range(len(edge_index[0])):
                if list_y[edge_index[0][i]] == 1 and list_y[edge_index[1][i]] == 1:
                    list_edge_y.append(1)
                else:
                    list_edge_y.append(0)
    #build data
    data = Batch(edge_index=torch.tensor(edge_index), 
                edge_attr=torch.tensor(total_weight, dtype = torch.float32),
                y=torch.tensor(list_y),
                batch=torch.tensor(batch))
    
    edge_label = torch.tensor(list_edge_y) if trainset else None

    return (data,edge_label,orcid,all_pappers_id)

def build_dataset(path):
    
    keys_list = list(author_names.keys())
    with mp.Pool(processes=10) as pool:  #multiprocessing
        results = pool.map(getdata,keys_list)
    with open(path, "wb") as f:
        pk.dump(results, f)
    print('finish')
    
def norm(data):
    """
    normalize venue, name and org, for build cleaned graph
    {
        id: str
        title: str
        authors:[{
            name
            org
        }]
        "abstract"
        "keywords"
        "venue"
        "year"
    }
    """
    venue = ''
    if data['venue']:
        venue = data["venue"].strip()
        venue = venue.lower()
        venue = re.sub(puncs, ' ', venue)
        venue = re.sub(r'\s{2,}', ' ', venue).strip()
        venue = venue.split(' ')
        venue = [word for word in venue if len(word) > 1]
        venue = [word for word in venue if word not in stopwords]
        venue = [word for word in venue if word not in stopwords_extend]
        venue = [word for word in venue if word not in stopwords_check]
        venue = ' '.join(venue)
    authors = []
    if data['authors']:
        for i in data['authors'][:50]:
            org = i['org']
            if org:
                org = org.strip()
                org = org.lower()
                org = re.sub(puncs, ' ', org)
                org = re.sub(r'\s{2,}', ' ', org).strip()
                org = org.split(' ')[:50]  
                org = [word for word in org if len(word) > 1]
                org = [word for word in org if word not in stopwords]
                org = [word for word in org if word not in stopwords_extend]
                org = " ".join(org)
            authors.append({
                "name": i['name'],
                "org": org
            })
    return {
        'id': data['id'],
        'title': data['title'],
        'venue': venue,
        'year': data['year'],
        'authors': authors,
        'keywords': data['keywords'],
        'abstract': data['abstract']
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default='train_author.json')
    parser.add_argument('--eval_dir', type=str, default='ind_valid_author_ground_truth.json')
    parser.add_argument('--test_dir', type=str, default='test_author.json')
    parser.add_argument('--pub_dir', type=str, default='pid_to_info_all.json')
    args = parser.parse_args()
    
    with open(args.pub_dir, "r", encoding = "utf-8") as f:
        papers_info = js.load(f)
    
    # clean pub, if needed
    # with mp.Pool(processes=10) as pool:
    #     results = pool.map(norm,[value for _,value  in papers_info.items()])
    # papers_info = {k:v for k,v in zip(papers_info.keys(),results)}
    # print('done clean pubs')

    #train
    with open(args.train_dir, "r", encoding="utf-8") as f:
        author_names = js.load(f)
    build_dataset( 'train.pkl')
    print('finish train')
    
    #test
    with open(args.test_dir, "r", encoding="utf-8") as f:
        author_names = js.load(f)

    build_dataset( 'test.pkl')
    print('finish test')
    
    #eval
    with open(args.eval_dir, "r", encoding="utf-8") as f:
        author_names = js.load(f)
    build_dataset( 'eval.pkl')
    print('all done')