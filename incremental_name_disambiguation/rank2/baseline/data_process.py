import json
import numpy as np
import random
from whole_config import configs
# from mongo_online import mongo
# from bson import ObjectId
import torch
from transformers import BertTokenizer
from collections import defaultdict
from operator import itemgetter
import copy
import multiprocessing
from tqdm import tqdm
from character.name_match.tool.interface import FindMain

random.seed(42)
np.random.seed(42)

class raw_data:
    def __init__(self, bertTokenizer):
        self.bertTokenizer = bertTokenizer
        self._load_raw_data()     


    def _load_raw_data(self):
        paper_data_dir = "./datas/"
        # train_data
        with open(paper_data_dir + "train_author_profile.json", 'r',encoding='utf-8') as files:
            self.train_author_profile = json.load(files) 
        with open(paper_data_dir + "train_author_unass.json", 'r',encoding='utf-8') as files:
            self.train_author_test = json.load(files)
        with open(paper_data_dir + "train_proNameAuthorPubs.json", 'r',encoding='utf-8') as files:
            self.train_name2aid2pid = json.load(files)
        
        # test_data
        with open(paper_data_dir + "test_author_profile.json", 'r',encoding='utf-8') as files:
            self.test_author_profile = json.load(files)
        with open(paper_data_dir + "test_author_unass.json", 'r',encoding='utf-8') as files:
            self.test_author_test = json.load(files)
        with open(paper_data_dir + "test_proNameAuthorPubs.json", 'r',encoding='utf-8') as files:
            self.test_name2aid2pid = json.load(files)

        # paper_info
        with open('datas/Task1/train/' + "train_pub.json", 'r',encoding='utf-8') as files:
            self.paper_info = json.load(files)

        self.train_paper2aid2name, self.train_paper_list = self.filter_raw_author_data(self.train_author_profile, self.train_author_test, configs["train_min_papers_each_author"], configs["train_neg_sample"])
        self.test_paper2aid2name, self.test_paper_list = self.filter_raw_author_data(self.test_author_profile, self.test_author_test, configs["train_min_papers_each_author"], configs["test_neg_sample"])
        print("Paper: Train: {} Test: {}".format(len(self.train_paper2aid2name), len(self.test_paper2aid2name)))
        # print("Paper: Train: {}".format(len(self.train_paper2aid2name)))

    def filter_raw_author_data(self, profile, unassign, min_paper, neg_sample):
        # author_data = {}
        # namepaper2pid = {}
        # author2name = {}
        valid_name = []
        paper2aid2name = {}
        paper_list = set()
        for author_name in profile:
            author_papers = profile[author_name]
            if(len(author_papers) <= (neg_sample)):
                continue
            valid_name.append(author_name)
        
        for each in valid_name:
            author2paper = unassign[each]
            for author, papers in author2paper.items():
                for pid in papers:
                    paper2aid2name[pid] = (author, each)
                    paper_list.add(pid)
        return paper2aid2name, list(paper_list)

    def generate_train_data(self, ins_num):
        instance = []
        # training_data = []
        paper2aid2name = self.train_paper2aid2name
        paper_list = self.train_paper_list
        # name2aid2pid = self.train_author_profile
        name2aid2pid = self.train_name2aid2pid
        neg_sample = configs["train_neg_sample"]
        random.shuffle(paper_list)
        sample_pid_id = random.sample(paper_list, ins_num)
        for pid in sample_pid_id:
            aid, name = paper2aid2name[pid]

            coauthors = [tmp["name"] for tmp in self.paper_info[pid]["authors"]]
            res = FindMain(name, coauthors)
            try:
                newpid = pid + '-' + str(res[0][0][1])
            except:
                continue
            else:
                # print(name, aid)
                candidate_authors = list(name2aid2pid[name].keys())
                if (len(candidate_authors) <= neg_sample):
                    print("training error! person num <= :", neg_sample, len(candidate_authors))
                    # exit()
                    continue

                copy_sample_list = copy.deepcopy(candidate_authors)
                copy_sample_list.remove(aid)
                random.shuffle(copy_sample_list)
                assert len(copy_sample_list) == (len(candidate_authors) - 1)

                neg_author_list = random.sample(copy_sample_list, neg_sample)
                assert len(neg_author_list) > 0
                # each_ins = (name, aid, pid, neg_author_list)
                instance.append(((name, aid, newpid, neg_author_list), name2aid2pid))
        print("#Training_instance: ", len(instance))
        # for each in instance:
        #     # pid, neg_author_list = each
        #     paper_pro, pos_pro, neg_pro = self.tokenizer_padding(each, name2aid2pid)            
        #     # tag, tokenizer_ins = self.tokenizer_padding(each_ins, author_dict)
        #     # count += 1
        #     training_data.append((paper_pro, pos_pro, neg_pro))

        return instance

    def atomic_process(self, each_ins):
        each, name2aid2pid = each_ins
        paper_infos, paper_pro, pos_pro, neg_pro = self.tokenizer_padding(each, name2aid2pid)
        return (paper_infos, paper_pro, pos_pro, neg_pro)

    def processed_training_data(self, total_ins):
        training_data = []
        for each_ins in tqdm(total_ins):
            each, name2aid2pid = each_ins
            paper_infos, paper_pro, pos_pro, neg_pro = self.tokenizer_padding(each, name2aid2pid)
            training_data.append((paper_infos, paper_pro, pos_pro, neg_pro))
        return training_data

    def multi_thread_processed_training_data(self, total_ins):
        function = self.atomic_process
        # num_thread = int(multiprocessing.cpu_count()/16)
        num_thread = 4

        # num_thread = 1
        pool = multiprocessing.Pool(num_thread)
        res = pool.map(function, total_ins)
        pool.close()
        pool.join()
        return res 



    def generate_test_data(self, ins_num):
        instance = []
        # training_data = []
        paper2aid2name = self.test_paper2aid2name
        paper_list = self.test_paper_list
        # name2aid2pid = self.test_author_profile
        name2aid2pid = self.test_name2aid2pid
        neg_sample = configs["test_neg_sample"]
        random.shuffle(paper_list)
        sample_pid_id = random.sample(paper_list, ins_num)
        for pid in sample_pid_id:
            aid, name = paper2aid2name[pid]
            coauthors = [tmp["name"] for tmp in self.paper_info[pid]["authors"]]
            res = FindMain(name, coauthors)
            try:
                newpid = pid + '-' + str(res[0][0][1])
            except:
                continue
            else:
                # print(name, aid)
                candidate_authors = list(name2aid2pid[name].keys())
                if (len(candidate_authors) <= neg_sample):
                    print("training error! person num <= :", neg_sample, len(candidate_authors))
                    exit()

                copy_sample_list = copy.deepcopy(candidate_authors)
                copy_sample_list.remove(aid)
                random.shuffle(copy_sample_list)
                assert len(copy_sample_list) == (len(candidate_authors) - 1)

                neg_author_list = random.sample(copy_sample_list, neg_sample)
                # each_ins = (name, aid, pid, neg_author_list)
                instance.append(((name, aid, newpid, neg_author_list), name2aid2pid))
        print("#Testing_instance: ", len(instance))
        # for each in instance:
        #     # pid, neg_author_list = each
        #     paper_pro, pos_pro, neg_pro = self.tokenizer_padding(each, name2aid2pid)            
        #     # tag, tokenizer_ins = self.tokenizer_padding(each_ins, author_dict)
        #     # count += 1
        #     training_data.append((paper_pro, pos_pro, neg_pro))

        return instance



    def tokenizer_padding(self, each, author_dict):
        # ins_pat, author_name, author_id, neg_author_id_list = each
        name, aid, pid, neg_author_list = each
        paper_infos, paper_pro, pos_pro, neg_pro = self.get_author_encoder(name, aid, pid, neg_author_list, author_dict)
        return paper_infos, paper_pro, pos_pro, neg_pro


    def get_author_encoder(self, author_name, author_id, paper_id, neg_author_list, author_dict):
        # paper 
        # str
        paper_str_features = []
        # embedding
        paper_inputs = []
        # positive author
        # str_feature
        # pos_str_features = []
        pos_per_str_features = []
        # embedding

        pos_per_inputs = []

        # negative author
        # str_feature
        # neg_str_features = []
        neg_per_str_features = []
        # embedding
        neg_per_inputs = []


        author_papers = author_dict[author_name][author_id]

        # author_papers_set = set(author_papers)
        author_papers_list = list(set(author_papers))
        sample_pos_papers = random.sample(author_papers_list, min(len(author_papers_list), configs["train_max_papers_each_author"]))
        # (name_info, org_str, venue, keywords_str, semi_padding_input_ids, semi_padding_attention_masks, semantic_padding_input_ids, semantic_padding_attention_masks) = self.paper_encoder(paper_id)
        paper_str, paper_bert = self.paper_encoder(paper_id)
        paper_str_features.append(paper_str)
        paper_inputs.append(paper_bert)

        pos_paper_ids = []
        for pid in sample_pos_papers:
            tmp_str, tmp_inputs = self.paper_encoder(pid)
            pos_per_str_features.append(tmp_str)
            pos_per_inputs.append(tmp_inputs)
            pos_paper_ids.append(author_id + '-' + pid)

        neg_paper_ids = []
        for neg_author_id in neg_author_list:
            neg_author_papers = author_dict[author_name][neg_author_id]
            neg_author_papers_list = list(set(neg_author_papers))
            sample_neg_papers = random.sample(neg_author_papers_list, min(len(neg_author_papers_list), configs["train_max_papers_each_author"]))

            each_neg_per_str_features = []
            each_neg_per_inputs = []

            for pid in sample_neg_papers:
                tmp_str, tmp_inputs = self.paper_encoder(pid)
                each_neg_per_str_features.append(tmp_str)
                each_neg_per_inputs.append(tmp_inputs)
                neg_paper_ids.append(neg_author_id + '-' + pid)

            neg_per_str_features.append(each_neg_per_str_features)
            neg_per_inputs.append(each_neg_per_inputs)



        return (author_name, author_id, paper_id, pos_paper_ids, neg_paper_ids, neg_author_list), \
        (paper_str_features, paper_inputs),\
        (pos_per_str_features, pos_per_inputs),\
        (neg_per_str_features, neg_per_inputs)

  
    def get_res_abs(self, papers_id):
        split_info = papers_id.split('-')
        pid = str(split_info[0])
        author_index = int(split_info[1])
        papers_attr = self.paper_info[pid]
        # papers_attr = self.paper_info[papers_id]
        # print(papers_attr)
        name_info = set()
        org_str = ""
        keywords_info = set()
        try:
            title = papers_attr["title"].strip().lower()
        except:
            title = ""

        try:
            venue = papers_attr["venue"].strip().lower()
        except:
            venue = ""

        # try:
        #     year = int(papers_attr["year"])
        # except:
        #     year = 0

        try:
            abstract = papers_attr["abstract"]
        except:
            abstract = ""
        
        try:
            keywords = papers_attr["keywords"]
        except:
            keywords = []

        for ins in keywords:
            keywords_info.add(ins.strip().lower())

        paper_authors = papers_attr["authors"]
        for ins_author_index in range(len(paper_authors)):
            ins_author = paper_authors[ins_author_index]
            if(ins_author_index == author_index):
                try:
                    orgnizations =ins_author["org"].strip().lower()
                except:
                    orgnizations = ""

                if(orgnizations.strip().lower() != ""):
                    org_str = orgnizations
            else:


                try:
                    name = ins_author["name"].strip().lower()
                except:
                    name = ""
                if(name != ""):
                    name_info.add(name)

        # name_str = " ".join(name_info).strip()

        # org_str = " ".join(org_info).strip()
        keywords_info = list(keywords_info)
        keywords_str = " ".join(keywords_info).strip()


        semi_str = org_str + venue
        semantic_str = title + " " + keywords_str

        return (name_info, org_str, venue, keywords_str, keywords_info, title, abstract, semi_str, semantic_str)

    def paper_encoder(self, paper_id):
        # papers_attr = self.paper_info[paper_id]
        paper_attr = self.get_res_abs(paper_id)
        name_info, org_str, venue, keywords_str, keywords_info, title, abs_info, semi_str, semantic_str = paper_attr

        # total_str = semi_str.strip() + ' ' + semantic_str.strip()

        bert_token = self.bertTokenizer.build_inputs(title=title, abstract=abs_info, venue=venue, authors=name_info,
                                                     concepts=keywords_info, affiliations=org_str)
        return (name_info, org_str, venue, keywords_str, title), (bert_token,)
