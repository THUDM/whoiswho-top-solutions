import numpy as np
import json
from operator import itemgetter
from pyjarowinkler import distance
import re
import random


def print_info(author_dict):
    name_count = 0
    author_count = 0
    paper_count = 0

    for author_name, authors in author_dict.items():
        name_count += 1
        for author_id, papers in authors.items():
            author_count += 1
            # for pid in papers:
            paper_count += len(papers)

    print("name: {} author: {} paper: {}".format(name_count, author_count, paper_count))


with open('./datas/Task1/train/train_author.json', 'r',encoding='utf-8') as files:
    author_pub = json.load(files)
print_info(author_pub)

tmp_keys = list(author_pub.keys())
random.shuffle(tmp_keys)

train_author, test_author = tmp_keys[:int(len(tmp_keys) * 0.8)], tmp_keys[int(len(tmp_keys) * 0.8):]
train_author_pub = {_k: _v for _k, _v in author_pub.items() if _k in train_author}
test_author_pub = {_k: _v for _k, _v in author_pub.items() if _k in test_author}

with open('./datas/Task1/train/train_pub.json', 'r',encoding='utf-8') as files:
    pub_dict = json.load(files)


def save_my_train(author_pub, data_types):
    new_author_profile = {}
    new_author_unassigned = {}
    new_pub_profile = {}
    new_pub_unass = {}

    splitRatio = 0.2
    for author_name, authors in author_pub.items():
        # valid_count = 0
        tmp_profile = {}
        tmp_unass = {}
        for a_id, papers in authors.items():
            paper_count = len(papers)
            unassgn_num = int(paper_count * splitRatio)
            papers_year = []
            for each in papers:
                try:
                    year = int(pub_dict[each]["year"])
                except:
                    year = 0
                papers_year.append((each, year))
            sorted_papers = sorted(papers_year, key=lambda e: e[1])
            profiles = sorted_papers[:(paper_count - unassgn_num)]
            unassgned = sorted_papers[(paper_count - unassgn_num):]

            profiles_id = [each[0] for each in profiles]
            unass_id = [each[0] for each in unassgned]
            tmp_profile[a_id] = []
            tmp_unass[a_id] = []
            for each_id in profiles_id:
                tmp_profile[a_id].append(each_id)
                new_pub_profile[each_id] = pub_dict[each_id]

            for each_id in unass_id:
                tmp_unass[a_id].append(each_id)
                new_pub_unass[each_id] = pub_dict[each_id]

        new_author_profile[author_name] = tmp_profile
        new_author_unassigned[author_name] = tmp_unass

    print("profile: ")
    print_info(new_author_profile)
    print(len(new_pub_profile))
    print("unass: ")
    print_info(new_author_unassigned)
    print(len(new_pub_unass))

    json.dump(new_author_profile, open(r'./datas/{}_author_profile.json'.format(data_types), 'w'))
    json.dump(new_author_unassigned, open(r'./datas/{}_author_unass.json'.format(data_types), 'w'))


save_my_train(train_author_pub, 'train')
save_my_train(test_author_pub, 'test')
