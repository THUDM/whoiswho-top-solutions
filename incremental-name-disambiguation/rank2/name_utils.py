#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import re
import json
import unicodedata
import pinyinsplit
import numpy as np
import _pickle as pickle
from tqdm import tqdm
from collections import Counter
from pypinyin import lazy_pinyin, Style
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

alphabet2int = dict(zip('abcdefghijklmnopqrstuvwxyz', list(range(26))))
pys = pinyinsplit.PinyinSplit()
english_stopwords = set(stopwords.words('english'))
resource_dir = r'./resource'


# ================================ 作者名字处理 ================================
def shave_marks(txt):
    """
    去掉全部变音符号， 如Sören处理成Soren
    refer to https://hellowac.github.io/programing%20teach/2017/05/10/fluentpython04.html
    """
    txt = txt.split()
    for i in range(len(txt)):
        if re.match(r'[jqly]ü', txt[i]):
            txt[i] = txt[i][:-1] + 'v'

    txt = ' '.join(txt)
    norm_txt = unicodedata.normalize('NFD', txt)  # 分解成基字符和组合记号。
    shaved = ''.join(c for c in norm_txt
                     if not unicodedata.combining(c))  # 过滤掉所有组合记号。
    return unicodedata.normalize('NFC', shaved)  # 重组所有字符。


def process_name(name):
    # name = unicodedata.normalize('NFKD', name)
    name = name.lower().strip()  # 解决大小写问题
    name = re.sub('[-_]', ' ', name)  # 消除连接符的影响
    name = shave_marks(name)  # 消除变音符号、无效编码等的影响
    name = re.sub(r'[^a-z\u4e00-\u9fa5\s\.]+', '', name)  # 消除除空格外其他字符的影响，包括日韩文
    if re.fullmatch(r'[\u4e00-\u9fa5\s]+', name):  # 若只包含中文，则转换为拼音
        name = ' '.join(lazy_pinyin(name, style=Style.NORMAL))
    else:  # 若中英混杂，则只保留英文
        name = re.sub(r'[\u4e00-\u9fa5]+', '', name)

    name = ' '.join(re.split(r'[\.\s]', name)).strip()
    return name


def get_abbr_name(raw_name):
    # 处理名字缩写情况，如"guanhua_du"，返回
    # 拼音缩写"g h du": 只保留名字拼音首字母，保留姓
    # 最简缩写"g du"：只保留名字第一个字拼音首字母，保留姓
    # 镜像拼音缩写"h d guan": 考虑姓在前面的情况，保留名字拼音首字母，保留姓
    # 镜像最简缩写"h guan"：只保留名字第一个字拼音首字母，保留姓
    name = re.split(r'\s+', raw_name)

    split_names = []
    for _name in name:
        try:
            split_name = pys.split(_name)[0]
        except:
            split_name = [_name]
        split_names.extend(split_name)

    abbr_name = [x[0] for x in split_names[:-1]] + [split_names[-1]]
    simp_name = abbr_name[0] + ' ' + abbr_name[-1]
    abbr_name = ' '.join(abbr_name)

    mirror_abbr_name = [x[0] for x in split_names[1:]] + [split_names[0]]
    mirror_simp_name = mirror_abbr_name[0] + ' ' + mirror_abbr_name[-1]
    mirror_abbr_name = ' '.join(mirror_abbr_name)
    return abbr_name, simp_name, mirror_abbr_name, mirror_simp_name


def get_int_for_name(name):
    # 将名字按字母编码26位数字，方便比较
    name = re.sub(r'\s+', '', name)
    name = re.sub(r'\W+', '', name)
    name_int = [0] * 26
    for a in name:
        name_int[alphabet2int[a]] += 1
    # name_int = int(''.join(list(map(str, name_int))))  # 变成int类型会额外占用时间，而且这种长度下直接字符串比较和数字比较时间几乎相等
    name_int = ''.join(list(map(str, name_int)))
    return name_int


def add_name(name, standard_name, name_int, author_ids, name2int, int2author_id):
    name2int[name] = name_int
    if name_int not in int2author_id:
        int2author_id[name_int] = {}

    if name not in int2author_id[name_int]:
        int2author_id[name_int][name] = {}

    if standard_name not in int2author_id[name_int][name]:
        int2author_id[name_int][name][standard_name] = []

    for author_id in author_ids:
        int2author_id[name_int][name][standard_name].append(author_id)


def get_train_name_encoder(train_name_infos):
    # 训练集作者名字编码
    full_name2int, full_int2author_id = {}, {}
    abbr_name2int, abbr_int2author_id = {}, {}
    simp_name2int, simp_int2author_id = {}, {}
    mirror_abbr_name2int, mirror_abbr_int2author_id = {}, {}
    mirror_simp_name2int, mirror_simp_int2author_id = {}, {}
    for name, name_infos in train_name_infos.items():
        author_ids = list(name_infos.keys())
        name = process_name(name)
        name_int = get_int_for_name(name)
        add_name(name, name, name_int, author_ids, full_name2int, full_int2author_id)

        abbr_name, simp_name, mirror_abbr_name, mirror_simp_name = get_refer_abbr_name(name)
        abbr_name_int = get_int_for_name(abbr_name)
        simp_name_int = get_int_for_name(simp_name)
        mirror_abbr_name_int = get_int_for_name(mirror_abbr_name)
        mirror_simp_name_int = get_int_for_name(mirror_simp_name)
        add_name(abbr_name, name, abbr_name_int, author_ids, abbr_name2int, abbr_int2author_id)
        add_name(simp_name, name, simp_name_int, author_ids, simp_name2int, simp_int2author_id)
        add_name(mirror_abbr_name, name, mirror_abbr_name_int, author_ids, mirror_abbr_name2int, mirror_abbr_int2author_id)
        add_name(mirror_simp_name, name, mirror_simp_name_int, author_ids, mirror_simp_name2int, mirror_simp_int2author_id)

    return full_name2int, full_int2author_id, abbr_name2int, abbr_int2author_id, simp_name2int, simp_int2author_id, \
           mirror_abbr_name2int, mirror_abbr_int2author_id, mirror_simp_name2int, mirror_simp_int2author_id


def get_train_name_papers(train_name_infos):
    # 训练集作者名字和论文对应
    full_name_papers = {}
    abbr_name_papers = {}
    simp_name_papers = {}
    mirror_abbr_name_papers = {}
    mirror_simp_name_papers = {}

    for name, name_infos in train_name_infos.items():
        name = process_name(name)
        abbr_name, simp_name, mirror_abbr_name, mirror_simp_name = get_abbr_name(name)

        if name not in full_name_papers:
            full_name_papers[name] = set()
        if abbr_name not in abbr_name_papers:
            abbr_name_papers[abbr_name] = set()
        if simp_name not in simp_name_papers:
            simp_name_papers[simp_name] = set()
        if mirror_abbr_name not in mirror_abbr_name_papers:
            mirror_abbr_name_papers[mirror_abbr_name] = set()
        if mirror_simp_name not in mirror_simp_name_papers:
            mirror_simp_name_papers[mirror_simp_name] = set()

        for author_id, paper_list in name_infos.items():
            full_name_papers[name] |= set(paper_list)
            abbr_name_papers[abbr_name] |= set(paper_list)
            simp_name_papers[simp_name] |= set(paper_list)
            mirror_abbr_name_papers[mirror_abbr_name] |= set(paper_list)
            mirror_simp_name_papers[mirror_simp_name] |= set(paper_list)

    return full_name_papers, abbr_name_papers, simp_name_papers, mirror_abbr_name_papers, mirror_simp_name_papers


def get_whole_name_encoder(whole_author_infos):
    # whole集作者名字编码
    full_name2int, full_int2author_id = {}, {}
    abbr_name2int, abbr_int2author_id = {}, {}
    simp_name2int, simp_int2author_id = {}, {}
    mirror_abbr_name2int, mirror_abbr_int2author_id = {}, {}
    mirror_simp_name2int, mirror_simp_int2author_id = {}, {}
    for author_id, author_infos in tqdm(whole_author_infos.items(), desc='get whole name encode'):
        name = process_name(author_infos['name'])
        name_int = get_int_for_name(name)
        add_name(name, name, name_int, [author_id], full_name2int, full_int2author_id)

        abbr_name, simp_name, mirror_abbr_name, mirror_simp_name = get_abbr_name(name)
        abbr_name_int = get_int_for_name(abbr_name)
        simp_name_int = get_int_for_name(simp_name)
        mirror_abbr_name_int = get_int_for_name(mirror_abbr_name)
        mirror_simp_name_int = get_int_for_name(mirror_simp_name)
        add_name(abbr_name, name, abbr_name_int, [author_id], abbr_name2int, abbr_int2author_id)
        add_name(simp_name, name, simp_name_int, [author_id], simp_name2int, simp_int2author_id)
        add_name(mirror_abbr_name, name, mirror_abbr_name_int, [author_id], mirror_abbr_name2int, mirror_abbr_int2author_id)
        add_name(mirror_simp_name, name, mirror_simp_name_int, [author_id], mirror_simp_name2int, mirror_simp_int2author_id)

    return full_name2int, full_int2author_id, abbr_name2int, abbr_int2author_id, simp_name2int, simp_int2author_id, \
           mirror_abbr_name2int, mirror_abbr_int2author_id, mirror_simp_name2int, mirror_simp_int2author_id


def get_whole_name_papers(whole_author_infos):
    # whole集作者名字和论文对应
    full_name_papers = {}
    abbr_name_papers = {}
    simp_name_papers = {}
    mirror_abbr_name_papers = {}
    mirror_simp_name_papers = {}

    for author_id, author_infos in whole_author_infos.items():
        name = process_name(author_infos['name'])
        abbr_name, simp_name, mirror_abbr_name, mirror_simp_name = get_refer_abbr_name(name)

        if name not in full_name_papers:
            full_name_papers[name] = set()
        if abbr_name not in abbr_name_papers:
            abbr_name_papers[abbr_name] = set()
        if simp_name not in simp_name_papers:
            simp_name_papers[simp_name] = set()
        if mirror_abbr_name not in mirror_abbr_name_papers:
            mirror_abbr_name_papers[mirror_abbr_name] = set()
        if mirror_simp_name not in mirror_simp_name_papers:
            mirror_simp_name_papers[mirror_simp_name] = set()

        paper_list = author_infos['pubs']
        full_name_papers[name] |= set(paper_list)
        abbr_name_papers[abbr_name] |= set(paper_list)
        simp_name_papers[simp_name] |= set(paper_list)
        mirror_abbr_name_papers[mirror_abbr_name] |= set(paper_list)
        mirror_simp_name_papers[mirror_simp_name] |= set(paper_list)

    return full_name_papers, abbr_name_papers, simp_name_papers, mirror_abbr_name_papers, mirror_simp_name_papers


def get_all_int_for_name(name):
    # 获取全名、简名、最简名、镜像简名、镜像最简名的编码
    name = process_name(name)
    abbr_name, simp_name, mirror_abbr_name, mirror_simp_name = get_abbr_name(name)

    full_name_int, abbr_name_int, simp_name_int, mirror_abbr_name_int, mirror_simp_name_int = \
        list(map(get_int_for_name, [name, abbr_name, simp_name, mirror_abbr_name, mirror_simp_name]))
    return full_name_int, abbr_name_int, simp_name_int, mirror_abbr_name_int, mirror_simp_name_int


def is_same_name(name1, name2):
    # 判断两名字是否相等
    name1, name2 = process_name(name1), process_name(name2)
    abbr_name2, simp_name2, mirror_abbr_name2, mirror_simp_name2 = get_abbr_name(name2)

    name_int1, full_name_int2, abbr_name_int2, simp_name_int2, mirror_abbr_name_int2, mirror_simp_name_int2 = \
        list(map(get_int_for_name, [name1, name2, abbr_name2, simp_name2, mirror_abbr_name2, mirror_simp_name2]))

    if full_name_int2 == name_int1 or abbr_name_int2 == name_int1 or simp_name_int2 == name_int1 or \
            mirror_abbr_name_int2 == name_int1 or mirror_simp_name_int2 == name_int1:
        return True
    else:
        return False


# ================================ 作者机构处理 ================================
def clean_orgs(orgs):
    orgs = re.sub(r'\W', ' ', orgs.lower())
    orgs = re.sub(r'[0-9]+', '', orgs)
    orgs = ' '.join([w for w in orgs.split() if w not in english_stopwords]).strip()
    return orgs


def get_train_author_orgs(train_name_infos, train_paper_infos, data_types):
    result_path = r'resource/{}_author_orgs.pkl'.format(data_types)
    if os.path.exists(result_path):
        author_orgs = pickle.load(open(result_path, 'rb'))
    else:
        author_orgs = {}
        for name, name_infos in tqdm(train_name_infos.items()):
            ref_full_name_int, ref_abbr_name_int, ref_simp_name_int, ref_mirror_abbr_name_int, \
                ref_mirror_simp_name_int = get_all_int_for_name(name)

            for author_id, paper_list in name_infos.items():
                if author_id not in author_orgs:
                    author_orgs[author_id] = []

                for paper_id in paper_list:
                    if paper_id not in train_paper_infos:
                        continue
                    paper = train_paper_infos[paper_id]
                    for paper_author in paper['authors']:
                        cur_full_name_int, cur_abbr_name_int, cur_simp_name_int, cur_mirror_abbr_name_int, \
                            cur_mirror_simp_name_int = get_all_int_for_name(paper_author['name'])

                        cur_org = clean_orgs(paper_author['org'])
                        if cur_org == '':
                            continue

                        if ref_full_name_int == cur_full_name_int:
                            author_orgs[author_id].append(cur_org)
                            author_orgs[author_id].append(cur_org)  # 全名匹对正确的单位名称给予2倍权重
                            continue  # 若全名能匹对，则不需要进行缩写匹对，下同

                        if ref_abbr_name_int == cur_abbr_name_int or ref_abbr_name_int == cur_mirror_abbr_name_int:
                            author_orgs[author_id].append(cur_org)
                            continue

                        if ref_mirror_abbr_name_int == cur_mirror_abbr_name_int or ref_mirror_abbr_name_int == cur_abbr_name_int:
                            author_orgs[author_id].append(cur_org)
                            continue

                        if ref_simp_name_int == cur_simp_name_int or ref_simp_name_int == cur_mirror_simp_name_int:
                            author_orgs[author_id].append(cur_org)
                            continue

                        if ref_mirror_simp_name_int == cur_mirror_simp_name_int or ref_mirror_simp_name_int == cur_simp_name_int:
                            author_orgs[author_id].append(cur_org)
                            continue
        pickle.dump(author_orgs, open(result_path, 'wb'))
    return author_orgs


def get_whole_author_orgs(whole_author_infos, whole_paper_infos):
    result_path = r'resource/whole_author_orgs.pkl'
    if os.path.exists(result_path):
        author_orgs = pickle.load(open(result_path, 'rb'))
    else:
        author_orgs = {}
        for author_id, author_infos in tqdm(whole_author_infos.items(), desc='get whole author orgs'):
            if author_id not in author_orgs:
                author_orgs[author_id] = []

            ref_full_name_int, ref_abbr_name_int, ref_simp_name_int, ref_mirror_abbr_name_int, \
                ref_mirror_simp_name_int = get_all_int_for_name(author_infos['name'])

            for paper_id in author_infos['pubs']:
                paper = whole_paper_infos[paper_id]
                for paper_author in paper['authors']:
                    cur_full_name_int, cur_abbr_name_int, cur_simp_name_int, cur_mirror_abbr_name_int, \
                        cur_mirror_simp_name_int = get_all_int_for_name(paper_author['name'])

                    cur_org = clean_orgs(paper_author['org'])
                    if cur_org == '':
                        continue

                    if ref_full_name_int == cur_full_name_int:
                        author_orgs[author_id].append((cur_org, 'full'))
                        continue

                    if ref_abbr_name_int == cur_abbr_name_int or ref_abbr_name_int == cur_mirror_abbr_name_int:
                        author_orgs[author_id].append((cur_org, 'abbr'))
                        continue

                    if ref_mirror_abbr_name_int == cur_mirror_abbr_name_int or ref_mirror_abbr_name_int == cur_abbr_name_int:
                        author_orgs[author_id].append((cur_org, 'abbr'))
                        continue

                    if ref_simp_name_int == cur_simp_name_int or ref_simp_name_int == cur_mirror_simp_name_int:
                        author_orgs[author_id].append((cur_org, 'simp'))
                        continue

                    if ref_mirror_simp_name_int == cur_mirror_simp_name_int or ref_mirror_simp_name_int == cur_simp_name_int:
                        author_orgs[author_id].append((cur_org, 'simp'))
                        continue
        pickle.dump(author_orgs, open(result_path, 'wb'))
    return author_orgs


def get_author_orgs_tfidf(author_orgs, data_types):
    # 用tfidf对每个作者计算单位的关键词
    result_path = r'resource/{}_author_orgs_tfidf.pkl'.format(data_types)
    if os.path.exists(result_path):
        author_org_tfidf = pickle.load(open(result_path, 'rb'))
    else:
        author_org_tfidf = {}
        for author, orgs in tqdm(author_orgs.items(), desc='get author org keys'):
            vectorizer = TfidfVectorizer()
            try:
                X = vectorizer.fit_transform(orgs)
            except:
                author_org_tfidf[author] = set()
            else:
                weight = np.sum(X.toarray(), axis=0)
                word_score = list(zip(vectorizer.get_feature_names(), weight))
                word_score = sorted(word_score, key=lambda x: x[1], reverse=True)

                top_words = [w for w, s in word_score[:10]]
                author_org_tfidf[author] = set(top_words)

        pickle.dump(author_org_tfidf, open(result_path, 'wb'))

    return author_org_tfidf


# ================================ 作者关键词处理 ================================
def get_train_author_keywords(train_name_infos, train_paper_infos, data_types):
    result_path = r'resource/{}_author_keywords.pkl'.format(data_types)
    if os.path.exists(result_path):
        author_keywords = pickle.load(open(result_path, 'rb'))
    else:
        author_keywords = {}
        for name, name_infos in tqdm(train_name_infos.items()):
            for author_id, paper_list in name_infos.items():
                if author_id not in author_keywords:
                    author_keywords[author_id] = []

                for paper_id in paper_list:
                    if paper_id not in train_paper_infos:
                        continue
                    keywords = train_paper_infos[paper_id]['keywords']
                    if (isinstance(keywords, str) and keywords.strip() == '') or \
                            (isinstance(keywords, list) and len(keywords) == 0) or \
                            (isinstance(keywords, list) and len(keywords) == 1 and
                             (keywords[0] == 'null' or keywords[0] == '')):
                        continue

                    author_keywords[author_id].extend(keywords)

        pickle.dump(author_keywords, open(result_path, 'wb'))
    return author_keywords


def get_whole_author_keywords(whole_author_infos, whole_paper_infos):
    result_path = r'resource/whole_author_keywords.pkl'
    if os.path.exists(result_path):
        author_keywords = pickle.load(open(result_path, 'rb'))
    else:
        author_keywords = {}
        for author_id, author_infos in tqdm(whole_author_infos.items(), desc='get whole author keywords'):
            if author_id not in author_keywords:
                author_keywords[author_id] = []

            for paper_id in author_infos['pubs']:
                keywords = whole_paper_infos[paper_id]['keywords']
                if (isinstance(keywords, str) and keywords.strip() == '') or \
                        (isinstance(keywords, list) and len(keywords) == 0) or \
                        (isinstance(keywords, list) and len(keywords) == 1 and
                         (keywords[0] == 'null' or keywords[0] == '')):
                    continue

                author_keywords[author_id].extend(keywords)
        pickle.dump(author_keywords, open(result_path, 'wb'))

    return author_keywords


def get_author_keywords_tfidf(author_keywords, data_types):
    # 用tfidf对每个作者计算关键词的top关键词
    result_path = r'resource/{}_author_keywords_tfidf.pkl'.format(data_types)
    if os.path.exists(result_path):
        author_keywords_tfidf = pickle.load(open(result_path, 'rb'))
    else:
        author_keywords_tfidf = {}
        for author, keywords in tqdm(author_keywords.items(), desc='get author keywords tfidf'):
            vectorizer = TfidfVectorizer()
            try:
                X = vectorizer.fit_transform(keywords)
            except:
                author_keywords_tfidf[author] = set()
            else:
                weight = np.sum(X.toarray(), axis=0)
                word_score = list(zip(vectorizer.get_feature_names(), weight))
                word_score = sorted(word_score, key=lambda x: x[1], reverse=True)

                top_words = [w for w, s in word_score[:20]]
                author_keywords_tfidf[author] = set(top_words)

        pickle.dump(author_keywords_tfidf, open(result_path, 'wb'))

    return author_keywords_tfidf


# ================================ 共同作者、共同机构处理 ================================
def get_coauthor(data_types, author_infos, paper_infos):
    # 获取共同作者、共同作者的单位
    if os.path.exists(os.path.join(resource_dir, '{}_author_coauthors.json'.format(data_types))):
        author_coauthors = json.load(open(os.path.join(resource_dir, '{}_author_coauthors.json'.format(data_types))))
        author_coorgs = json.load(open(os.path.join(resource_dir, '{}_author_coorgs.json'.format(data_types))))
    else:
        author_id_name_papers = {}
        if data_types == 'whole':
            for author_id, author_infos in author_infos.items():
                name = process_name(author_infos['name'])
                author_id_name_papers[author_id] = {name: author_infos['pubs']}
        else:
            for name, name_infos in author_infos.items():
                name = process_name(name)
                for author_id, paper_list in name_infos.items():
                    author_id_name_papers[author_id] = {name: paper_list}

        author_coauthors = {}
        author_coorgs = {}
        for author_id, author_infos in tqdm(author_id_name_papers.items()):
            author_coauthors[author_id], author_coorgs[author_id] = Counter(), Counter()
            for name, paper_list in author_infos.items():
                full_name_int, abbr_name_int, simp_name_int, mirror_abbr_name_int, mirror_simp_name_int = \
                    get_all_int_for_name(name)

                for paper in paper_list:
                    if paper not in paper_infos:
                        continue
                    paper = paper_infos[paper]
                    for co_author in paper['authors']:
                        co_name, co_org = co_author['name'], co_author['org']
                        co_name = process_name(co_name)
                        co_org = clean_orgs(co_org)

                        co_name_int = get_int_for_name(co_name)
                        if co_name_int == full_name_int or co_name_int == abbr_name_int or co_name_int == simp_name_int or \
                                co_name_int == mirror_abbr_name_int or co_name_int == mirror_simp_name_int:
                            continue

                        author_coauthors[author_id][co_name] += 1
                        if co_org != '':
                            author_coorgs[author_id][co_org] += 1

        for author_id in author_coauthors:
            co_authors = author_coauthors[author_id]
            co_authors = co_authors.most_common()
            author_coauthors[author_id] = co_authors

        for author_id in author_coorgs:
            co_orgs = author_coorgs[author_id]
            co_orgs = co_orgs.most_common()
            author_coorgs[author_id] = co_orgs

        json.dump(author_coauthors, open(os.path.join(resource_dir, '{}_author_coauthors.json'.format(data_types)), 'w'))
        json.dump(author_coorgs, open(os.path.join(resource_dir, '{}_author_coorgs.json'.format(data_types)), 'w'))

    return author_coauthors, author_coorgs


if __name__ == '__main__':
    data_dir = r'./datas/Task1'
    with open(os.path.join(data_dir, r'train/train_author.json')) as f:
        train_name_infos = json.load(f)

    with open(os.path.join(data_dir, r'train/train_pub.json')) as f:
        train_paper_infos = json.load(f)

    with open(os.path.join(data_dir, r'cna-valid/whole_author_profiles.json')) as f:
        whole_author_infos = json.load(f)

    with open(os.path.join(data_dir, r'cna-valid/whole_author_profiles_pub.json')) as f:
        whole_paper_infos = json.load(f)

    train_author_orgs = get_train_author_orgs(train_name_infos, train_paper_infos, 'train')
    train_author_orgs_tfidf = get_author_orgs_tfidf(train_author_orgs, 'train')

    whole_author_orgs = get_whole_author_orgs(whole_author_infos, whole_paper_infos)
    whole_author_orgs_tfidf = get_author_orgs_tfidf(whole_author_orgs, 'whole')

    get_coauthor('train', train_name_infos, train_paper_infos)
    get_coauthor('whole', whole_author_infos, whole_paper_infos)

    train_author_keywords = get_train_author_keywords(train_name_infos, train_paper_infos, 'train')
    train_author_keywords_tfidf = get_author_keywords_tfidf(train_author_keywords, 'train')

    whole_author_keywords = get_whole_author_keywords(whole_author_infos, whole_paper_infos)
    whole_author_keywords_tfidf = get_author_keywords_tfidf(whole_author_keywords, 'whole')
