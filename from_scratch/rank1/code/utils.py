import codecs
import json
from os.path import join
import pickle
import os
import re
import numpy as np
from gensim.models import word2vec
import unicodedata
import pinyin

def check_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

################# Load and Save Data ################

def load_json(rfname):
    with codecs.open(rfname, 'r', encoding='utf-8') as rf:
        return json.load(rf)


def dump_json(obj, wfname, indent=None):
    with codecs.open(wfname, 'w', encoding='utf-8') as wf:
        json.dump(obj, wf, ensure_ascii=False, indent=indent)


def dump_data(obj, wfname):
    with open(wfname, 'wb') as wf:
        pickle.dump(obj, wf)


def load_data(rfname):
    with open(rfname, 'rb') as rf:
        return pickle.load(rf)


################# Random Walk ################

import random


class MetaPathGenerator:
    def __init__(self):
        self.paper_author = dict()
        self.author_paper = dict()
        self.paper_org = dict()
        self.org_paper = dict()
        self.paper_conf = dict()
        self.conf_paper = dict()

    def read_data(self, dirpath):
        temp = set()

        with open(dirpath + "/paper_org.txt", encoding='utf-8') as pafile:
            for line in pafile:
                temp.add(line)
        for line in temp:
            toks = line.strip().split("\t")
            if len(toks) == 2:
                p, a = toks[0], toks[1]
                if p not in self.paper_org:
                    self.paper_org[p] = []
                self.paper_org[p].append(a)
                if a not in self.org_paper:
                    self.org_paper[a] = []
                self.org_paper[a].append(p)
        temp.clear()

        with open(dirpath + "/paper_author.txt", encoding='utf-8') as pafile:
            for line in pafile:
                temp.add(line)
        for line in temp:
            toks = line.strip().split("\t")
            if len(toks) == 2:
                p, a = toks[0], toks[1]
                if p not in self.paper_author:
                    self.paper_author[p] = []
                self.paper_author[p].append(a)
                if a not in self.author_paper:
                    self.author_paper[a] = []
                self.author_paper[a].append(p)
        temp.clear()

        with open(dirpath + "/paper_conf.txt", encoding='utf-8') as pcfile:
            for line in pcfile:
                temp.add(line)
        for line in temp:
            toks = line.strip().split("\t")
            if len(toks) == 2:
                p, a = toks[0], toks[1]
                if p not in self.paper_conf:
                    self.paper_conf[p] = []
                self.paper_conf[p].append(a)
                if a not in self.conf_paper:
                    self.conf_paper[a] = []
                self.conf_paper[a].append(p)
        temp.clear()

        # print("#papers ", len(self.paper_conf))
        # print("#authors", len(self.author_paper))
        # print("#org_words", len(self.org_paper))
        # print("#confs  ", len(self.conf_paper))

    def generate_WMRW(self, outfilename, numwalks, walklength):
        outfile = open(outfilename, 'w')
        for paper0 in self.paper_conf:
            for j in range(0, numwalks):  # wnum walks
                paper = paper0
                outline = ""
                i = 0
                while i < walklength:
                    i = i + 1
                    if paper in self.paper_author:
                        authors = self.paper_author[paper]
                        numa = len(authors)
                        authorid = random.randrange(numa)
                        author = authors[authorid]

                        papers = self.author_paper[author]
                        nump = len(papers)
                        if nump > 1:
                            paperid = random.randrange(nump)
                            paper1 = papers[paperid]
                            while paper1 == paper:
                                paperid = random.randrange(nump)
                                paper1 = papers[paperid]
                            paper = paper1
                            outline += " " + paper
                        # if nump==1 就是自环，没有和其他paper连接
                    if paper in self.paper_org:
                        words = self.paper_org[paper]
                        numw = len(words)
                        wordid = random.randrange(numw)
                        word = words[wordid]

                        papers = self.org_paper[word]
                        nump = len(papers)
                        if nump > 1:
                            paperid = random.randrange(nump)
                            paper1 = papers[paperid]
                            while paper1 == paper:
                                paperid = random.randrange(nump)
                                paper1 = papers[paperid]
                            paper = paper1
                            outline += " " + paper

                outfile.write(outline + "\n")
        outfile.close()

        # print("walks done")


################# Compare Lists ################

def tanimoto(p, q):
    c = [v for v in p if v in q]
    return float(len(c) / (len(p) + len(q) - len(c)))


################# Paper similarity ################

def generate_pair(pubs, outlier):  ##求匹配相似度
    dirpath = 'gene'

    paper_org = {}
    paper_conf = {}
    paper_author = {}
    paper_word = {}

    temp = set()
    with open(dirpath + "/paper_org.txt", encoding='utf-8') as pafile:
        for line in pafile:
            temp.add(line)
    for line in temp:
        toks = line.strip().split("\t")
        if len(toks) == 2:
            p, a = toks[0], toks[1]
            if p not in paper_org:
                paper_org[p] = []
            paper_org[p].append(a)
    temp.clear()

    with open(dirpath + "/paper_conf.txt", encoding='utf-8') as pafile:
        for line in pafile:
            temp.add(line)
    for line in temp:
        toks = line.strip().split("\t")
        if len(toks) == 2:
            p, a = toks[0], toks[1]
            if p not in paper_conf:
                paper_conf[p] = []
            paper_conf[p] = a
    temp.clear()

    with open(dirpath + "/paper_author.txt", encoding='utf-8') as pafile:
        for line in pafile:
            temp.add(line)
    for line in temp:
        toks = line.strip().split("\t")
        if len(toks) == 2:
            p, a = toks[0], toks[1]
            if p not in paper_author:
                paper_author[p] = []
            paper_author[p].append(a)
    temp.clear()

    with open(dirpath + "/paper_word.txt", encoding='utf-8') as pafile:
        for line in pafile:
            temp.add(line)
    for line in temp:
        toks = line.strip().split("\t")
        if len(toks) == 2:
            p, a = toks[0], toks[1]
            if p not in paper_word:
                paper_word[p] = []
            paper_word[p].append(a)
    temp.clear()

    paper_paper = np.zeros((len(pubs), len(pubs)))
    for i, pid in enumerate(pubs):
        if i not in outlier:
            continue
        for j, pjd in enumerate(pubs):
            if j == i:
                continue
            ca = 0
            cv = 0
            co = 0
            ct = 0

            if pid in paper_author and pjd in paper_author:
                ca = len(set(paper_author[pid]) & set(paper_author[pjd])) * 1.5
            if pid in paper_conf and pjd in paper_conf and 'null' not in paper_conf[pid]:
                cv = tanimoto(set(paper_conf[pid]), set(paper_conf[pjd]))
            if pid in paper_org and pjd in paper_org:
                co = tanimoto(set(paper_org[pid]), set(paper_org[pjd]))
            if pid in paper_word and pjd in paper_word:
                ct = len(set(paper_word[pid]) & set(paper_word[pjd])) / 3

            paper_paper[i][j] = ca + cv + co + ct

    return paper_paper


################# Evaluate ################

def pairwise_evaluate(correct_labels, pred_labels):
    TP = 0.0  # Pairs Correctly Predicted To SameAuthor
    TP_FP = 0.0  # Total Pairs Predicted To SameAuthor
    TP_FN = 0.0  # Total Pairs To SameAuthor

    for i in range(len(correct_labels)):
        for j in range(i + 1, len(correct_labels)):
            if correct_labels[i] == correct_labels[j]:
                TP_FN += 1
            if pred_labels[i] == pred_labels[j]:
                TP_FP += 1
            if (correct_labels[i] == correct_labels[j]) and (pred_labels[i] == pred_labels[j]):
                TP += 1

    if TP == 0:
        pairwise_precision = 0
        pairwise_recall = 0
        pairwise_f1 = 0
    else:
        pairwise_precision = TP / TP_FP
        pairwise_recall = TP / TP_FN
        pairwise_f1 = (2 * pairwise_precision * pairwise_recall) / (pairwise_precision + pairwise_recall)
    return pairwise_precision, pairwise_recall, pairwise_f1


names_wrong = [
    # find in train
    (['takahiro', 'toshiyuki', 'takeshi', 'toshiyuki', 'tomohiro', 'takamitsu', 'takahisa', 'takashi',
     'takahiko', 'takayuki'], 'ta(d|k)ashi'),
    (['akimasa', 'akio', 'akito'], 'akira'),
    (['kentarok'], 'kentaro'),
    (['xiaohuatony', 'tonyxiaohua'], 'xiaohua'),
    (['ulrich'], 'ulrike'),
    # find in valid
    (['naoto', 'naomi'], 'naoki'),
    (['junko'], 'junichi'),
    # find in test
    (['isaku'], 'isao')
]


# 检查是否含有中文字符
def is_contains_chinese(strs):
    for _char in strs:
        if '\u4e00' <= _char <= '\u9fa5':
            return True
    return False


def match_name(name, target_name):
    [first_name, last_name] = target_name.split('_')
    first_name = re.sub('-', '', first_name)
    # 中文名转化为拼音形式
    if is_contains_chinese(name):
        name = re.sub('[^ \u4e00-\u9fa5]', '', name).strip()
        name = pinyin.get(name, format='strip')
        # 两个字的人名中间可能有空格
        name = re.sub(' ', '', name)
        target_name = last_name + first_name
        return name == target_name
    else:
        # 处理带声调的拼音
        str_bytes = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore')
        name = str_bytes.decode('ascii')

        name = name.lower()
        name = re.sub('[^a-zA-Z]', ' ', name)
        tokens = name.split()

        if len(tokens) < 2:
            return False
        if len(tokens) == 3:
            # just ignore middle name
            if re.match(tokens[0], first_name) and re.match(tokens[-1], last_name):
                return True
            # ignore tail noise char
            if tokens[-1] == 'a' or tokens[-1] == 'c':
                tokens = tokens[:-1]

        if re.match(tokens[0], last_name):
            # 中文名两个字母缩写(如果只有一个字母原方案能解决)
            if len(tokens) == 2 and len(tokens[1]) == 2:
                if re.match(f'{tokens[1][0]}.*{tokens[1][1]}.*', first_name):
                    return True
            remain = '.*'.join(tokens[1:]) + '.*'

            if re.match(remain, first_name):
                return True
            if len(tokens) == 3 and len(tokens[1]) == 1 and len(tokens[2]) == 1:
                remain_reverse = f'{tokens[2]}.*{tokens[1]}.*'
                if re.match(remain_reverse, first_name):
                    return True
        if re.match(tokens[-1], last_name):
            candidate = ''.join(tokens[:-1])
            find_remain = False
            for (wrong_list, right_one) in names_wrong:
                if candidate in wrong_list:
                    remain = right_one
                    find_remain = True
                    break
            if not find_remain:
                remain = '.*'.join(tokens[:-1]) + '.*'

            if re.match(remain, first_name):
                return True
            if len(tokens) == 3 and len(tokens[0]) == 1 and len(tokens[1]) == 1:
                remain_reverse = f'{tokens[1]}.*{tokens[0]}.*'
                if re.match(remain_reverse, first_name):
                    return True
        return False
