#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import time
import json
import numpy as np
import _pickle as pickle
import multiprocessing as mp
from tqdm import tqdm

zero_embeds = np.zeros((300,), dtype=np.float32)


def compute_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        end_time = time.time()
        print('cost {:.5f}s'.format(end_time - start_time))
    return wrapper


def get_text_embeds(text):
    text = text.lower().strip().split()
    embeds = []
    for word in text:
        if word in glove_vectors:
            embeds.append(glove_vectors[word])
        else:
            embeds.append(zero_embeds)

    return np.mean(embeds, axis=0), np.max(embeds, axis=0)


def get_keywords_embeds(keywords_list):
    mean_embeds, max_embeds = [], []
    for keywords in keywords_list:
        keywords = keywords.strip()
        if keywords == '':
            cur_mean_embeds, cur_max_embeds = zero_embeds, zero_embeds
        else:
            keywords = keywords.split()
            cur_embeds = []
            for kw in keywords:
                kw = kw.lower()
                if kw in glove_vectors:
                    cur_embeds.append(glove_vectors[kw])
                else:
                    cur_embeds.append(zero_embeds)

            cur_mean_embeds = np.mean(cur_embeds, axis=0)
            cur_max_embeds = np.max(cur_embeds, axis=0)

        mean_embeds.append(cur_mean_embeds)
        max_embeds.append(cur_max_embeds)

    mean_embeds = np.mean(mean_embeds, axis=0)
    max_embeds = np.max(max_embeds, axis=0)
    return mean_embeds, max_embeds


def get_paper_vector(paper_datas):
    paper2vector_title_mean, paper2vector_title_max = {}, {}
    paper2vector_abstract_mean, paper2vector_abstract_max = {}, {}
    paper2vector_keywords_completion_mean, paper2vector_keywords_completion_max = {}, {}
    paper2vector_keywords_partial_mean, paper2vector_keywords_partial_max = {}, {}
    for paper_id, paper_infos in tqdm(paper_datas.items()):
        title = paper_infos['title'].strip()
        abstract = paper_infos['abstract'].strip()
        keywords = paper_infos['keywords']

        title_mean, title_max = get_text_embeds(title)
        paper2vector_title_mean[paper_id] = title_mean
        paper2vector_title_max[paper_id] = title_max

        if abstract == '':
            abstract_mean, abstract_max = zero_embeds, zero_embeds
        else:
            abstract_mean, abstract_max = get_text_embeds(abstract)
        paper2vector_abstract_mean[paper_id] = abstract_mean
        paper2vector_abstract_max[paper_id] = abstract_max

        if (isinstance(keywords, str) and keywords.strip() == '') or \
                (isinstance(keywords, list) and len(keywords) == 0) or \
                (isinstance(keywords, list) and len(keywords) == 1 and (keywords[0] == 'null' or keywords[0] == '')):
            keywords_completion_mean, keywords_completion_max = zero_embeds, zero_embeds
        else:
            keywords_completion_mean, keywords_completion_max = get_keywords_embeds(keywords)
        paper2vector_keywords_completion_mean[paper_id] = keywords_completion_mean
        paper2vector_keywords_completion_max[paper_id] = keywords_completion_max

        partial_keywords = list(map(lambda x: x.lower().strip().split(), keywords))
        partial_keywords = sum(partial_keywords, [])
        if (len(partial_keywords) == 0) or \
                ((len(partial_keywords) == 1) and (keywords[0] == 'null' or keywords[0] == '')):
            keywords_partial_mean, keywords_partial_max = zero_embeds, zero_embeds
        else:
            keywords_partial_mean, keywords_partial_max = get_keywords_embeds(partial_keywords)
        paper2vector_keywords_partial_mean[paper_id] = keywords_partial_mean
        paper2vector_keywords_partial_max[paper_id] = keywords_partial_max

    return paper2vector_title_mean, paper2vector_title_max, \
           paper2vector_abstract_mean, paper2vector_abstract_max, \
           paper2vector_keywords_completion_mean, paper2vector_keywords_completion_max, \
           paper2vector_keywords_partial_mean, paper2vector_keywords_partial_max


def save_paper_vectors(paper_datas, data_types, output_dir):
    print('getting {} paper vectors...'.format(data_types))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    paper2vector_title_mean, paper2vector_title_max, \
    paper2vector_abstract_mean, paper2vector_abstract_max, \
    paper2vector_keywords_completion_mean, paper2vector_keywords_completion_max, \
    paper2vector_keywords_partial_mean, paper2vector_keywords_partial_max = get_paper_vector(paper_datas)

    pickle.dump(paper2vector_title_mean,
                open(os.path.join(output_dir, r'{}_paper2vector_title_mean.pkl'.format(data_types)), 'wb'))
    pickle.dump(paper2vector_title_max,
                open(os.path.join(output_dir, r'{}_paper2vector_title_max.pkl'.format(data_types)), 'wb'))
    pickle.dump(paper2vector_abstract_mean,
                open(os.path.join(output_dir, r'{}_paper2vector_abstract_mean.pkl'.format(data_types)), 'wb'))
    pickle.dump(paper2vector_abstract_max,
                open(os.path.join(output_dir, r'{}_paper2vector_abstract_max.pkl'.format(data_types)), 'wb'))
    pickle.dump(paper2vector_keywords_completion_mean,
                open(os.path.join(output_dir, r'{}_paper2vector_keywords_completion_mean.pkl'.format(data_types)), 'wb'))
    pickle.dump(paper2vector_keywords_completion_max,
                open(os.path.join(output_dir, r'{}_paper2vector_keywords_completion_max.pkl'.format(data_types)), 'wb'))
    pickle.dump(paper2vector_keywords_partial_mean,
                open(os.path.join(output_dir, r'{}_paper2vector_keywords_partial_mean.pkl'.format(data_types)), 'wb'))
    pickle.dump(paper2vector_keywords_partial_max,
                open(os.path.join(output_dir, r'{}_paper2vector_keywords_partial_max.pkl'.format(data_types)), 'wb'))

    return paper2vector_title_mean, paper2vector_title_max, \
               paper2vector_abstract_mean, paper2vector_abstract_max, \
               paper2vector_keywords_completion_mean, paper2vector_keywords_completion_max, \
               paper2vector_keywords_partial_mean, paper2vector_keywords_partial_max


def get_vector_for_each_author_each_embed_type(author_id, paper_list, paper_mean_vectors, paper_max_vectors,
                                               mean_mean_vectors, mean_max_vectors, max_mean_vectors, max_max_vectors):
    cur_mean_mean_vectors = []
    cur_mean_max_vectors = []
    cur_max_mean_vectors = []
    cur_max_max_vectors = []

    for paper in paper_list:
        if paper in paper_mean_vectors:
            cur_mean_mean_vectors.append(paper_mean_vectors[paper])
            cur_mean_max_vectors.append(paper_mean_vectors[paper])

        if paper in paper_max_vectors:
            cur_max_mean_vectors.append(paper_max_vectors[paper])
            cur_max_max_vectors.append(paper_max_vectors[paper])

    if len(cur_mean_mean_vectors) != 0:
        mean_mean_vectors[author_id] = np.mean(cur_mean_mean_vectors, axis=0)
        mean_max_vectors[author_id] = np.max(cur_mean_max_vectors, axis=0)
    else:
        mean_mean_vectors[author_id] = zero_embeds
        mean_max_vectors[author_id] = zero_embeds

    if len(cur_max_mean_vectors) != 0:
        max_mean_vectors[author_id] = np.mean(cur_max_mean_vectors, axis=0)
        max_max_vectors[author_id] = np.max(cur_max_max_vectors, axis=0)
    else:
        max_mean_vectors[author_id] = zero_embeds
        max_max_vectors[author_id] = zero_embeds


def get_author_vectors(author_datas, paper2vector, data_types, filter_papers=None):
    author2vector_title_mean_mean, author2vector_title_mean_max = {}, {}  # 论文内部是mean，作者部分也是mean；论文内部是mean，作者部分是max
    author2vector_title_max_mean, author2vector_title_max_max = {}, {}
    author2vector_abstract_mean_mean, author2vector_abstract_mean_max = {}, {}
    author2vector_abstract_max_mean, author2vector_abstract_max_max = {}, {}
    author2vector_keywords_completion_mean_mean, author2vector_keywords_completion_mean_max = {}, {}
    author2vector_keywords_completion_max_mean, author2vector_keywords_completion_max_max = {}, {}
    author2vector_keywords_partial_mean_mean, author2vector_keywords_partial_mean_max = {}, {}
    author2vector_keywords_partial_max_mean, author2vector_keywords_partial_max_max = {}, {}

    paper2vector_title_mean, paper2vector_title_max, \
    paper2vector_abstract_mean, paper2vector_abstract_max, \
    paper2vector_keywords_completion_mean, paper2vector_keywords_completion_max, \
    paper2vector_keywords_partial_mean, paper2vector_keywords_partial_max = paper2vector

    if data_types == 'whole':
        for author_id, author_infos in tqdm(author_datas.items()):
            get_vector_for_each_author_each_embed_type(author_id, author_infos['pubs'], paper2vector_title_mean,
                                                       paper2vector_title_max, author2vector_title_mean_mean,
                                                       author2vector_title_mean_max, author2vector_title_max_mean,
                                                       author2vector_title_max_max)
            get_vector_for_each_author_each_embed_type(author_id, author_infos['pubs'], paper2vector_abstract_mean,
                                                       paper2vector_abstract_max, author2vector_abstract_mean_mean,
                                                       author2vector_abstract_mean_max, author2vector_abstract_max_mean,
                                                       author2vector_abstract_max_max)
            get_vector_for_each_author_each_embed_type(author_id, author_infos['pubs'],
                                                       paper2vector_keywords_completion_mean,
                                                       paper2vector_keywords_completion_max,
                                                       author2vector_keywords_completion_mean_mean,
                                                       author2vector_keywords_completion_mean_max,
                                                       author2vector_keywords_completion_max_mean,
                                                       author2vector_keywords_completion_max_max)
            get_vector_for_each_author_each_embed_type(author_id, author_infos['pubs'],
                                                       paper2vector_keywords_partial_mean,
                                                       paper2vector_keywords_partial_max,
                                                       author2vector_keywords_partial_mean_mean,
                                                       author2vector_keywords_partial_mean_max,
                                                       author2vector_keywords_partial_max_mean,
                                                       author2vector_keywords_partial_max_max)
    else:
        for name, name_infos in tqdm(author_datas.items()):
            for author_id, author_paper_list in name_infos.items():
                if filter_papers is not None:
                    paper_list = list(set(author_paper_list) & set(filter_papers[author_id]))
                else:
                    paper_list = author_paper_list

                get_vector_for_each_author_each_embed_type(author_id, paper_list, paper2vector_title_mean,
                                                           paper2vector_title_max, author2vector_title_mean_mean,
                                                           author2vector_title_mean_max, author2vector_title_max_mean,
                                                           author2vector_title_max_max)
                get_vector_for_each_author_each_embed_type(author_id, paper_list, paper2vector_abstract_mean,
                                                           paper2vector_abstract_max, author2vector_abstract_mean_mean,
                                                           author2vector_abstract_mean_max,
                                                           author2vector_abstract_max_mean,
                                                           author2vector_abstract_max_max)
                get_vector_for_each_author_each_embed_type(author_id, paper_list,
                                                           paper2vector_keywords_completion_mean,
                                                           paper2vector_keywords_completion_max,
                                                           author2vector_keywords_completion_mean_mean,
                                                           author2vector_keywords_completion_mean_max,
                                                           author2vector_keywords_completion_max_mean,
                                                           author2vector_keywords_completion_max_max)
                get_vector_for_each_author_each_embed_type(author_id, paper_list,
                                                           paper2vector_keywords_partial_mean,
                                                           paper2vector_keywords_partial_max,
                                                           author2vector_keywords_partial_mean_mean,
                                                           author2vector_keywords_partial_mean_max,
                                                           author2vector_keywords_partial_max_mean,
                                                           author2vector_keywords_partial_max_max)

    return author2vector_title_mean_mean, author2vector_title_mean_max,\
           author2vector_title_max_mean, author2vector_title_max_max,\
           author2vector_abstract_mean_mean, author2vector_abstract_mean_max,\
           author2vector_abstract_max_mean, author2vector_abstract_max_max,\
           author2vector_keywords_completion_mean_mean, author2vector_keywords_completion_mean_max,\
           author2vector_keywords_completion_max_mean, author2vector_keywords_completion_max_max, \
           author2vector_keywords_partial_mean_mean, author2vector_keywords_partial_mean_max,\
           author2vector_keywords_partial_max_mean, author2vector_keywords_partial_max_max


def save_author_vectors(author_datas, paper2vector, data_types, output_dir, filter_papers=None):
    print('getting {} author vectors...'.format(data_types))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    author2vector_title_mean_mean, author2vector_title_mean_max, \
    author2vector_title_max_mean, author2vector_title_max_max, \
    author2vector_abstract_mean_mean, author2vector_abstract_mean_max, \
    author2vector_abstract_max_mean, author2vector_abstract_max_max, \
    author2vector_keywords_completion_mean_mean, author2vector_keywords_completion_mean_max, \
    author2vector_keywords_completion_max_mean, author2vector_keywords_completion_max_max, \
    author2vector_keywords_partial_mean_mean, author2vector_keywords_partial_mean_max, \
    author2vector_keywords_partial_max_mean, author2vector_keywords_partial_max_max = \
        get_author_vectors(author_datas, paper2vector, data_types, filter_papers)

    pickle.dump(author2vector_title_mean_mean,
                open(os.path.join(output_dir, r'{}_author2vector_title_mean_mean.pkl'.format(data_types)), 'wb'))
    pickle.dump(author2vector_title_mean_max,
                open(os.path.join(output_dir, r'{}_author2vector_title_mean_max.pkl'.format(data_types)), 'wb'))
    pickle.dump(author2vector_title_max_mean,
                open(os.path.join(output_dir, r'{}_author2vector_title_max_mean.pkl'.format(data_types)), 'wb'))
    pickle.dump(author2vector_title_max_max,
                open(os.path.join(output_dir, r'{}_author2vector_title_max_max.pkl'.format(data_types)), 'wb'))

    pickle.dump(author2vector_abstract_mean_mean,
                open(os.path.join(output_dir, r'{}_author2vector_abstract_mean_mean.pkl'.format(data_types)), 'wb'))
    pickle.dump(author2vector_abstract_mean_max,
                open(os.path.join(output_dir, r'{}_author2vector_abstract_mean_max.pkl'.format(data_types)), 'wb'))
    pickle.dump(author2vector_abstract_max_mean,
                open(os.path.join(output_dir, r'{}_author2vector_abstract_max_mean.pkl'.format(data_types)), 'wb'))
    pickle.dump(author2vector_abstract_max_max,
                open(os.path.join(output_dir, r'{}_author2vector_abstract_max_max.pkl'.format(data_types)), 'wb'))

    pickle.dump(author2vector_keywords_completion_mean_mean,
                open(os.path.join(output_dir, r'{}_author2vector_keywords_completion_mean_mean.pkl'.format(data_types)), 'wb'))
    pickle.dump(author2vector_keywords_completion_mean_max,
                open(os.path.join(output_dir, r'{}_author2vector_keywords_completion_mean_max.pkl'.format(data_types)), 'wb'))
    pickle.dump(author2vector_keywords_completion_max_mean,
                open(os.path.join(output_dir, r'{}_author2vector_keywords_completion_max_mean.pkl'.format(data_types)), 'wb'))
    pickle.dump(author2vector_keywords_completion_max_max,
                open(os.path.join(output_dir, r'{}_author2vector_keywords_completion_max_max.pkl'.format(data_types)), 'wb'))

    pickle.dump(author2vector_keywords_partial_mean_mean,
                open(os.path.join(output_dir, r'{}_author2vector_keywords_partial_mean_mean.pkl'.format(data_types)), 'wb'))
    pickle.dump(author2vector_keywords_partial_mean_max,
                open(os.path.join(output_dir, r'{}_author2vector_keywords_partial_mean_max.pkl'.format(data_types)), 'wb'))
    pickle.dump(author2vector_keywords_partial_max_mean,
                open(os.path.join(output_dir, r'{}_author2vector_keywords_partial_max_mean.pkl'.format(data_types)), 'wb'))
    pickle.dump(author2vector_keywords_partial_max_max,
                open(os.path.join(output_dir, r'{}_author2vector_keywords_partial_max_max.pkl'.format(data_types)), 'wb'))


def load_paper_vectors(load_dir, data_types):
    paper2vector_title_mean = pickle.load(open(os.path.join(load_dir, '{}_paper2vector_title_mean.pkl'.format(data_types)), 'rb'))
    paper2vector_title_max = pickle.load(open(os.path.join(load_dir, '{}_paper2vector_title_max.pkl'.format(data_types)), 'rb'))
    paper2vector_abstract_mean = pickle.load(open(os.path.join(load_dir, '{}_paper2vector_abstract_mean.pkl'.format(data_types)), 'rb'))
    paper2vector_abstract_max = pickle.load(open(os.path.join(load_dir, '{}_paper2vector_abstract_max.pkl'.format(data_types)), 'rb'))
    paper2vector_keywords_completion_mean = pickle.load(open(os.path.join(load_dir, '{}_paper2vector_keywords_completion_mean.pkl'.format(data_types)), 'rb'))
    paper2vector_keywords_completion_max = pickle.load(open(os.path.join(load_dir, '{}_paper2vector_keywords_completion_max.pkl'.format(data_types)), 'rb'))
    paper2vector_keywords_partial_mean = pickle.load(open(os.path.join(load_dir, '{}_paper2vector_keywords_partial_mean.pkl'.format(data_types)), 'rb'))
    paper2vector_keywords_partial_max = pickle.load(open(os.path.join(load_dir, '{}_paper2vector_keywords_partial_max.pkl'.format(data_types)), 'rb'))

    return paper2vector_title_mean, paper2vector_title_max, \
           paper2vector_abstract_mean, paper2vector_abstract_max, \
           paper2vector_keywords_completion_mean, paper2vector_keywords_completion_max, \
           paper2vector_keywords_partial_mean, paper2vector_keywords_partial_max


def load_author_vectors(load_dir, data_types):
    author2vector_title_mean_mean = pickle.load(open(os.path.join(load_dir, r'{}_author2vector_title_mean_mean.pkl'.format(data_types)), 'rb'))
    author2vector_title_mean_max = pickle.load(open(os.path.join(load_dir, r'{}_author2vector_title_mean_max.pkl'.format(data_types)), 'rb'))
    author2vector_title_max_mean = pickle.load(open(os.path.join(load_dir, r'{}_author2vector_title_max_mean.pkl'.format(data_types)), 'rb'))
    author2vector_title_max_max = pickle.load(open(os.path.join(load_dir, r'{}_author2vector_title_max_max.pkl'.format(data_types)), 'rb'))
    author2vector_abstract_mean_mean = pickle.load(open(os.path.join(load_dir, r'{}_author2vector_abstract_mean_mean.pkl'.format(data_types)), 'rb'))
    author2vector_abstract_mean_max = pickle.load(open(os.path.join(load_dir, r'{}_author2vector_abstract_mean_max.pkl'.format(data_types)), 'rb'))
    author2vector_abstract_max_mean = pickle.load(open(os.path.join(load_dir, r'{}_author2vector_abstract_max_mean.pkl'.format(data_types)), 'rb'))
    author2vector_abstract_max_max = pickle.load(open(os.path.join(load_dir, r'{}_author2vector_abstract_max_max.pkl'.format(data_types)), 'rb'))
    author2vector_keywords_completion_mean_mean = pickle.load(open(os.path.join(load_dir, r'{}_author2vector_keywords_completion_mean_mean.pkl'.format(data_types)), 'rb'))
    author2vector_keywords_completion_mean_max = pickle.load(open(os.path.join(load_dir, r'{}_author2vector_keywords_completion_mean_max.pkl'.format(data_types)), 'rb'))
    author2vector_keywords_completion_max_mean = pickle.load(open(os.path.join(load_dir, r'{}_author2vector_keywords_completion_max_mean.pkl'.format(data_types)), 'rb'))
    author2vector_keywords_completion_max_max = pickle.load(open(os.path.join(load_dir, r'{}_author2vector_keywords_completion_max_max.pkl'.format(data_types)), 'rb'))
    author2vector_keywords_partial_mean_mean = pickle.load(open(os.path.join(load_dir, r'{}_author2vector_keywords_partial_mean_mean.pkl'.format(data_types)), 'rb'))
    author2vector_keywords_partial_mean_max = pickle.load(open(os.path.join(load_dir, r'{}_author2vector_keywords_partial_mean_max.pkl'.format(data_types)), 'rb'))
    author2vector_keywords_partial_max_mean = pickle.load(open(os.path.join(load_dir, r'{}_author2vector_keywords_partial_max_mean.pkl'.format(data_types)), 'rb'))
    author2vector_keywords_partial_max_max = pickle.load(open(os.path.join(load_dir, r'{}_author2vector_keywords_partial_max_max.pkl'.format(data_types)), 'rb'))

    return author2vector_title_mean_mean, author2vector_title_mean_max, \
            author2vector_title_max_mean, author2vector_title_max_max, \
            author2vector_abstract_mean_mean, author2vector_abstract_mean_max, \
            author2vector_abstract_max_mean, author2vector_abstract_max_max, \
            author2vector_keywords_completion_mean_mean, author2vector_keywords_completion_mean_max, \
            author2vector_keywords_completion_max_mean, author2vector_keywords_completion_max_max, \
            author2vector_keywords_partial_mean_mean, author2vector_keywords_partial_mean_max, \
            author2vector_keywords_partial_max_mean, author2vector_keywords_partial_max_max


if __name__ == '__main__':
    # 读取数据
    glove_vectors = {}
    with open(r'./glove.840B.300d.txt') as f:
        for line in tqdm(f):
            line = line.strip().split()
            word, vector = line[:-300], list(map(float, line[-300:]))
            if len(word) > 1 or len(word) == 0:  # 大于1的基本都是邮箱格式，或者连续好几个点，直接忽略即可
                continue

            glove_vectors[word[0]] = np.array(vector)

    from data_utils import load_datas

    train_name_infos, train_paper_infos, whole_author_infos, whole_paper_infos, valid_unass, valid_paper_infos, \
        test_unass, test_paper_infos = load_datas('./datas/Task1')

    paper_output_dir = r'resource/glove_embeddings/papers'
    # author_output_dir = r'resource/glove_embeddings/authors'

    whole_paper_vectors = save_paper_vectors(whole_paper_infos, 'whole', paper_output_dir)
    valid_paper_vectors = save_paper_vectors(valid_paper_infos, 'valid', paper_output_dir)
    train_paper_vectors = save_paper_vectors(train_paper_infos, 'train', paper_output_dir)
    test_paper_vectors = save_paper_vectors(test_paper_infos, 'test', paper_output_dir)

    # whole_paper_vectors = load_paper_vectors(paper_output_dir, 'whole')
    # save_author_vectors(whole_author_infos, whole_paper_vectors, 'whole', author_output_dir)

    # train_paper_vectors = load_paper_vectors(paper_output_dir, 'train')
    # save_author_vectors(train_name_infos, train_paper_vectors, 'train', author_output_dir)

