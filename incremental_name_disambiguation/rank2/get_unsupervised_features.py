#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import json
import time
import regex as re
import numpy as np
import _pickle as pickle
from tqdm import tqdm, trange
from copy import copy, deepcopy
from itertools import chain
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from build_glove_embeds import load_paper_vectors
from data_utils import load_datas
from name_utils import process_name, get_int_for_name, clean_orgs, get_all_int_for_name

# import warnings
# warnings.filterwarnings('ignore')
from warnings import simplefilter
simplefilter(action='ignore', category='DeprecationWarning')

stopword = {'at', 'based', 'in', 'of', 'for', 'on', 'and', 'to', 'an', 'using', 'with', 'the', 'by', 'we', 'be', ' is',
            'are', 'can'}


def get_train_author_org_weights(datas, paper_infos):
    paper_author_org_weights = []
    for i in trange(len(datas)):
        author_name, aid, pid_order, pos_list, neg_list = datas[i][0][:5]
        pid, order = pid_order.split('-')
        paper = paper_infos[pid]
        author = paper['authors'][int(order)]
        org = set(clean_orgs(author['org']).split())

        author_orgs = {}
        for each in pos_list:
            aid, pid, order = each.split('-')
            if aid not in author_orgs:
                author_orgs[aid] = []
            paper = paper_infos[pid]
            author = paper['authors'][int(order)]
            cur_org = clean_orgs(author['org'])
            author_orgs[aid].append(cur_org)

        for each in neg_list:
            aid, pid, order = each.split('-')
            if aid not in author_orgs:
                author_orgs[aid] = []
            paper = paper_infos[pid]
            author = paper['authors'][int(order)]
            cur_org = clean_orgs(author['org'])
            author_orgs[aid].append(cur_org)

        # 每个作者都单独算一个tfidf得到关键词
        author_org_tfidf = {}
        for author, orgs in author_orgs.items():
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

        author_weights = {}
        for author, top_org_words in author_org_tfidf.items():
            if len(org) != 0 and len(top_org_words) != 0:
                org_weight = len(org & top_org_words) / (len(org) + len(top_org_words) + 1e-20)
            else:
                org_weight = -1
            author_weights[author] = org_weight

        paper_author_org_weights.append(author_weights)

    return paper_author_org_weights


def get_train_author_keywords_weights(datas, paper_infos):
    paper_author_keywords_weights = []
    for i in trange(len(datas)):
        author_name, aid, pid_order, pos_list, neg_list = datas[i][0][:5]
        pid, order = pid_order.split('-')
        paper = paper_infos[pid]
        keywords = paper['keywords']
        if (isinstance(keywords, str) and keywords.strip() == '') or \
                (isinstance(keywords, list) and len(keywords) == 0) or \
                (isinstance(keywords, list) and len(keywords) == 1 and
                 (keywords[0] == 'null' or keywords[0] == '')):
            keywords = set()
        else:
            keywords = list(map(lambda x: x.strip().split(), keywords))
            keywords = set(chain.from_iterable(keywords))

        author_keywords = {}
        for each in pos_list:
            aid, pid, order = each.split('-')
            if aid not in author_keywords:
                author_keywords[aid] = []
            paper = paper_infos[pid]
            cur_keywords = paper['keywords']
            if (isinstance(cur_keywords, str) and cur_keywords.strip() == '') or \
                    (isinstance(cur_keywords, list) and len(cur_keywords) == 0) or \
                    (isinstance(cur_keywords, list) and len(cur_keywords) == 1 and
                     (cur_keywords[0] == 'null' or cur_keywords[0] == '')):
                continue
            else:
                cur_keywords = list(map(lambda x: x.strip().split(), cur_keywords))
                cur_keywords = ' '.join(list(chain.from_iterable(cur_keywords)))
            author_keywords[aid].append(cur_keywords)

        for each in neg_list:
            aid, pid, order = each.split('-')
            if aid not in author_keywords:
                author_keywords[aid] = []
            paper = paper_infos[pid]
            cur_keywords = paper['keywords']
            if (isinstance(cur_keywords, str) and cur_keywords.strip() == '') or \
                    (isinstance(cur_keywords, list) and len(cur_keywords) == 0) or \
                    (isinstance(cur_keywords, list) and len(cur_keywords) == 1 and
                     (cur_keywords[0] == 'null' or cur_keywords[0] == '')):
                continue
            else:
                cur_keywords = list(map(lambda x: x.strip().split(), cur_keywords))
                cur_keywords = ' '.join(list(chain.from_iterable(cur_keywords)))
            author_keywords[aid].append(cur_keywords)

        # 每个作者都单独算一个tfidf得到关键词
        author_keywords_tfidf = {}
        for author, cur_keywords in author_keywords.items():
            vectorizer = TfidfVectorizer()
            try:
                X = vectorizer.fit_transform(cur_keywords)
            except:
                author_keywords_tfidf[author] = set()
            else:
                weight = np.sum(X.toarray(), axis=0)
                word_score = list(zip(vectorizer.get_feature_names(), weight))
                word_score = sorted(word_score, key=lambda x: x[1], reverse=True)

                top_words = [w for w, s in word_score[:20]]
                author_keywords_tfidf[author] = set(top_words)

        author_weights = {}
        for author, top_keywords_words in author_keywords_tfidf.items():
            if len(keywords) != 0 and len(top_keywords_words) != 0:
                keywords_weight = len(keywords & top_keywords_words) / (len(keywords) + len(top_keywords_words) + 1e-20)
            else:
                keywords_weight = -1
            author_weights[author] = keywords_weight

        paper_author_keywords_weights.append(author_weights)

    return paper_author_keywords_weights


def get_train_author_coauthor_weights(datas, paper_infos):
    paper_author_coauthor_weights = []
    for i in trange(len(datas)):
        author_name, aid, pid_order, pos_list, neg_list = datas[i][0][:5]
        pid, order = pid_order.split('-')
        paper = paper_infos[pid]
        coauthors = list(map(lambda x: x['name'], paper['authors']))
        coauthors.pop(int(order))

        coauthor_ints = []
        for author in coauthors:
            coauthor_ints.append(get_int_for_name(process_name(author)))

        author_coauthors = {}
        for each in pos_list:
            aid, pid, order = each.split('-')
            if aid not in author_coauthors:
                author_coauthors[aid] = []
            paper = paper_infos[pid]
            cur_coauthors = list(map(lambda x: x['name'], paper['authors']))
            cur_coauthors.pop(int(order))
            author_coauthors[aid].extend(cur_coauthors)

        for each in neg_list:
            aid, pid, order = each.split('-')
            if aid not in author_coauthors:
                author_coauthors[aid] = []
            paper = paper_infos[pid]
            cur_coauthors = list(map(lambda x: x['name'], paper['authors']))
            cur_coauthors.pop(int(order))
            author_coauthors[aid].extend(cur_coauthors)

        author_weights = {}
        for author, coauthors in author_coauthors.items():
            matching_cnt = 0
            for _coauthor in coauthors:
                name_ints = get_all_int_for_name(_coauthor)
                if len(set(name_ints) & set(coauthor_ints)) != 0:
                    matching_cnt += 1

            weight = matching_cnt / (len(coauthor_ints) + len(coauthors) - 2 * matching_cnt + 1e-20)
            author_weights[author] = weight

        paper_author_coauthor_weights.append(author_weights)

    return paper_author_coauthor_weights


def get_train_author_embeds(paper_list, paper_title_embeds, paper_abstract_embeds, paper_keywords_embeds,
                            author_title_embeds, author_abstract_embeds, author_keywords_embeds):
    for each in paper_list:
        aid, pid, order = each.split('-')
        if aid not in author_title_embeds:
            author_title_embeds[aid] = []
        if aid not in author_abstract_embeds:
            author_abstract_embeds[aid] = []
        if aid not in author_keywords_embeds:
            author_keywords_embeds[aid] = []

        author_title_embeds[aid].append(paper_title_embeds[pid])
        author_abstract_embeds[aid].append(paper_abstract_embeds[pid])
        author_keywords_embeds[aid].append(paper_keywords_embeds[pid])


def get_train_embed_similarity(datas, paper_vector_dir):
    train_paper2vector_title_mean, train_paper2vector_title_max, \
    train_paper2vector_abstract_mean, train_paper2vector_abstract_max, \
    train_paper2vector_keywords_completion_mean, train_paper2vector_keywords_completion_max, \
    train_paper2vector_keywords_partial_mean, train_paper2vector_keywords_partial_max = \
        load_paper_vectors(paper_vector_dir, 'train')

    paper_author_embeds_similarity = []
    for i in trange(len(datas)):
        author_name, aid, pid_order, pos_list, neg_list = datas[i][0][:5]
        pid, order = pid_order.split('-')

        title_vector = np.array(train_paper2vector_title_mean[pid]).reshape(1, -1)
        abstract_vector = np.array(train_paper2vector_abstract_mean[pid]).reshape(1, -1)
        keywords_vector = np.array(train_paper2vector_keywords_completion_mean[pid]).reshape(1, -1)

        author_title_embeds, author_abstract_embeds, author_keywords_embeds = {}, {}, {}
        get_train_author_embeds(pos_list, train_paper2vector_title_mean, train_paper2vector_abstract_mean,
                                train_paper2vector_keywords_completion_mean, author_title_embeds, author_abstract_embeds,
                                author_keywords_embeds)
        get_train_author_embeds(neg_list, train_paper2vector_title_mean, train_paper2vector_abstract_mean,
                                train_paper2vector_keywords_completion_mean, author_title_embeds, author_abstract_embeds,
                                author_keywords_embeds)

        for author, embeds in author_title_embeds.items():
            if len(author_title_embeds) != 0:
                author_title_embeds[author] = np.mean(embeds, axis=0)
            else:
                author_title_embeds[author] = np.zeros((300,), dtype=float)

        for author, embeds in author_abstract_embeds.items():
            if len(author_abstract_embeds) != 0:
                author_abstract_embeds[author] = np.mean(embeds, axis=0)
            else:
                author_abstract_embeds[author] = np.zeros((300,), dtype=float)

        for author, embeds in author_keywords_embeds.items():
            if len(author_keywords_embeds) != 0:
                author_keywords_embeds[author] = np.mean(embeds, axis=0)
            else:
                author_keywords_embeds[author] = np.zeros((300,), dtype=float)

        author2idx = dict(zip(list(author_title_embeds.keys()), list(range(len(author_title_embeds)))))
        idx2author = {i: p for p, i in author2idx.items()}

        title_similarity = cosine_similarity(title_vector, np.array(list(author_title_embeds.values())))[0]
        abstract_similarity = cosine_similarity(abstract_vector, np.array(list(author_abstract_embeds.values())))[0]
        keywords_similarity = cosine_similarity(keywords_vector, np.array(list(author_keywords_embeds.values())))[0]
        similarity = title_similarity + abstract_similarity + keywords_similarity

        title_order = np.argsort(-title_similarity)
        abstract_order = np.argsort(-abstract_similarity)
        keywords_order = np.argsort(-keywords_similarity)
        order = np.argsort(-similarity)

        author_similarity = {}
        for idx, score in enumerate(title_similarity):
            aid = idx2author[idx]
            cur_similarity = [title_similarity[idx], abstract_similarity[idx], keywords_similarity[idx], similarity[idx]]
            cur_order = [title_order[idx] + 1, abstract_order[idx] + 1, keywords_order[idx] + 1, order[idx] + 1]
            author_similarity[aid] = cur_similarity + cur_order

        paper_author_embeds_similarity.append(author_similarity)

    return paper_author_embeds_similarity


def get_whole_author_org_weights_official_recall(datas, unass_paper_infos, refer_paper_infos, nameAidPid):
    paper_author_org_weights = []
    for i in trange(len(datas)):
        pid_order, name = datas[i]
        pid, order = pid_order.split('-')
        paper = unass_paper_infos[pid]
        author = paper['authors'][int(order)]
        org = set(clean_orgs(author['org']).split())

        author_orgs = {}
        candiAuthors = list(nameAidPid[name].keys())
        for aid in candiAuthors:
            author_orgs[aid] = []
            totalPubs = nameAidPid[name][aid]
            for pub in totalPubs:
                pid, order = pub.split('-')
                paper = refer_paper_infos[pid]
                author = paper['authors'][int(order)]
                cur_org = clean_orgs(author['org'])
                author_orgs[aid].append(cur_org)

        author_org_tfidf = {}
        for author, orgs in author_orgs.items():
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

        author_weights = {}
        for author, top_org_words in author_org_tfidf.items():
            if len(org) != 0 and len(top_org_words) != 0:
                org_weight = len(org & top_org_words) / (len(org) + len(top_org_words) + 1e-20)
            else:
                org_weight = -1
            author_weights[author] = org_weight

        paper_author_org_weights.append(author_weights)

    return paper_author_org_weights


def get_whole_author_keywords_weights_official_recall(datas, unass_paper_infos, refer_paper_infos, nameAidPid):
    paper_author_keywords_weights = []
    for i in trange(len(datas)):
        pid_order, name = datas[i]
        pid, order = pid_order.split('-')
        paper = unass_paper_infos[pid]
        keywords = paper['keywords']
        if (isinstance(keywords, str) and keywords.strip() == '') or \
                (isinstance(keywords, list) and len(keywords) == 0) or \
                (isinstance(keywords, list) and len(keywords) == 1 and
                 (keywords[0] == 'null' or keywords[0] == '')):
            keywords = set()
        else:
            keywords = list(map(lambda x: x.strip().split(), keywords))
            keywords = set(chain.from_iterable(keywords))

        author_keywords = {}
        candiAuthors = list(nameAidPid[name].keys())
        for aid in candiAuthors:
            author_keywords[aid] = []
            totalPubs = nameAidPid[name][aid]
            for pub in totalPubs:
                pid, order = pub.split('-')
                paper = refer_paper_infos[pid]
                cur_keywords = paper['keywords']
                if (isinstance(cur_keywords, str) and cur_keywords.strip() == '') or \
                        (isinstance(cur_keywords, list) and len(cur_keywords) == 0) or \
                        (isinstance(cur_keywords, list) and len(cur_keywords) == 1 and
                         (cur_keywords[0] == 'null' or cur_keywords[0] == '')):
                    continue
                else:
                    cur_keywords = list(map(lambda x: x.strip().split(), cur_keywords))
                    cur_keywords = ' '.join(list(chain.from_iterable(cur_keywords)))
                author_keywords[aid].append(cur_keywords)

        author_keywords_tfidf = {}
        for author, cur_keywords in author_keywords.items():
            vectorizer = TfidfVectorizer()
            try:
                X = vectorizer.fit_transform(cur_keywords)
            except:
                author_keywords_tfidf[author] = set()
            else:
                weight = np.sum(X.toarray(), axis=0)
                word_score = list(zip(vectorizer.get_feature_names(), weight))
                word_score = sorted(word_score, key=lambda x: x[1], reverse=True)

                top_words = [w for w, s in word_score[:20]]
                author_keywords_tfidf[author] = set(top_words)

        author_weights = {}
        for author, top_keywords_words in author_keywords_tfidf.items():
            if len(keywords) != 0 and len(top_keywords_words) != 0:
                keywords_weight = len(keywords & top_keywords_words) / (len(keywords) + len(top_keywords_words) + 1e-20)
            else:
                keywords_weight = -1
            author_weights[author] = keywords_weight

        paper_author_keywords_weights.append(author_weights)

    return paper_author_keywords_weights


def get_whole_author_coauthor_weights_official_recall(datas, unass_paper_infos, refer_paper_infos, nameAidPid):
    paper_author_coauthor_weights = []
    for i in trange(len(datas)):
        pid_order, name = datas[i]
        pid, order = pid_order.split('-')
        paper = unass_paper_infos[pid]
        coauthors = list(map(lambda x: x['name'], paper['authors']))
        coauthors.pop(int(order))

        coauthor_ints = []
        for author in coauthors:
            coauthor_ints.append(get_int_for_name(process_name(author)))

        author_coauthors = {}
        candiAuthors = list(nameAidPid[name].keys())
        for aid in candiAuthors:
            author_coauthors[aid] = []
            totalPubs = nameAidPid[name][aid]
            for pub in totalPubs:
                pid, order = pub.split('-')
                paper = refer_paper_infos[pid]
                cur_coauthors = list(map(lambda x: x['name'], paper['authors']))
                cur_coauthors.pop(int(order))
                author_coauthors[aid].extend(cur_coauthors)

        author_weights = {}
        for author, coauthors in author_coauthors.items():
            matching_cnt = 0
            for _coauthor in coauthors:
                name_ints = get_all_int_for_name(_coauthor)
                if len(set(name_ints) & set(coauthor_ints)) != 0:
                    matching_cnt += 1

            weight = matching_cnt / (len(coauthor_ints) + len(coauthors) - 2 * matching_cnt + 1e-20)
            author_weights[author] = weight

        paper_author_coauthor_weights.append(author_weights)

    return paper_author_coauthor_weights


def get_whole_embed_similarity_official_recall(datas, paper_vector_dir, nameAidPid, data_type='valid'):
    valid_paper2vector_title_mean, valid_paper2vector_title_max, \
    valid_paper2vector_abstract_mean, valid_paper2vector_abstract_max, \
    valid_paper2vector_keywords_completion_mean, valid_paper2vector_keywords_completion_max, \
    valid_paper2vector_keywords_partial_mean, valid_paper2vector_keywords_partial_max = \
        load_paper_vectors(paper_vector_dir, data_type)

    whole_paper2vector_title_mean, whole_paper2vector_title_max, \
    whole_paper2vector_abstract_mean, whole_paper2vector_abstract_max, \
    whole_paper2vector_keywords_completion_mean, whole_paper2vector_keywords_completion_max, \
    whole_paper2vector_keywords_partial_mean, whole_paper2vector_keywords_partial_max = \
        load_paper_vectors(paper_vector_dir, 'whole')

    paper_author_embeds_similarity = []
    for i in trange(len(datas)):
        pid_order, name = datas[i]
        pid, order = pid_order.split('-')

        title_vector = np.array(valid_paper2vector_title_mean[pid]).reshape(1, -1)
        abstract_vector = np.array(valid_paper2vector_abstract_mean[pid]).reshape(1, -1)
        keywords_vector = np.array(valid_paper2vector_keywords_completion_mean[pid]).reshape(1, -1)

        author_title_embeds, author_abstract_embeds, author_keywords_embeds = {}, {}, {}
        candiAuthors = list(nameAidPid[name].keys())
        for aid in candiAuthors:
            author_title_embeds[aid] = []
            author_abstract_embeds[aid] = []
            author_keywords_embeds[aid] = []

            totalPubs = nameAidPid[name][aid]
            for pub in totalPubs:
                pid, order = pub.split('-')
                author_title_embeds[aid].append(whole_paper2vector_title_mean[pid])
                author_abstract_embeds[aid].append(whole_paper2vector_abstract_mean[pid])
                author_keywords_embeds[aid].append(whole_paper2vector_keywords_completion_mean[pid])

        for author, embeds in author_title_embeds.items():
            if len(embeds) != 0:
                author_title_embeds[author] = np.mean(embeds, axis=0)
            else:
                author_title_embeds[author] = np.zeros((300,), dtype=float)

        for author, embeds in author_abstract_embeds.items():
            if len(embeds) != 0:
                author_abstract_embeds[author] = np.mean(embeds, axis=0)
            else:
                author_abstract_embeds[author] = np.zeros((300,), dtype=float)

        for author, embeds in author_keywords_embeds.items():
            if len(embeds) != 0:
                author_keywords_embeds[author] = np.mean(embeds, axis=0)
            else:
                author_keywords_embeds[author] = np.zeros((300,), dtype=float)

        author2idx = dict(zip(list(author_title_embeds.keys()), list(range(len(author_title_embeds)))))
        idx2author = {i: p for p, i in author2idx.items()}

        title_similarity = cosine_similarity(title_vector, np.array(list(author_title_embeds.values())))[0]
        abstract_similarity = cosine_similarity(abstract_vector, np.array(list(author_abstract_embeds.values())))[0]
        keywords_similarity = cosine_similarity(keywords_vector, np.array(list(author_keywords_embeds.values())))[0]
        similarity = title_similarity + abstract_similarity + keywords_similarity

        title_order = np.argsort(-title_similarity)
        abstract_order = np.argsort(-abstract_similarity)
        keywords_order = np.argsort(-keywords_similarity)
        order = np.argsort(-similarity)

        author_similarity = {}
        for idx, score in enumerate(title_similarity):
            aid = idx2author[idx]
            cur_similarity = [title_similarity[idx], abstract_similarity[idx], keywords_similarity[idx],
                              similarity[idx]]
            cur_order = [title_order[idx] + 1, abstract_order[idx] + 1, keywords_order[idx] + 1, order[idx] + 1]
            author_similarity[aid] = cur_similarity + cur_order

        paper_author_embeds_similarity.append(author_similarity)

    return paper_author_embeds_similarity


def clean_abstract(abstract):
    abstract = re.sub(r'\W+', ' ', abstract)
    abstract = re.sub(r'\_', ' ', abstract)
    abstract = abstract.lower().strip()
    return abstract


def get_org_counts(nameAidPid, paper_infos):
    ngram_counts = {}
    aid2len = {}

    aid2orgs = {}
    for name, name_infos in nameAidPid.items():
        for aid, pubs in name_infos.items():
            aid2orgs[aid] = []
            for pid_order in pubs:
                pid, order = pid_order.split('-')
                author = paper_infos[pid]['authors'][int(order)]
                org = clean_orgs(author['org'])
                if org == '':
                    continue
                aid2orgs[aid].append(org)

    for aid, orgs in tqdm(aid2orgs.items()):
        orgs = ' '.join(orgs)
        orgs = orgs.split()
        orgs = [w for w in orgs if w not in stopword]
        aid2len[aid] = len(orgs)

        for i in range(len(orgs)):
            unigram = orgs[i]
            if unigram not in ngram_counts:
                ngram_counts[unigram] = Counter()
            ngram_counts[unigram][aid] += 1

            bigram = tuple(orgs[i:i + 2])
            if len(bigram) < 2:
                continue
            if bigram not in ngram_counts:
                ngram_counts[bigram] = Counter()
            ngram_counts[bigram][aid] += 1

    pickle.dump(ngram_counts, open(os.path.join(save_dir, 'org_ngram_cnt.pkl'), 'wb'))
    pickle.dump(aid2len, open(os.path.join(save_dir, 'org_aid2len.pkl'), 'wb'))


def get_abstract_ngram_counts(author_infos, paper_infos):
    unigram_counts, bigram_counts, trigram_counts, forgram_counts, fivgram_counts = {}, {}, {}, {}, {}
    aid2len = {}

    aid2abstracts = {}
    for aid, infos in tqdm(author_infos.items()):
        aid2abstracts[aid] = []
        for pid in infos['pubs']:
            abstract = clean_abstract(paper_infos[pid]['abstract'])
            if abstract == '':
                continue
            aid2abstracts[aid].append(abstract)

    print(len(aid2abstracts))

    for aid, abstracts in tqdm(aid2abstracts.items()):
        abstracts = ' '.join(abstracts)
        abstracts = abstracts.split()
        abstracts = [w for w in abstracts if w not in stopword]
        aid2len[aid] = len(abstracts)

        for i in range(len(abstracts)):
            unigram = abstracts[i]
            if unigram not in unigram_counts:
                unigram_counts[unigram] = Counter()
            unigram_counts[unigram][aid] += 1

            bigram = tuple(abstracts[i:i + 2])
            if len(bigram) < 2:
                continue
            if bigram not in bigram_counts:
                bigram_counts[bigram] = Counter()
            bigram_counts[bigram][aid] += 1

            trigram = tuple(abstracts[i:i + 3])
            if len(trigram) < 3:
                continue
            if trigram not in trigram_counts:
                trigram_counts[trigram] = Counter()
            trigram_counts[trigram][aid] += 1

            forgram = tuple(abstracts[i:i + 4])
            if len(forgram) < 4:
                continue
            if forgram not in forgram_counts:
                forgram_counts[forgram] = Counter()
            forgram_counts[forgram][aid] += 1

            fivgram = tuple(abstracts[i:i + 5])
            if len(fivgram) < 5:
                continue
            if fivgram not in fivgram_counts:
                fivgram_counts[fivgram] = Counter()
            fivgram_counts[fivgram][aid] += 1

    print(len(unigram_counts), len(bigram_counts), len(trigram_counts), len(forgram_counts), len(fivgram_counts))

    pickle.dump(unigram_counts, open(os.path.join(save_dir, 'abstract_unigram_cnt.pkl'), 'wb'))
    pickle.dump(bigram_counts, open(os.path.join(save_dir, 'abstract_bigram_cnt.pkl'), 'wb'))
    pickle.dump(trigram_counts, open(os.path.join(save_dir, 'abstract_trigram_cnt.pkl'), 'wb'))
    pickle.dump(forgram_counts, open(os.path.join(save_dir, 'abstract_forgram_cnt.pkl'), 'wb'))
    pickle.dump(fivgram_counts, open(os.path.join(save_dir, 'abstract_fivgram_cnt.pkl'), 'wb'))
    pickle.dump(aid2len, open(os.path.join(save_dir, 'abstract_aid2len.pkl'), 'wb'))


def get_keywords_ngram_counts(author_infos, paper_infos):
    unigram_counts, bigram_counts, trigram_counts = {}, {}, {}
    aid2len = {}

    aid2keywords = {}
    for aid, infos in tqdm(author_infos.items()):
        aid2keywords[aid] = []
        for pid in infos['pubs']:
            keywords = paper_infos[pid]['keywords']
            if (isinstance(keywords, str) and keywords.strip() == '') or \
                    (isinstance(keywords, list) and len(keywords) == 0) or \
                    (isinstance(keywords, list) and len(keywords) == 1 and
                     (keywords[0] == 'null' or keywords[0] == '')):
                continue
            aid2keywords[aid].extend(keywords)

    print(len(aid2keywords))

    for aid, keywords in tqdm(aid2keywords.items()):
        keywords_len = 0
        for keyword in keywords:
            keyword = keyword.lower().split()
            keywords_len += len(keyword)

            for i in range(len(keyword)):
                unigram = keyword[i]
                if unigram not in unigram_counts:
                    unigram_counts[unigram] = Counter()
                unigram_counts[unigram][aid] += 1

                bigram = tuple(keyword[i:i + 2])
                if len(bigram) < 2:
                    continue
                if bigram not in bigram_counts:
                    bigram_counts[bigram] = Counter()
                bigram_counts[bigram][aid] += 1

                trigram = tuple(keyword[i:i + 3])
                if len(trigram) < 3:
                    continue
                if trigram not in trigram_counts:
                    trigram_counts[trigram] = Counter()
                trigram_counts[trigram][aid] += 1

        aid2len[aid] = keywords_len

    print(len(unigram_counts), len(bigram_counts), len(trigram_counts))

    pickle.dump(unigram_counts, open(os.path.join(save_dir, 'keywords_unigram_cnt.pkl'), 'wb'))
    pickle.dump(bigram_counts, open(os.path.join(save_dir, 'keywords_bigram_cnt.pkl'), 'wb'))
    pickle.dump(trigram_counts, open(os.path.join(save_dir, 'keywords_trigram_cnt.pkl'), 'wb'))
    pickle.dump(aid2len, open(os.path.join(save_dir, 'keywords_aid2len.pkl'), 'wb'))


def get_coauthor_ngram_counts(nameAidPid, paper_infos):
    ngram_counts = {}
    aid2len = {}

    aid2coauthors = {}
    for name, name_infos in nameAidPid.items():
        for aid, pubs in name_infos.items():
            aid2coauthors[aid] = []
            for pid_order in pubs:
                pid, order = pid_order.split('-')
                authors = paper_infos[pid]['authors']
                coauthors = deepcopy(authors)

                to_be_ass_author = coauthors[int(order)]
                coauthors.remove(to_be_ass_author)

                coauthors = list(map(lambda x: process_name(x['name']), coauthors))
                aid2coauthors[aid].extend(coauthors)

    for aid, coauthors in tqdm(aid2coauthors.items()):
        coauthors_len = 0
        for coauthor in coauthors:
            coauthor = coauthor.lower().split()
            coauthors_len += len(coauthor)

            for i in range(len(coauthor)):
                unigram = coauthor[i]
                if unigram not in ngram_counts:
                    ngram_counts[unigram] = Counter()
                ngram_counts[unigram][aid] += 1

                bigram = tuple(coauthor[i:i + 2])
                if len(bigram) < 2:
                    continue
                if bigram not in ngram_counts:
                    ngram_counts[bigram] = Counter()
                ngram_counts[bigram][aid] += 1

        aid2len[aid] = coauthors_len

    pickle.dump(ngram_counts, open(os.path.join(save_dir, 'coauthor_ngram_cnt.pkl'), 'wb'))
    pickle.dump(aid2len, open(os.path.join(save_dir, 'coauthor_aid2len.pkl'), 'wb'))


def get_title_ngram_counts(author_infos, paper_infos):
    ngram_counts = {}
    aid2len = {}

    aid2titles = {}
    for aid, infos in tqdm(author_infos.items()):
        aid2titles[aid] = []
        for pid in infos['pubs']:
            title = clean_abstract(paper_infos[pid]['title'])
            if title == '':
                continue
            aid2titles[aid].append(title)

    print(len(aid2titles))

    for aid, titles in tqdm(aid2titles.items()):
        titles = ' '.join(titles)
        titles = titles.split()
        titles = [w for w in titles if w not in stopword]
        aid2len[aid] = len(titles)

        for i in range(len(titles)):
            unigram = titles[i]
            if unigram not in ngram_counts:
                ngram_counts[unigram] = Counter()
            ngram_counts[unigram][aid] += 1

            bigram = tuple(titles[i:i + 2])
            if len(bigram) < 2:
                continue
            if bigram not in ngram_counts:
                ngram_counts[bigram] = Counter()
            ngram_counts[bigram][aid] += 1

    print(len(ngram_counts))

    pickle.dump(ngram_counts, open(os.path.join(save_dir, 'title_ngram_cnt.pkl'), 'wb'))
    pickle.dump(aid2len, open(os.path.join(save_dir, 'title_aid2len.pkl'), 'wb'))


def get_tfidf(ngram, ngram_counts, aid, aid2len):
    if ngram not in ngram_counts:
        return 0.
    if aid not in ngram_counts[ngram]:
        return 0.

    tf = ngram_counts[ngram][aid] / aid2len[aid]
    df = len(ngram_counts[ngram])
    idf = np.log(len(aid2len) / (df + 1))
    return tf * idf


def get_ngram_tfidf_score(ngrams, ngram_counts, aid, aid2len):
    tfidf_scores = []
    if len(ngrams) == 0:
        return 0.
    else:
        for ngram in ngrams:
            score = get_tfidf(ngram, ngram_counts, aid, aid2len)
            tfidf_scores.append(score)
        return np.mean(tfidf_scores)


def get_train_abstract_ngram_weights(datas, paper_infos):
    start_time = time.time()
    unigram_counts = pickle.load(open(os.path.join(save_dir, 'abstract_unigram_cnt.pkl'), 'rb'))
    bigram_counts = pickle.load(open(os.path.join(save_dir, 'abstract_bigram_cnt.pkl'), 'rb'))
    trigram_counts = pickle.load(open(os.path.join(save_dir, 'abstract_trigram_cnt.pkl'), 'rb'))
    forgram_counts = pickle.load(open(os.path.join(save_dir, 'abstract_forgram_cnt.pkl'), 'rb'))
    fivgram_counts = pickle.load(open(os.path.join(save_dir, 'abstract_fivgram_cnt.pkl'), 'rb'))
    aid2len = pickle.load(open(os.path.join(save_dir, 'abstract_aid2len.pkl'), 'rb'))
    print('load ngram counts cost {:.5f} s'.format(time.time() - start_time))

    paper_author_ngram_weights = []
    for i in trange(len(datas)):
        author_name, aid, pid_order, pos_list, neg_list = datas[i][0][:5]
        pid, order = pid_order.split('-')
        paper = paper_infos[pid]
        abstract = clean_abstract(paper['abstract'])
        abstract = abstract.split()
        unigrams = [w for w in abstract]
        bigrams = [tuple(abstract[i:i + 2]) for i in range(len(abstract) - 1)]
        trigrams = [tuple(abstract[i:i + 3]) for i in range(len(abstract) - 2)]
        forgrams = [tuple(abstract[i:i + 4]) for i in range(len(abstract) - 3)]
        fivgrams = [tuple(abstract[i:i + 5]) for i in range(len(abstract) - 4)]

        author_ngrams = {}

        # positive
        unigram_score = get_ngram_tfidf_score(unigrams, unigram_counts, aid, aid2len)
        bigram_score = get_ngram_tfidf_score(bigrams, bigram_counts, aid, aid2len)
        trigram_score = get_ngram_tfidf_score(trigrams, trigram_counts, aid, aid2len)
        forgram_score = get_ngram_tfidf_score(forgrams, forgram_counts, aid, aid2len)
        fivgram_score = get_ngram_tfidf_score(fivgrams, fivgram_counts, aid, aid2len)
        if aid not in author_ngrams:
            author_ngrams[aid] = [unigram_score, bigram_score, trigram_score, forgram_score, fivgram_score]

        # negative
        neg_aids = set(map(lambda x: x.split('-')[0], neg_list))
        for naid in neg_aids:
            unigram_score = get_ngram_tfidf_score(unigrams, unigram_counts, naid, aid2len)
            bigram_score = get_ngram_tfidf_score(bigrams, bigram_counts, naid, aid2len)
            trigram_score = get_ngram_tfidf_score(trigrams, trigram_counts, naid, aid2len)
            forgram_score = get_ngram_tfidf_score(forgrams, forgram_counts, naid, aid2len)
            fivgram_score = get_ngram_tfidf_score(fivgrams, fivgram_counts, naid, aid2len)
            if naid not in author_ngrams:
                author_ngrams[naid] = [unigram_score, bigram_score, trigram_score, forgram_score, fivgram_score]

        paper_author_ngram_weights.append(author_ngrams)

    return paper_author_ngram_weights


def get_unass_abstract_ngram_weights(datas, unass_paper_infos, refer_paper_infos, nameAidPid, data_type='valid'):
    start_time = time.time()
    unigram_counts = pickle.load(open(os.path.join(save_dir, 'abstract_unigram_cnt.pkl'), 'rb'))
    bigram_counts = pickle.load(open(os.path.join(save_dir, 'abstract_bigram_cnt.pkl'), 'rb'))
    trigram_counts = pickle.load(open(os.path.join(save_dir, 'abstract_trigram_cnt.pkl'), 'rb'))
    forgram_counts = pickle.load(open(os.path.join(save_dir, 'abstract_forgram_cnt.pkl'), 'rb'))
    fivgram_counts = pickle.load(open(os.path.join(save_dir, 'abstract_fivgram_cnt.pkl'), 'rb'))
    aid2len = pickle.load(open(os.path.join(save_dir, 'abstract_aid2len.pkl'), 'rb'))
    print('load ngram counts cost {:.5f} s'.format(time.time() - start_time))

    paper_author_ngram_weights = []
    for i in trange(len(datas)):
        pid_order, name = datas[i]
        pid, order = pid_order.split('-')

        paper = unass_paper_infos[pid]
        abstract = clean_abstract(paper['abstract'])
        abstract = abstract.split()
        unigrams = [w for w in abstract]
        bigrams = [tuple(abstract[i:i + 2]) for i in range(len(abstract) - 1)]
        trigrams = [tuple(abstract[i:i + 3]) for i in range(len(abstract) - 2)]
        forgrams = [tuple(abstract[i:i + 4]) for i in range(len(abstract) - 3)]
        fivgrams = [tuple(abstract[i:i + 5]) for i in range(len(abstract) - 4)]

        author_ngrams = {}
        candiAuthors = list(nameAidPid[name].keys())
        for aid in candiAuthors:
            unigram_score = get_ngram_tfidf_score(unigrams, unigram_counts, aid, aid2len)
            bigram_score = get_ngram_tfidf_score(bigrams, bigram_counts, aid, aid2len)
            trigram_score = get_ngram_tfidf_score(trigrams, trigram_counts, aid, aid2len)
            forgram_score = get_ngram_tfidf_score(forgrams, forgram_counts, aid, aid2len)
            fivgram_score = get_ngram_tfidf_score(fivgrams, fivgram_counts, aid, aid2len)
            if aid not in author_ngrams:
                author_ngrams[aid] = [unigram_score, bigram_score, trigram_score, forgram_score, fivgram_score]

        paper_author_ngram_weights.append(author_ngrams)

    return paper_author_ngram_weights


def get_train_org_ngram_weights(datas, paper_infos):
    start_time = time.time()
    ngram_counts = pickle.load(open(os.path.join(save_dir, 'org_ngram_cnt.pkl'), 'rb'))
    aid2len = pickle.load(open(os.path.join(save_dir, 'org_aid2len.pkl'), 'rb'))
    print('load ngram counts cost {:.5f} s'.format(time.time() - start_time))

    paper_author_ngram_weights = []
    for i in trange(len(datas)):
        author_name, aid, pid_order, pos_list, neg_list = datas[i][0][:5]
        pid, order = pid_order.split('-')
        paper = paper_infos[pid]
        author = paper['authors'][int(order)]
        orgs = clean_orgs(author['org'])
        if orgs == '':
            unigrams, bigrams = [], []
        else:
            unigrams = [w for w in orgs]
            bigrams = [tuple(orgs[i:i + 2]) for i in range(len(orgs) - 1)]

        author_ngrams = {}

        # positive
        ngram_scores = []
        if len(unigrams) == 0:
            ngram_scores.append(0.)
        else:
            for unigram in unigrams:
                score = get_tfidf(unigram, ngram_counts, aid, aid2len)
                ngram_scores.append(score)

        if len(bigrams) == 0:
            ngram_scores.append(0.)
        else:
            for bigram in bigrams:
                score = get_tfidf(bigram, ngram_counts, aid, aid2len)
                ngram_scores.append(score)
        ngram_scores = np.mean(ngram_scores)

        if aid not in author_ngrams:
            author_ngrams[aid] = ngram_scores

        # negative
        neg_aids = set(map(lambda x: x.split('-')[0], neg_list))
        for naid in neg_aids:
            ngram_scores = []
            if len(unigrams) == 0:
                ngram_scores.append(0.)
            else:
                for unigram in unigrams:
                    score = get_tfidf(unigram, ngram_counts, naid, aid2len)
                    ngram_scores.append(score)

            if len(bigrams) == 0:
                ngram_scores.append(0.)
            else:
                for bigram in bigrams:
                    score = get_tfidf(bigram, ngram_counts, naid, aid2len)
                    ngram_scores.append(score)
            ngram_scores = np.mean(ngram_scores)
            if naid not in author_ngrams:
                author_ngrams[naid] = ngram_scores

        paper_author_ngram_weights.append(author_ngrams)

    return paper_author_ngram_weights


def get_unass_org_ngram_weights(datas, unass_paper_infos, refer_paper_infos, nameAidPid, data_type='valid'):
    start_time = time.time()
    ngram_counts = pickle.load(open(os.path.join(save_dir, 'org_ngram_cnt.pkl'), 'rb'))
    aid2len = pickle.load(open(os.path.join(save_dir, 'org_aid2len.pkl'), 'rb'))
    print('load ngram counts cost {:.5f} s'.format(time.time() - start_time))

    paper_author_ngram_weights = []
    for i in trange(len(datas)):
        pid_order, name = datas[i]
        pid, order = pid_order.split('-')

        paper = unass_paper_infos[pid]
        author = paper['authors'][int(order)]
        orgs = clean_orgs(author['org'])
        if orgs == '':
            unigrams, bigrams = [], []
        else:
            unigrams = [w for w in orgs]
            bigrams = [tuple(orgs[i:i + 2]) for i in range(len(orgs) - 1)]

        author_ngrams = {}
        candiAuthors = list(nameAidPid[name].keys())
        for aid in candiAuthors:
            ngram_scores = []
            if len(unigrams) == 0:
                ngram_scores.append(0.)
            else:
                for unigram in unigrams:
                    score = get_tfidf(unigram, ngram_counts, aid, aid2len)
                    ngram_scores.append(score)

            if len(bigrams) == 0:
                ngram_scores.append(0.)
            else:
                for bigram in bigrams:
                    score = get_tfidf(bigram, ngram_counts, aid, aid2len)
                    ngram_scores.append(score)
            ngram_scores = np.mean(ngram_scores)

            if aid not in author_ngrams:
                author_ngrams[aid] = ngram_scores

        paper_author_ngram_weights.append(author_ngrams)

    return paper_author_ngram_weights


def get_train_keywords_ngram_weights(datas, paper_infos):
    start_time = time.time()
    unigram_counts = pickle.load(open(os.path.join(save_dir, 'keywords_unigram_cnt.pkl'), 'rb'))
    bigram_counts = pickle.load(open(os.path.join(save_dir, 'keywords_bigram_cnt.pkl'), 'rb'))
    trigram_counts = pickle.load(open(os.path.join(save_dir, 'keywords_trigram_cnt.pkl'), 'rb'))
    aid2len = pickle.load(open(os.path.join(save_dir, 'keywords_aid2len.pkl'), 'rb'))
    print('load ngram counts cost {:.5f} s'.format(time.time() - start_time))

    paper_author_ngram_weights = []
    for i in trange(len(datas)):
        author_name, aid, pid_order, pos_list, neg_list = datas[i][0][:5]
        pid, order = pid_order.split('-')
        paper = paper_infos[pid]
        keywords = paper['keywords']

        unigrams, bigrams, trigrams = [], [], []
        for keyword in keywords:
            keyword = keyword.lower().split()
            cur_unigrams = [w for w in keyword]
            cur_bigrams = [tuple(keyword[i:i + 2]) for i in range(len(keyword) - 1) if len(tuple(keyword[i:i + 2])) == 2]
            cur_trigrams = [tuple(keyword[i:i + 3]) for i in range(len(keyword) - 2) if len(tuple(keyword[i:i + 3])) == 3]
            unigrams.extend(cur_unigrams)
            bigrams.extend(cur_bigrams)
            trigrams.extend(cur_trigrams)

        author_ngrams = {}

        # positive
        unigram_score = get_ngram_tfidf_score(unigrams, unigram_counts, aid, aid2len)
        bigram_score = get_ngram_tfidf_score(bigrams, bigram_counts, aid, aid2len)
        trigram_score = get_ngram_tfidf_score(trigrams, trigram_counts, aid, aid2len)
        if aid not in author_ngrams:
            author_ngrams[aid] = [unigram_score, bigram_score, trigram_score]

        # negative
        neg_aids = set(map(lambda x: x.split('-')[0], neg_list))
        for naid in neg_aids:
            unigram_score = get_ngram_tfidf_score(unigrams, unigram_counts, naid, aid2len)
            bigram_score = get_ngram_tfidf_score(bigrams, bigram_counts, naid, aid2len)
            trigram_score = get_ngram_tfidf_score(trigrams, trigram_counts, naid, aid2len)
            if naid not in author_ngrams:
                author_ngrams[naid] = [unigram_score, bigram_score, trigram_score]

        paper_author_ngram_weights.append(author_ngrams)

    return paper_author_ngram_weights


def get_unass_keywords_ngram_weights(datas, unass_paper_infos, refer_paper_infos, nameAidPid, data_type='valid'):
    start_time = time.time()
    unigram_counts = pickle.load(open(os.path.join(save_dir, 'keywords_unigram_cnt.pkl'), 'rb'))
    bigram_counts = pickle.load(open(os.path.join(save_dir, 'keywords_bigram_cnt.pkl'), 'rb'))
    trigram_counts = pickle.load(open(os.path.join(save_dir, 'keywords_trigram_cnt.pkl'), 'rb'))
    aid2len = pickle.load(open(os.path.join(save_dir, 'keywords_aid2len.pkl'), 'rb'))
    print('load ngram counts cost {:.5f} s'.format(time.time() - start_time))

    paper_author_ngram_weights = []
    for i in trange(len(datas)):
        pid_order, name = datas[i]
        pid, order = pid_order.split('-')

        paper = unass_paper_infos[pid]
        keywords = paper['keywords']

        unigrams, bigrams, trigrams = [], [], []
        for keyword in keywords:
            keyword = keyword.lower().split()
            cur_unigrams = [w for w in keyword]
            cur_bigrams = [tuple(keyword[i:i + 2]) for i in range(len(keyword) - 1) if
                           len(tuple(keyword[i:i + 2])) == 2]
            cur_trigrams = [tuple(keyword[i:i + 3]) for i in range(len(keyword) - 2) if
                            len(tuple(keyword[i:i + 3])) == 3]
            unigrams.extend(cur_unigrams)
            bigrams.extend(cur_bigrams)
            trigrams.extend(cur_trigrams)

        author_ngrams = {}
        candiAuthors = list(nameAidPid[name].keys())
        for aid in candiAuthors:
            unigram_score = get_ngram_tfidf_score(unigrams, unigram_counts, aid, aid2len)
            bigram_score = get_ngram_tfidf_score(bigrams, bigram_counts, aid, aid2len)
            trigram_score = get_ngram_tfidf_score(trigrams, trigram_counts, aid, aid2len)
            if aid not in author_ngrams:
                author_ngrams[aid] = [unigram_score, bigram_score, trigram_score]

        paper_author_ngram_weights.append(author_ngrams)

    return paper_author_ngram_weights


def get_train_coauthors_ngram_weights(datas, paper_infos):
    start_time = time.time()
    ngram_counts = pickle.load(open(os.path.join(save_dir, 'coauthor_ngram_cnt.pkl'), 'rb'))
    aid2len = pickle.load(open(os.path.join(save_dir, 'coauthor_aid2len.pkl'), 'rb'))
    print('load ngram counts cost {:.5f} s'.format(time.time() - start_time))

    paper_author_ngram_weights = []
    for i in trange(len(datas)):
        author_name, aid, pid_order, pos_list, neg_list = datas[i][0][:5]
        pid, order = pid_order.split('-')
        paper = paper_infos[pid]
        authors = paper['authors']

        coauthors = deepcopy(authors)
        to_be_ass_author = coauthors[int(order)]
        coauthors.remove(to_be_ass_author)

        coauthors = list(map(lambda x: process_name(x['name']), coauthors))

        unigrams, bigrams = [], []
        for coauthor in coauthors:
            coauthor = coauthor.lower().split()
            cur_unigram = [w for w in coauthor]
            cur_bigram = [tuple(coauthor[i:i + 2]) for i in range(len(coauthor)) if len(tuple(coauthor[i:i + 2])) == 2]
            unigrams.extend(cur_unigram)
            bigrams.extend(cur_bigram)

        author_ngrams = {}

        # positive
        ngram_scores = []
        if len(unigrams) == 0:
            ngram_scores.append(0.)
        else:
            for unigram in unigrams:
                score = get_tfidf(unigram, ngram_counts, aid, aid2len)
                ngram_scores.append(score)

        if len(bigrams) == 0:
            ngram_scores.append(0.)
        else:
            for bigram in bigrams:
                score = get_tfidf(bigram, ngram_counts, aid, aid2len)
                ngram_scores.append(score)
        ngram_scores = np.mean(ngram_scores)

        if aid not in author_ngrams:
            author_ngrams[aid] = ngram_scores

        # negative
        neg_aids = set(map(lambda x: x.split('-')[0], neg_list))
        for naid in neg_aids:
            ngram_scores = []
            if len(unigrams) == 0:
                ngram_scores.append(0.)
            else:
                for unigram in unigrams:
                    score = get_tfidf(unigram, ngram_counts, naid, aid2len)
                    ngram_scores.append(score)

            if len(bigrams) == 0:
                ngram_scores.append(0.)
            else:
                for bigram in bigrams:
                    score = get_tfidf(bigram, ngram_counts, naid, aid2len)
                    ngram_scores.append(score)
            ngram_scores = np.mean(ngram_scores)
            if naid not in author_ngrams:
                author_ngrams[naid] = ngram_scores

        paper_author_ngram_weights.append(author_ngrams)

    return paper_author_ngram_weights


def get_unass_coauthors_ngram_weights(datas, unass_paper_infos, refer_paper_infos, nameAidPid, data_types='valid'):
    start_time = time.time()
    ngram_counts = pickle.load(open(os.path.join(save_dir, 'coauthor_ngram_cnt.pkl'), 'rb'))
    aid2len = pickle.load(open(os.path.join(save_dir, 'coauthor_aid2len.pkl'), 'rb'))
    print('load ngram counts cost {:.5f} s'.format(time.time() - start_time))

    paper_author_ngram_weights = []
    for i in trange(len(datas)):
        pid_order, name = datas[i]
        pid, order = pid_order.split('-')

        paper = unass_paper_infos[pid]
        authors = paper['authors']

        coauthors = deepcopy(authors)
        to_be_ass_author = coauthors[int(order)]
        coauthors.remove(to_be_ass_author)

        coauthors = list(map(lambda x: process_name(x['name']), coauthors))

        unigrams, bigrams = [], []
        for coauthor in coauthors:
            coauthor = coauthor.lower().split()
            cur_unigram = [w for w in coauthor]
            cur_bigram = [tuple(coauthor[i:i + 2]) for i in range(len(coauthor)) if len(tuple(coauthor[i:i + 2])) == 2]
            unigrams.extend(cur_unigram)
            bigrams.extend(cur_bigram)

        author_ngrams = {}
        candiAuthors = list(nameAidPid[name].keys())
        for aid in candiAuthors:
            ngram_scores = []
            if len(unigrams) == 0:
                ngram_scores.append(0.)
            else:
                for unigram in unigrams:
                    score = get_tfidf(unigram, ngram_counts, aid, aid2len)
                    ngram_scores.append(score)

            if len(bigrams) == 0:
                ngram_scores.append(0.)
            else:
                for bigram in bigrams:
                    score = get_tfidf(bigram, ngram_counts, aid, aid2len)
                    ngram_scores.append(score)
            ngram_scores = np.mean(ngram_scores)
            if aid not in author_ngrams:
                author_ngrams[aid] = ngram_scores

        paper_author_ngram_weights.append(author_ngrams)

    return paper_author_ngram_weights


def get_train_title_ngram_weights(datas, paper_infos):
    start_time = time.time()
    ngram_counts = pickle.load(open(os.path.join(save_dir, 'title_ngram_cnt.pkl'), 'rb'))
    aid2len = pickle.load(open(os.path.join(save_dir, 'title_aid2len.pkl'), 'rb'))
    print('load ngram counts cost {:.5f} s'.format(time.time() - start_time))

    paper_author_ngram_weights = []
    for i in trange(len(datas)):
        author_name, aid, pid_order, pos_list, neg_list = datas[i][0][:5]
        pid, order = pid_order.split('-')
        paper = paper_infos[pid]
        title = clean_abstract(paper['title'])
        title = title.split()
        unigrams = [w for w in title]
        bigrams = [tuple(title[i:i + 2]) for i in range(len(title) - 1) if len(tuple(title[i:i + 2])) == 2]

        author_ngrams = {}

        # positive
        ngram_scores = []
        for unigram in unigrams:
            score = get_tfidf(unigram, ngram_counts, aid, aid2len)
            ngram_scores.append(score)
        for bigram in bigrams:
            score = get_tfidf(bigram, ngram_counts, aid, aid2len)
            ngram_scores.append(score)
        ngram_scores = np.mean(ngram_scores)

        if aid not in author_ngrams:
            author_ngrams[aid] = ngram_scores

        # negative
        neg_aids = set(map(lambda x: x.split('-')[0], neg_list))
        for naid in neg_aids:
            ngram_scores = []
            for unigram in unigrams:
                score = get_tfidf(unigram, ngram_counts, naid, aid2len)
                ngram_scores.append(score)
            for bigram in bigrams:
                score = get_tfidf(bigram, ngram_counts, naid, aid2len)
                ngram_scores.append(score)
            ngram_scores = np.mean(ngram_scores)

            if naid not in author_ngrams:
                author_ngrams[naid] = ngram_scores
        paper_author_ngram_weights.append(author_ngrams)

    return paper_author_ngram_weights


def get_unass_title_ngram_weights(datas, unass_paper_infos, refer_paper_infos, nameAidPid, data_type='valid'):
    start_time = time.time()
    ngram_counts = pickle.load(open(os.path.join(save_dir, 'title_ngram_cnt.pkl'), 'rb'))
    aid2len = pickle.load(open(os.path.join(save_dir, 'title_aid2len.pkl'), 'rb'))
    print('load ngram counts cost {:.5f} s'.format(time.time() - start_time))

    paper_author_ngram_weights = []
    for i in trange(len(datas)):
        pid_order, name = datas[i]
        pid, order = pid_order.split('-')

        paper = unass_paper_infos[pid]
        title = clean_abstract(paper['title'])
        title = title.split()
        unigrams = [w for w in title]
        bigrams = [tuple(title[i:i + 2]) for i in range(len(title) - 1) if len(tuple(title[i:i + 2])) == 2]

        author_ngrams = {}
        candiAuthors = list(nameAidPid[name].keys())
        for aid in candiAuthors:
            ngram_scores = []
            for unigram in unigrams:
                score = get_tfidf(unigram, ngram_counts, aid, aid2len)
                ngram_scores.append(score)
            for bigram in bigrams:
                score = get_tfidf(bigram, ngram_counts, aid, aid2len)
                ngram_scores.append(score)
            ngram_scores = np.mean(ngram_scores)

            if aid not in author_ngrams:
                author_ngrams[aid] = ngram_scores

        paper_author_ngram_weights.append(author_ngrams)

    return paper_author_ngram_weights


def get_author_org_coauthor(nameAidPid, paper_infos):
    aid2orgs_coauthors = {}
    for name, name_infos in nameAidPid.items():
        for aid, pubs in name_infos.items():
            aid2orgs_coauthors[aid] = {'orgs': [], 'coauthors': []}
            for pid_order in pubs:
                pid, order = pid_order.split('-')
                authors = paper_infos[pid]['authors']

                to_be_ass_author = authors[int(order)]
                org = clean_orgs(to_be_ass_author['org'])
                if org == '':
                    continue
                aid2orgs_coauthors[aid]['orgs'].append(org)

                coauthors = deepcopy(authors)
                coauthors.remove(to_be_ass_author)
                coauthors = list(map(lambda x: process_name(x['name']), coauthors))
                aid2orgs_coauthors[aid]['coauthors'].extend(coauthors)

    for aid, org_coauthors in aid2orgs_coauthors.items():
        orgs, coauthors = org_coauthors['orgs'], org_coauthors['coauthors']
        orgs, coauthors = list(set(orgs)), list(set(coauthors))

        aid2orgs_coauthors[aid] = {'org': orgs, 'coauthors': coauthors}

    pickle.dump(aid2orgs_coauthors, open(os.path.join(save_dir, 'aid2org_coauthors.pkl'), 'wb'))


if __name__ == '__main__':
    data_dir = r'./datas/Task1'
    # official_data_dir = r'baseline/datas'
    official_data_dir = r'./datas'
    paper_vector_dir = 'resource/glove_embeddings/papers'
    save_dir = r'resource/add_features_to_official_features'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_name_infos, train_paper_infos, \
        whole_author_infos, whole_paper_infos, \
        valid_unass, valid_paper_infos, \
        test_unass, test_paper_infos = load_datas(data_dir)

    # ===================== 生成辅助文件 ========================
    
    train_author_profiles = json.load(open(os.path.join(official_data_dir, 'train_author_profile.json')))
    test_author_profiles = json.load(open(os.path.join(official_data_dir, 'test_author_profile.json')))
    train_author_profiles.update(test_author_profiles)
    
    author_infos = {}
    for name, name_infos in train_author_profiles.items():
       for aid, pubs in name_infos.items():
           author_infos[aid] = {'name': name, 'pubs': pubs}
    
    author_infos.update(whole_author_infos)
    
    paper_infos = train_paper_infos
    paper_infos.update(whole_paper_infos)
    
    get_abstract_ngram_counts(author_infos, paper_infos)
    get_keywords_ngram_counts(author_infos, paper_infos)
    get_title_ngram_counts(author_infos, paper_infos)
    """

    """
    train_nameAidPid = json.load(open(os.path.join(official_data_dir, 'train_proNameAuthorPubs.json')))
    test_nameAidPid = json.load(open(os.path.join(official_data_dir, 'test_proNameAuthorPubs.json')))
    whole_nameAidPid = json.load(open(os.path.join(official_data_dir, 'proNameAuthorPubs.json')))
    nameAidPid = train_nameAidPid
    nameAidPid.update(test_nameAidPid)
    nameAidPid.update(whole_nameAidPid)

    paper_infos = train_paper_infos
    paper_infos.update(whole_paper_infos)

    get_org_counts(nameAidPid, paper_infos)
    get_coauthor_ngram_counts(nameAidPid, paper_infos)
    get_author_org_coauthor(nameAidPid, paper_infos)


    # ============================ train and test data ===============================
    train_datas = pickle.load(open(os.path.join(official_data_dir, 'train_data.pkl'), 'rb'))
    test_datas = pickle.load(open(os.path.join(official_data_dir, 'test_data.pkl'), 'rb'))

    train_author_org_weights = get_train_author_org_weights(train_datas, train_paper_infos)
    pickle.dump(train_author_org_weights, open(os.path.join(save_dir, 'train_author_org_weights.pkl'), 'wb'))
    test_author_org_weights = get_train_author_org_weights(test_datas, train_paper_infos)
    pickle.dump(test_author_org_weights, open(os.path.join(save_dir, 'test_author_org_weights.pkl'), 'wb'))

    train_author_keywords_weights = get_train_author_keywords_weights(train_datas, train_paper_infos)
    pickle.dump(train_author_keywords_weights, open(os.path.join(save_dir, 'train_author_keywords_weights.pkl'), 'wb'))
    test_author_keywords_weights = get_train_author_keywords_weights(test_datas, train_paper_infos)
    pickle.dump(test_author_keywords_weights, open(os.path.join(save_dir, 'test_author_keywords_weights.pkl'), 'wb'))

    train_author_coauthor_weights = get_train_author_coauthor_weights(train_datas, train_paper_infos)
    pickle.dump(train_author_coauthor_weights, open(os.path.join(save_dir, 'train_author_coauthor_weights.pkl'), 'wb'))
    test_author_coauthor_weights = get_train_author_coauthor_weights(test_datas, train_paper_infos)
    pickle.dump(test_author_coauthor_weights, open(os.path.join(save_dir, 'test_author_coauthor_weights.pkl'), 'wb'))

    train_embeds_similarity = get_train_embed_similarity(train_datas, paper_vector_dir)
    pickle.dump(train_embeds_similarity, open(os.path.join(save_dir, 'train_glove_similarity.pkl'), 'wb'))
    test_embeds_similarity = get_train_embed_similarity(test_datas, paper_vector_dir)
    pickle.dump(test_embeds_similarity, open(os.path.join(save_dir, 'test_glove_similarity.pkl'), 'wb'))

    train_author_abstract_ngram_weights = get_train_abstract_ngram_weights(train_datas, train_paper_infos)
    pickle.dump(train_author_abstract_ngram_weights, open(os.path.join(save_dir, 'train_author_abstract_ngram_weights.pkl'), 'wb'))
    test_author_abstract_ngram_weights = get_train_abstract_ngram_weights(test_datas, train_paper_infos)
    pickle.dump(test_author_abstract_ngram_weights, open(os.path.join(save_dir, 'test_author_abstract_ngram_weights.pkl'), 'wb'))

    train_author_org_ngram_weights = get_train_org_ngram_weights(train_datas, train_paper_infos)
    pickle.dump(train_author_org_ngram_weights, open(os.path.join(save_dir, 'train_author_org_ngram_weights.pkl'), 'wb'))
    test_author_org_ngram_weights = get_train_org_ngram_weights(test_datas, train_paper_infos)
    pickle.dump(test_author_org_ngram_weights, open(os.path.join(save_dir, 'test_author_org_ngram_weights.pkl'), 'wb'))

    train_author_keywords_ngram_weights = get_train_keywords_ngram_weights(train_datas, train_paper_infos)
    pickle.dump(train_author_keywords_ngram_weights, open(os.path.join(save_dir, 'train_author_keywords_ngram_weights.pkl'), 'wb'))
    test_author_keywords_ngram_weights = get_train_keywords_ngram_weights(test_datas, train_paper_infos)
    pickle.dump(test_author_keywords_ngram_weights, open(os.path.join(save_dir, 'test_author_keywords_ngram_weights.pkl'), 'wb'))

    train_author_coauthors_ngram_weights = get_train_coauthors_ngram_weights(train_datas, train_paper_infos)
    pickle.dump(train_author_coauthors_ngram_weights, open(os.path.join(save_dir, 'train_author_coauthors_ngram_weights.pkl'), 'wb'))
    test_author_coauthors_ngram_weights = get_train_coauthors_ngram_weights(test_datas, train_paper_infos)
    pickle.dump(test_author_coauthors_ngram_weights, open(os.path.join(save_dir, 'test_author_coauthors_ngram_weights.pkl'), 'wb'))

    train_author_title_ngram_weights = get_train_title_ngram_weights(train_datas, train_paper_infos)
    pickle.dump(train_author_title_ngram_weights,
                open(os.path.join(save_dir, 'train_author_title_ngram_weights.pkl'), 'wb'))
    test_author_title_ngram_weights = get_train_title_ngram_weights(test_datas, train_paper_infos)
    pickle.dump(test_author_title_ngram_weights,
                open(os.path.join(save_dir, 'test_author_title_ngram_weights.pkl'), 'wb'))

    # ===================================== valid unass ==================================
    nameAidPid = json.load(open(os.path.join(official_data_dir, 'proNameAuthorPubs.json')))
    unass_datas = json.load(open(os.path.join(official_data_dir, 'valid_unassCandi.json')))
    valid_unass_author_org_weights = \
        get_whole_author_org_weights_official_recall(unass_datas, valid_paper_infos, whole_paper_infos, nameAidPid)
    pickle.dump(valid_unass_author_org_weights,
                open(os.path.join(save_dir, 'valid_unass_author_org_weights_official_recall.pkl'), 'wb'))

    valid_unass_author_keywords_weights = \
        get_whole_author_keywords_weights_official_recall(unass_datas, valid_paper_infos, whole_paper_infos, nameAidPid)
    pickle.dump(valid_unass_author_keywords_weights,
                open(os.path.join(save_dir, 'valid_unass_author_keywords_weights_official_recall.pkl'), 'wb'))

    valid_unass_author_coauthor_weights = \
        get_whole_author_coauthor_weights_official_recall(unass_datas, valid_paper_infos, whole_paper_infos, nameAidPid)
    pickle.dump(valid_unass_author_coauthor_weights,
                open(os.path.join(save_dir, 'valid_unass_author_coauthor_weights_official_recall.pkl'), 'wb'))

    valid_unass_glove_similarity = \
        get_whole_embed_similarity_official_recall(unass_datas, paper_vector_dir, nameAidPid)
    pickle.dump(valid_unass_glove_similarity,
                open(os.path.join(save_dir, 'valid_unass_glove_similarity_official_recall.pkl'), 'wb'))

    valid_unass_author_abstract_ngram_weights = \
        get_unass_abstract_ngram_weights(unass_datas, valid_paper_infos, whole_paper_infos, nameAidPid, 'valid')
    pickle.dump(valid_unass_author_abstract_ngram_weights,
                open(os.path.join(save_dir, 'valid_unass_author_abstract_ngram_weights.pkl'), 'wb'))

    valid_unass_author_org_ngram_weights = \
        get_unass_org_ngram_weights(unass_datas, valid_paper_infos, whole_paper_infos, nameAidPid, 'valid')
    pickle.dump(valid_unass_author_org_ngram_weights,
                open(os.path.join(save_dir, 'valid_unass_author_org_ngram_weights.pkl'), 'wb'))

    valid_unass_author_keywords_ngram_weights = \
        get_unass_keywords_ngram_weights(unass_datas, valid_paper_infos, whole_paper_infos, nameAidPid, 'valid')
    pickle.dump(valid_unass_author_keywords_ngram_weights,
                open(os.path.join(save_dir, 'valid_unass_author_keywords_ngram_weights.pkl'), 'wb'))

    valid_unass_author_coauthors_ngram_weights = \
        get_unass_coauthors_ngram_weights(unass_datas, valid_paper_infos, whole_paper_infos, nameAidPid, 'valid')
    pickle.dump(valid_unass_author_coauthors_ngram_weights,
                open(os.path.join(save_dir, 'valid_unass_author_coauthors_ngram_weights.pkl'), 'wb'))

    valid_unass_author_title_ngram_weights = \
        get_unass_title_ngram_weights(unass_datas, valid_paper_infos, whole_paper_infos, nameAidPid, 'valid')
    pickle.dump(valid_unass_author_title_ngram_weights,
                open(os.path.join(save_dir, 'valid_unass_author_title_ngram_weights.pkl'), 'wb'))

    # =================================== test unass ================================
    nameAidPid = json.load(open(os.path.join(official_data_dir, 'proNameAuthorPubs.json')))
    unass_datas = json.load(open(os.path.join(official_data_dir, 'test_unassCandi.json')))
    test_unass_author_org_weights = \
        get_whole_author_org_weights_official_recall(unass_datas, test_paper_infos, whole_paper_infos, nameAidPid)
    pickle.dump(test_unass_author_org_weights,
                open(os.path.join(save_dir, 'test_unass_author_org_weights_official_recall.pkl'), 'wb'))

    test_unass_author_keywords_weights = \
        get_whole_author_keywords_weights_official_recall(unass_datas, test_paper_infos, whole_paper_infos, nameAidPid)
    pickle.dump(test_unass_author_keywords_weights,
                open(os.path.join(save_dir, 'test_unass_author_keywords_weights_official_recall.pkl'), 'wb'))

    test_unass_author_coauthor_weights = \
        get_whole_author_coauthor_weights_official_recall(unass_datas, test_paper_infos, whole_paper_infos, nameAidPid)
    pickle.dump(test_unass_author_coauthor_weights,
                open(os.path.join(save_dir, 'test_unass_author_coauthor_weights_official_recall.pkl'), 'wb'))

    test_unass_glove_similarity = \
        get_whole_embed_similarity_official_recall(unass_datas, paper_vector_dir, nameAidPid, 'test')
    pickle.dump(test_unass_glove_similarity,
                open(os.path.join(save_dir, 'test_unass_glove_similarity_official_recall.pkl'), 'wb'))

    test_unass_author_abstract_ngram_weights = \
        get_unass_abstract_ngram_weights(unass_datas, test_paper_infos, whole_paper_infos, nameAidPid, 'test')
    pickle.dump(test_unass_author_abstract_ngram_weights,
                open(os.path.join(save_dir, 'test_unass_author_abstract_ngram_weights.pkl'), 'wb'))

    test_unass_author_org_ngram_weights = \
        get_unass_org_ngram_weights(unass_datas, test_paper_infos, whole_paper_infos, nameAidPid, 'test')
    pickle.dump(test_unass_author_org_ngram_weights,
                open(os.path.join(save_dir, 'test_unass_author_org_ngram_weights.pkl'), 'wb'))

    test_unass_author_keywords_ngram_weights = \
        get_unass_keywords_ngram_weights(unass_datas, test_paper_infos, whole_paper_infos, nameAidPid, 'test')
    pickle.dump(test_unass_author_keywords_ngram_weights,
                open(os.path.join(save_dir, 'test_unass_author_keywords_ngram_weights.pkl'), 'wb'))

    test_unass_author_coauthors_ngram_weights = \
        get_unass_coauthors_ngram_weights(unass_datas, test_paper_infos, whole_paper_infos, nameAidPid, 'test')
    pickle.dump(test_unass_author_coauthors_ngram_weights,
                open(os.path.join(save_dir, 'test_unass_author_coauthors_ngram_weights.pkl'), 'wb'))

    test_unass_author_title_ngram_weights = \
        get_unass_title_ngram_weights(unass_datas, test_paper_infos, whole_paper_infos, nameAidPid, 'test')
    pickle.dump(test_unass_author_title_ngram_weights,
                open(os.path.join(save_dir, 'test_unass_author_title_ngram_weights.pkl'), 'wb'))



