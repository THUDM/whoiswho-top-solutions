#!/usr/bin/python3
# -*- coding: utf-8 -*-

import json
import os
import numpy as np
import _pickle as pickle
from tqdm import tqdm, trange
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

stopword = {'at', 'based', 'in', 'of', 'for', 'on', 'and', 'to', 'an', 'using', 'with', 'the', 'by', 'we', 'be', ' is',
            'are', 'can'}


def get_keywords(paper):
    title, keywords = paper['title'], paper['keywords']
    title = title.lower().split()
    title = [w for w in title if w not in stopword]

    if (isinstance(keywords, str) and keywords.strip() == '') or \
            (isinstance(keywords, list) and len(keywords) == 0) or \
            (isinstance(keywords, list) and len(keywords) == 1 and (
                    keywords[0] == 'null' or keywords[0] == '')):
        keywords = []
    else:
        keywords = ' '.join(keywords).lower()
        keywords = keywords.split()
        keywords = [w for w in keywords if w not in stopword]

    return title + keywords


def get_author_top_keywords(nameAidPid, paper_infos):
    aid2keywords = {}
    for name, author_ids in tqdm(nameAidPid.items()):
        for aid, pids in author_ids.items():
            aid2keywords[aid] = []
            for pid in pids:
                pid = pid.split('-')
                ass_author = int(pid[1])
                pid = pid[0]

                paper = paper_infos[pid]
                cur_words = get_keywords(paper)
                cur_words = ' '.join(cur_words)
                aid2keywords[aid].append(cur_words)

    author_keywords_tfidf = {}
    for author, cur_keywords in tqdm(aid2keywords.items()):
        vectorizer = TfidfVectorizer()
        try:
            X = vectorizer.fit_transform(cur_keywords)
        except:
            author_keywords_tfidf[author] = set()
        else:
            weight = np.sum(X.toarray(), axis=0)
            word_score = list(zip(vectorizer.get_feature_names(), weight))
            word_score = sorted(word_score, key=lambda x: x[1], reverse=True)

            top_words = [w for w, s in word_score[:50]]
            author_keywords_tfidf[author] = set(top_words)
    return author_keywords_tfidf


def get_paper_keywords(paper_infos):
    paper_keywords = {}
    for pid, paper in tqdm(paper_infos.items()):
        cur_words = get_keywords(paper)
        paper_keywords[pid] = set(cur_words)
    return paper_keywords


def build_graph(nameAidPid, author_keywords, ass_paper_infos, unass_paper_infos, ass_paper_keywords, unass_paper_keywords):
    # 通过关键词来对aid和pid进行建图，关键词重叠越多，边权重越大
    graph = {}

    # ass: aid <-> pid
    for name, author_ids in tqdm(nameAidPid.items(), desc='ass aid pid'):
        for aid, pids in author_ids.items():
            if aid not in graph:
                graph[aid] = {}
            for pid in pids:
                pid = pid.split('-')
                ass_author = int(pid[1])
                pid = pid[0]
                if pid not in graph:
                    graph[pid] = {}

                cur_words = ass_paper_keywords[pid]
                num_intersec = len(cur_words & author_keywords[aid])
                if num_intersec > 2:
                    graph[aid][pid] = num_intersec
                    graph[pid][aid] = num_intersec

    # unass: aid <-> pid
    for pid, paper in tqdm(unass_paper_infos.items(), desc='unass aid pid'):
        if pid not in graph:
            graph[pid] = {}
        cur_words = unass_paper_keywords[pid]
        for aid in author_keywords:
            if aid not in graph:
                graph[aid] = {}

            num_intersec = len(cur_words & author_keywords[aid])
            if num_intersec > 2:
                graph[pid][aid] = num_intersec
                graph[aid][pid] = num_intersec

    # for i, (cur_aid, cur_keywords) in enumerate(tqdm(author_keywords.items(), desc='aid aid')):
    #     if cur_aid not in graph:
    #         graph[cur_aid] = {}
    #     for j, (nxt_aid, nxt_keywords) in enumerate(author_keywords.items()):
    #         if i >= j:
    #             continue
    #         if nxt_aid not in graph:
    #             graph[nxt_aid] = {}
    #
    #         num_intersec = len(cur_keywords & nxt_keywords)
    #         if num_intersec > 2:
    #             graph[cur_aid][nxt_aid] = num_intersec
    #             graph[nxt_aid][cur_aid] = num_intersec

    all_paper_infos = ass_paper_infos
    all_paper_infos.update(unass_paper_infos)
    all_paper_keywords = ass_paper_keywords
    all_paper_keywords.update(unass_paper_keywords)
    for i, (cur_pid, cur_paper_infos) in enumerate(tqdm(all_paper_infos.items(), desc='pid pid')):
        if cur_pid not in graph:
            graph[cur_pid] = {}
        cur_keywords = all_paper_keywords[cur_pid]

        for j, (nxt_pid, nxt_paper_infos) in enumerate(all_paper_infos.items()):
            if i >= j:
                continue
            if nxt_pid not in graph:
                graph[nxt_pid] = {}
            nxt_keywords = all_paper_keywords[nxt_pid]

            num_intersec = cur_keywords & nxt_keywords
            num_intersec = len(num_intersec)
            if num_intersec > 2:
                graph[cur_pid][nxt_pid] = num_intersec
                graph[nxt_pid][cur_pid] = num_intersec

    return graph


def save_graph(graph, path_to_save):
    with open(path_to_save, 'w') as f:
        for start_node, node_infos in tqdm(graph.items()):
            for end_node, weight in node_infos.items():
                f.write(start_node + ' ' + end_node + ' ' + str(weight) + '\n')


if __name__ == '__main__':
    # ============================== 获取作者关键词 ==============================
    train_nameAidPid = json.load(open(r'./datas/train_proNameAuthorPubs.json'))
    train_paper_infos = json.load(open(r'./datas/Task1/train/train_pub.json'))
    train_author_top_keywords = get_author_top_keywords(train_nameAidPid, train_paper_infos)
    os.makedirs("resource/node2vec", exist_ok=True)
    pickle.dump(train_author_top_keywords, open(r'./resource/node2vec/train_author_top_keywords.pkl', 'wb'))

    test_nameAidPid = json.load(open(r'./datas/test_proNameAuthorPubs.json'))
    test_paper_infos = json.load(open(r'./datas/Task1/train/train_pub.json'))
    test_author_top_keywords = get_author_top_keywords(test_nameAidPid, test_paper_infos)
    pickle.dump(test_author_top_keywords, open(r'./resource/node2vec/test_author_top_keywords.pkl', 'wb'))

    whole_nameAidPid = json.load(open(r'./datas/proNameAuthorPubs.json'))
    whole_paper_infos = json.load(open(r'./datas/Task1/cna-valid/whole_author_profiles_pub.json'))
    whole_author_top_keywords = get_author_top_keywords(whole_nameAidPid, whole_paper_infos)
    pickle.dump(whole_author_top_keywords, open(r'./resource/node2vec/whole_author_top_keywords.pkl', 'wb'))

    # =============================== 获取论文关键词 ============================
    train_paper_infos = json.load(open(r'./datas/Task1/train/train_pub.json'))
    train_paper_keywords = get_paper_keywords(train_paper_infos)
    pickle.dump(train_paper_keywords, open(r'./resource/node2vec/train_paper_keywords.pkl', 'wb'))

    whole_paper_infos = json.load(open(r'./datas/Task1/cna-valid/whole_author_profiles_pub.json'))
    whole_paper_keywords = get_paper_keywords(whole_paper_infos)
    pickle.dump(whole_paper_keywords, open(r'./resource/node2vec/whole_paper_keywords.pkl', 'wb'))

    valid_unass_paper_infos = json.load(open(r'./datas/Task1/cna-valid/cna_valid_unass_pub.json'))
    valid_unass_keywords = get_paper_keywords(valid_unass_paper_infos)
    pickle.dump(valid_unass_keywords, open(r'./resource/node2vec/valid_unass_paper_keywords.pkl', 'wb'))

    test_unass_paper_infos = json.load(open(r'./datas/Task1/cna-test/cna_test_unass_pub.json'))
    test_unass_keywords = get_paper_keywords(test_unass_paper_infos)
    pickle.dump(test_unass_keywords, open(r'./resource/node2vec/test_unass_paper_keywords.pkl', 'wb'))

    # ======================================= build train graph =====================================
    train_nameAidPid = json.load(open(r'./datas/train_proNameAuthorPubs.json'))
    train_paper_infos = json.load(open(r'./datas/Task1/train/train_pub.json'))
    train_author_ass = json.load(open(r'./datas/train_author_profile.json'))
    train_author_unass = json.load(open(r'./datas/train_author_unass.json'))
    train_author_top_keywords = pickle.load(open(r'./resource/node2vec/train_author_top_keywords.pkl', 'rb'))
    train_paper_keywords = pickle.load(open(r'./resource/node2vec/train_paper_keywords.pkl', 'rb'))

    train_ass_papers = []
    for name, name_infos in train_author_ass.items():
        for author_id, pubs in name_infos.items():
            train_ass_papers.extend(pubs)
    train_ass_papers = set(train_ass_papers)

    train_unass_papers = []
    for name, name_infos in train_author_unass.items():
        for author_id, pubs in name_infos.items():
            train_unass_papers.extend(pubs)
    train_unass_papers = set(train_unass_papers)
    print(len(train_ass_papers), len(train_unass_papers))

    ass_paper_infos = {pid: infos for pid, infos in train_paper_infos.items() if pid in train_ass_papers}
    unass_paper_infos = {pid: infos for pid, infos in train_paper_infos.items() if pid in train_unass_papers}
    train_graph = build_graph(train_nameAidPid, train_author_top_keywords, ass_paper_infos, unass_paper_infos, train_paper_keywords, train_paper_keywords)
    save_graph(train_graph, r'./resource/node2vec/train_graph.txt')

    # ======================================= build test graph =====================================
    test_nameAidPid = json.load(open(r'./datas/test_proNameAuthorPubs.json'))
    test_paper_infos = json.load(open(r'./datas/Task1/train/train_pub.json'))
    test_author_top_keywords = pickle.load(open(r'./resource/node2vec/test_author_top_keywords.pkl', 'rb'))
    test_paper_keywords = pickle.load(open(r'./resource/node2vec/train_paper_keywords.pkl', 'rb'))
    test_author_ass = json.load(open(r'./datas/test_author_profile.json'))
    test_author_unass = json.load(open(r'./datas/test_author_unass.json'))

    test_ass_papers = []
    for name, name_infos in test_author_ass.items():
        for author_id, pubs in name_infos.items():
            test_ass_papers.extend(pubs)
    test_ass_papers = set(test_ass_papers)

    test_unass_papers = []
    for name, name_infos in test_author_unass.items():
        for author_id, pubs in name_infos.items():
            test_unass_papers.extend(pubs)
    test_unass_papers = set(test_unass_papers)
    print(len(test_ass_papers), len(test_unass_papers))

    ass_paper_infos = {pid: infos for pid, infos in test_paper_infos.items() if pid in test_ass_papers}
    unass_paper_infos = {pid: infos for pid, infos in test_paper_infos.items() if pid in test_unass_papers}
    test_graph = build_graph(test_nameAidPid, test_author_top_keywords, ass_paper_infos, unass_paper_infos, test_paper_keywords, test_paper_keywords)
    save_graph(test_graph, r'./resource/node2vec/test_graph.txt')

    # ===================================== build valid unass graph =============================
    whole_nameAidPid = json.load(open(r'./datas/proNameAuthorPubs.json'))
    whole_paper_infos = json.load(open(r'./datas/Task1/cna-valid/whole_author_profiles_pub.json'))
    whole_author_top_keywords = pickle.load(open(r'./resource/node2vec/whole_author_top_keywords.pkl', 'rb'))
    whole_paper_keywords = pickle.load(open(r'./resource/node2vec/whole_paper_keywords.pkl', 'rb'))
    valid_unass_paper_infos = json.load(open(r'./datas/Task1/cna-valid/cna_valid_unass_pub.json'))
    valid_unass_paper_keywords = pickle.load(open(r'./resource/node2vec/valid_unass_paper_keywords.pkl', 'rb'))
    valid_unass_graph = build_graph(whole_nameAidPid, whole_author_top_keywords, whole_paper_infos,
                                    valid_unass_paper_infos, whole_paper_keywords, valid_unass_paper_keywords)
    save_graph(valid_unass_graph, r'./resource/node2vec/valid_unass_graph.txt')

    # ==================================== build test unass graph ==============================
    whole_nameAidPid = json.load(open(r'./datas/proNameAuthorPubs.json'))
    whole_paper_infos = json.load(open(r'./datas/Task1/cna-valid/whole_author_profiles_pub.json'))
    whole_author_top_keywords = pickle.load(open(r'./resource/node2vec/whole_author_top_keywords.pkl', 'rb'))
    whole_paper_keywords = pickle.load(open(r'./resource/node2vec/whole_paper_keywords.pkl', 'rb'))
    test_unass_paper_infos = json.load(open(r'./datas/Task1/cna-test/cna_test_unass_pub.json'))
    test_unass_paper_keywords = pickle.load(open(r'./resource/node2vec/test_unass_paper_keywords.pkl', 'rb'))
    test_unass_graph = build_graph(whole_nameAidPid, whole_author_top_keywords, whole_paper_infos,
                                    test_unass_paper_infos, whole_paper_keywords, test_unass_paper_keywords)
    save_graph(test_unass_graph, r'./resource/node2vec/test_unass_graph.txt')

    # ==================================== build complete graph =====================================
    train_nameAidPid = json.load(open(r'./datas/train_proNameAuthorPubs.json'))
    test_nameAidPid = json.load(open(r'./datas/test_proNameAuthorPubs.json'))
    whole_nameAidPid = json.load(open(r'./datas/proNameAuthorPubs.json'))
    train_paper_infos = json.load(open(r'./datas/Task1/train/train_pub.json'))
    test_paper_infos = json.load(open(r'./datas/Task1/train/train_pub.json'))
    whole_paper_infos = json.load(open(r'./datas/Task1/cna-valid/whole_author_profiles_pub.json'))

    train_author_top_keywords = pickle.load(open(r'./resource/node2vec/train_author_top_keywords.pkl', 'rb'))
    test_author_top_keywords = pickle.load(open(r'./resource/node2vec/test_author_top_keywords.pkl', 'rb'))
    whole_author_top_keywords = pickle.load(open(r'./resource/node2vec/whole_author_top_keywords.pkl', 'rb'))

    paper_keywords = pickle.load(open(r'./resource/node2vec/train_paper_keywords.pkl', 'rb'))
    whole_paper_keywords = pickle.load(open(r'./resource/node2vec/whole_paper_keywords.pkl', 'rb'))

    valid_unass_paper_infos = json.load(open(r'./datas/Task1/cna-valid/cna_valid_unass_pub.json'))
    valid_unass_paper_keywords = pickle.load(open(r'./resource/node2vec/valid_unass_paper_keywords.pkl', 'rb'))
    test_unass_paper_infos = json.load(open(r'./datas/Task1/cna-test/cna_test_unass_pub.json'))
    test_unass_paper_keywords = pickle.load(open(r'./resource/node2vec/test_unass_paper_keywords.pkl', 'rb'))

    train_author_ass = json.load(open(r'./datas/train_author_profile.json'))
    train_author_unass = json.load(open(r'./datas/train_author_unass.json'))
    test_author_ass = json.load(open(r'./datas/test_author_profile.json'))
    test_author_unass = json.load(open(r'./datas/test_author_unass.json'))

    train_ass_papers = []
    for name, name_infos in train_author_ass.items():
        for author_id, pubs in name_infos.items():
            train_ass_papers.extend(pubs)
    train_ass_papers = set(train_ass_papers)

    train_unass_papers = []
    for name, name_infos in train_author_unass.items():
        for author_id, pubs in name_infos.items():
            train_unass_papers.extend(pubs)
    train_unass_papers = set(train_unass_papers)

    train_ass_paper_infos = {pid: infos for pid, infos in train_paper_infos.items() if pid in train_ass_papers}
    train_unass_paper_infos = {pid: infos for pid, infos in train_paper_infos.items() if pid in train_unass_papers}

    test_ass_papers = []
    for name, name_infos in test_author_ass.items():
        for author_id, pubs in name_infos.items():
            test_ass_papers.extend(pubs)
    test_ass_papers = set(test_ass_papers)

    test_unass_papers = []
    for name, name_infos in test_author_unass.items():
        for author_id, pubs in name_infos.items():
            test_unass_papers.extend(pubs)
    test_unass_papers = set(test_unass_papers)

    test_ass_paper_infos = {pid: infos for pid, infos in test_paper_infos.items() if pid in test_ass_papers}
    test_unass_paper_infos = {pid: infos for pid, infos in test_paper_infos.items() if pid in test_unass_papers}

    all_nameAidPid = train_nameAidPid
    all_nameAidPid.update(test_nameAidPid)
    all_nameAidPid.update(whole_nameAidPid)

    all_author_top_keywords = train_author_top_keywords
    all_author_top_keywords.update(test_author_top_keywords)
    all_author_top_keywords.update(whole_author_top_keywords)

    ass_paper_infos = train_ass_paper_infos
    ass_paper_infos.update(test_ass_paper_infos)

    unass_paper_infos = train_unass_paper_infos
    unass_paper_infos.update(test_unass_paper_infos)
    unass_paper_infos.update(valid_unass_paper_infos)
    unass_paper_infos.update(test_unass_paper_infos)

    ass_paper_keywords = paper_keywords
    ass_paper_keywords.update(whole_paper_keywords)

    unass_paper_keywords = valid_unass_paper_keywords
    unass_paper_keywords.update(test_unass_paper_keywords)
    unass_paper_keywords.update(paper_keywords)

    graph = build_graph(all_nameAidPid, all_author_top_keywords, ass_paper_infos,
                        unass_paper_infos, ass_paper_keywords, unass_paper_keywords)

    save_graph(graph, r'./resource/node2vec/all_graph.txt')
