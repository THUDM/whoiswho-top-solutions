#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import time
import json
import joblib
import logging
import random
import _pickle as pickle
import numpy as np
import xgboost as xgb
import catboost as cab
from copy import copy, deepcopy
from tqdm import tqdm, trange
from collections import defaultdict
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity

from name_utils import clean_orgs, process_name

raw_feature_dir = r'datas'
add_feature_dir = r'resource/add_features_to_official_features'
node2vec_dir = r'resource/node2vec'


def eval_hits(predictions, test_len):
    top_k = [1, 3, 5]
    mrr = 0
    top_k_metric = np.array([0 for k in top_k])

    lengths = []
    for i in range(len(predictions)):
        tmp_pre = np.array(predictions[i])
        rank = np.argsort(-tmp_pre)

        true_index = np.where(rank == (len(tmp_pre) - 1))[0][0]

        lengths.append(len(rank))
        mrr += 1 / (true_index + 1)

        for k in range(len(top_k)):
            if true_index < top_k[k]:
                top_k_metric[k] += 1

    mrr = round(mrr / test_len, 3)
    ratio_top_k = np.array([0 for i in top_k], dtype=np.float32)

    for i in range(len(ratio_top_k)):
        ratio_top_k[i] = round(top_k_metric[i] / test_len, 3)

    return top_k, ratio_top_k, mrr


def load_data_features(data_types):
    start = time.time()
    data = pickle.load(open(os.path.join(raw_feature_dir, '{}_data.pkl'.format(data_types)), 'rb'))
    feature_data = pickle.load(open(os.path.join(raw_feature_dir, '{}_feature_data.pkl'.format(data_types)), 'rb'))
    sim_data = pickle.load(open(os.path.join(raw_feature_dir, '{}_sim_data.pkl'.format(data_types)), 'rb'))
    author_org_weights = pickle.load(
        open(os.path.join(add_feature_dir, '{}_author_org_weights.pkl'.format(data_types)), 'rb'))
    author_keywords_weights = pickle.load(
        open(os.path.join(add_feature_dir, '{}_author_keywords_weights.pkl'.format(data_types)), 'rb'))
    author_coauthor_weights = pickle.load(
        open(os.path.join(add_feature_dir, '{}_author_coauthor_weights.pkl'.format(data_types)), 'rb'))
    embed_similarity = pickle.load(
        open(os.path.join(add_feature_dir, '{}_glove_similarity.pkl'.format(data_types)), 'rb'))
    separate_node2vec_data = KeyedVectors.load_word2vec_format(
        os.path.join(node2vec_dir, "node2vec_fast_{}.bin".format(data_types)))
    complete_node2vec_data = KeyedVectors.load_word2vec_format(
        os.path.join(node2vec_dir, "node2vec_fast_all_50_walklen.bin"))
    author_abstract_ngram_weights = pickle.load(
        open(os.path.join(add_feature_dir, '{}_author_abstract_ngram_weights.pkl'.format(data_types)), 'rb'))
    author_org_ngram_weights = pickle.load(
        open(os.path.join(add_feature_dir, '{}_author_org_ngram_weights.pkl'.format(data_types)), 'rb'))
    author_keywords_ngram_weights = pickle.load(
        open(os.path.join(add_feature_dir, '{}_author_keywords_ngram_weights.pkl'.format(data_types)), 'rb'))
    author_coauthors_ngram_weights = pickle.load(
        open(os.path.join(add_feature_dir, '{}_author_coauthors_ngram_weights.pkl'.format(data_types)), 'rb'))
    author_title_ngram_weights = pickle.load(
        open(os.path.join(add_feature_dir, '{}_author_title_ngram_weights.pkl'.format(data_types)), 'rb'))
    print('load datas cost {:.7f} s'.format(time.time() - start))
    return data, feature_data, sim_data, author_org_weights, author_keywords_weights, author_coauthor_weights, \
           embed_similarity, separate_node2vec_data, complete_node2vec_data, author_abstract_ngram_weights, \
           author_org_ngram_weights, author_keywords_ngram_weights, author_coauthors_ngram_weights, \
           author_title_ngram_weights


def load_pred_features(data_types):
    start = time.time()
    author_org_weights = pickle.load(
        open(os.path.join(add_feature_dir, '{}_unass_author_org_weights_official_recall.pkl'.format(data_types)), 'rb'))
    author_keywords_weights = pickle.load(
        open(os.path.join(add_feature_dir, '{}_unass_author_keywords_weights_official_recall.pkl'.format(data_types)),
             'rb'))
    author_coauthor_weights = pickle.load(
        open(os.path.join(add_feature_dir, '{}_unass_author_coauthor_weights_official_recall.pkl'.format(data_types)),
             'rb'))
    embed_similarity = pickle.load(
        open(os.path.join(add_feature_dir, '{}_unass_glove_similarity_official_recall.pkl'.format(data_types)), 'rb'))
    author_abstract_ngram_weights = pickle.load(
        open(os.path.join(add_feature_dir, '{}_unass_author_abstract_ngram_weights.pkl'.format(data_types)), 'rb'))
    author_org_ngram_weights = pickle.load(
        open(os.path.join(add_feature_dir, '{}_unass_author_org_ngram_weights.pkl'.format(data_types)), 'rb'))
    author_keywords_ngram_weights = pickle.load(
        open(os.path.join(add_feature_dir, '{}_unass_author_keywords_ngram_weights.pkl'.format(data_types)), 'rb'))
    author_coauthors_ngram_weights = pickle.load(
        open(os.path.join(add_feature_dir, '{}_unass_author_coauthors_ngram_weights.pkl'.format(data_types)), 'rb'))
    author_title_ngram_weights = pickle.load(
        open(os.path.join(add_feature_dir, '{}_unass_author_title_ngram_weights.pkl'.format(data_types)), 'rb'))
    print('load datas cost {:.7f} s'.format(time.time() - start))
    return author_org_weights, author_keywords_weights, author_coauthor_weights, embed_similarity, \
           author_abstract_ngram_weights, author_org_ngram_weights, author_keywords_ngram_weights, \
           author_coauthors_ngram_weights, author_title_ngram_weights


def get_fold_data(train_data, train_feature_data, train_sim_data, train_author_org_weights,
                  train_author_keywords_weights, train_author_coauthor_weights, train_embed_similarity,
                  train_separate_node2vec, train_complete_node2vec, train_author_abstract_ngram_weights,
                  train_author_org_ngram_weights, train_author_keywords_ngram_weights,
                  train_author_coauthors_ngram_weights, train_author_title_ngram_weights,
                  test_data, test_feature_data, test_sim_data, test_author_org_weights,
                  test_author_keywords_weights, test_author_coauthor_weights, test_embed_similarity,
                  test_separate_node2vec, test_complete_node2vec, test_author_abstract_ngram_weights,
                  test_author_org_ngram_weights, test_author_keywords_ngram_weights,
                  test_author_coauthors_ngram_weights, test_author_title_ngram_weights,
                  data_index):
    fold_x, fold_y, fold_data = [], [], []
    separate_node2vec_features, complete_node2vec_features = [], []
    for idx in tqdm(data_index):
        if idx < len(train_data):
            datas = train_data[idx]
            features = train_feature_data[idx][0]
            sims = train_sim_data[idx]
            author_org_weights = train_author_org_weights[idx]
            author_keywords_weights = train_author_keywords_weights[idx]
            author_coauthor_weights = train_author_coauthor_weights[idx]
            embed_similarity = train_embed_similarity[idx]
            separate_node2vec_datas = train_separate_node2vec
            complete_node2vec_datas = train_complete_node2vec
            author_abstract_ngram_weights = train_author_abstract_ngram_weights[idx]
            author_org_ngram_weights = train_author_org_ngram_weights[idx]
            author_keywords_ngram_weights = train_author_keywords_ngram_weights[idx]
            author_coauthors_ngram_weights = train_author_coauthors_ngram_weights[idx]
            author_title_ngram_weights = train_author_title_ngram_weights[idx]
        else:
            idx = idx - len(train_data)
            datas = test_data[idx]
            features = test_feature_data[idx][0]
            sims = test_sim_data[idx]
            author_org_weights = test_author_org_weights[idx]
            author_keywords_weights = test_author_keywords_weights[idx]
            author_coauthor_weights = test_author_coauthor_weights[idx]
            embed_similarity = test_embed_similarity[idx]
            separate_node2vec_datas = test_separate_node2vec
            complete_node2vec_datas = test_complete_node2vec
            author_abstract_ngram_weights = test_author_abstract_ngram_weights[idx]
            author_org_ngram_weights = test_author_org_ngram_weights[idx]
            author_keywords_ngram_weights = test_author_keywords_ngram_weights[idx]
            author_coauthors_ngram_weights = test_author_coauthors_ngram_weights[idx]
            author_title_ngram_weights = test_author_title_ngram_weights[idx]

        fold_data.append(datas)
        neg_features, pos_features = features[:-1], features[-1]
        neg_sim, pos_sim = sims[:-1], sims[-1]

        data_infos, paper_pro, pos_pro, neg_pro = datas
        name, aid, pid_order, pos_list, neg_list = data_infos[:5]

        pid = pid_order.split('-')[0]
        if pid in separate_node2vec_datas:
            separate_node2vec_paper = separate_node2vec_datas[pid]
        else:
            separate_node2vec_paper = np.zeros(32)

        if pid in complete_node2vec_datas:
            complete_node2vec_paper = complete_node2vec_datas[pid]
        else:
            complete_node2vec_paper = np.zeros(32)

        j = 0
        separate_node2vec_author_neg, complete_node2vec_author_neg = [], []
        for each_i, each in enumerate(neg_pro[0]):
            if len(each) == 0:
                features = neg_sim[each_i] + neg_features[each_i] + [-1, -1, 0., 0.]  # + [-1, -1, 0.] + [0.] * 8
                if pid in separate_node2vec_datas:
                    separate_node2vec_author_neg.append(np.zeros(32))
                else:
                    separate_node2vec_author_neg.append(np.ones(32))

                if pid in complete_node2vec_datas:
                    complete_node2vec_author_neg.append(np.zeros(32))
                else:
                    complete_node2vec_author_neg.append(np.ones(32))

                features.extend([0., 0.])  # 对应abstract_ngram_weights
                features.append(0.)  # 对应org_ngram_weights
                features.extend([0., 0., 0.])  # 对应keywords_ngram_weights
                features.append(0.)  # 对应coauthor_ngram_weights
                features.append(0.)  # 对应title_ngram_weights
                fold_x.append(features)
                fold_y.append(0)
            else:
                cur_neg_list = list(map(lambda x: x.split('-')[0], neg_list[j:j + len(each)]))
                author_id = cur_neg_list[0]

                if author_id in separate_node2vec_datas:
                    separate_node2vec_author_neg.append(separate_node2vec_datas[author_id])
                elif pid not in separate_node2vec_datas:
                    separate_node2vec_author_neg.append(np.ones(32))
                else:
                    separate_node2vec_author_neg.append(np.zeros(32))

                if author_id in complete_node2vec_datas:
                    complete_node2vec_author_neg.append(complete_node2vec_datas[author_id])
                elif pid not in separate_node2vec_datas:
                    complete_node2vec_author_neg.append(np.ones(32))
                else:
                    complete_node2vec_author_neg.append(np.zeros(32))

                org_weight = author_org_weights[author_id]
                keywords_weight = author_keywords_weights[author_id]
                coauthor_weight = author_coauthor_weights[author_id]
                embed_sim = embed_similarity[author_id]
                abstract_ngram_weight = author_abstract_ngram_weights[author_id][:2]
                org_ngram_weight = author_org_ngram_weights[author_id]
                keywords_ngram_weights = author_keywords_ngram_weights[author_id]
                coauthors_ngram_weights = author_coauthors_ngram_weights[author_id]
                title_ngram_weights = author_title_ngram_weights[author_id]

                features = neg_sim[each_i] + neg_features[each_i] + \
                           [org_weight, keywords_weight, coauthor_weight, embed_sim[3]]
                features.extend(abstract_ngram_weight)
                features.append(org_ngram_weight)
                features.extend(keywords_ngram_weights)
                features.append(coauthors_ngram_weights)
                features.append(title_ngram_weights)
                fold_x.append(features)
                fold_y.append(0)

                j += len(each)

        if aid in author_org_weights:
            org_weight = author_org_weights[aid]
            keywords_weight = author_keywords_weights[aid]
            coauthor_weight = author_coauthor_weights[aid]
            embed_sim = embed_similarity[aid]
            abstract_ngram_weight = author_abstract_ngram_weights[aid][:2]
            org_ngram_weight = author_org_ngram_weights[aid]
            keywords_ngram_weights = author_keywords_ngram_weights[aid]
            coauthor_ngram_weights = author_coauthors_ngram_weights[aid]
            title_ngram_weights = author_title_ngram_weights[aid]

            features = pos_sim + pos_features + [org_weight, keywords_weight, coauthor_weight, embed_sim[3]]
            features.extend(abstract_ngram_weight)
            features.append(org_ngram_weight)
            features.extend(keywords_ngram_weights)
            features.append(coauthor_ngram_weights)
            features.append(title_ngram_weights)
        else:
            features = pos_sim + pos_features + [-1, -1, 0., 0.]
            features.extend([0., 0.])
            features.append(0.)
            features.extend([0., 0., 0.])
            features.append(0.)
            features.append(0.)

        if aid in separate_node2vec_datas:
            separate_node2vec_author_neg.append(separate_node2vec_datas[aid])
        elif pid not in separate_node2vec_datas:
            separate_node2vec_author_neg.append(np.ones(32))
        else:
            separate_node2vec_author_neg.append(np.zeros(32))

        if aid in complete_node2vec_datas:
            complete_node2vec_author_neg.append(complete_node2vec_datas[aid])
        elif pid not in separate_node2vec_datas:
            complete_node2vec_author_neg.append(np.ones(32))
        else:
            complete_node2vec_author_neg.append(np.zeros(32))

        separate_node2vec_sim = cosine_similarity(np.array([separate_node2vec_paper]),
                                                  np.array(separate_node2vec_author_neg)).T
        separate_node2vec_paper = np.array([separate_node2vec_paper]).repeat(len(separate_node2vec_author_neg), axis=0)
        separate_node2vec_features.append(np.concatenate((separate_node2vec_sim, separate_node2vec_paper), axis=1))

        complete_node2vec_sim = cosine_similarity(np.array([complete_node2vec_paper]),
                                                  np.array(complete_node2vec_author_neg)).T
        complete_node2vec_paper = np.array([complete_node2vec_paper]).repeat(len(complete_node2vec_author_neg), axis=0)
        complete_node2vec_features.append(np.concatenate((complete_node2vec_sim, complete_node2vec_paper), axis=1))

        fold_x.append(features)
        fold_y.append(1)

    # fold_x = np.array(fold_x)
    separate_node2vec_features = np.concatenate(separate_node2vec_features, axis=0)
    complete_node2vec_features = np.concatenate(complete_node2vec_features, axis=0)
    fold_x = np.concatenate((fold_x, separate_node2vec_features, complete_node2vec_features), axis=1)
    fold_y = np.array(fold_y)
    return fold_x, fold_y, fold_data


def k_fold_train(model_save_name, k=5):
    train_data, train_feature_data, train_sim_data, train_author_org_weights, train_author_keywords_weights, \
    train_author_coauthor_weights, train_embed_similarity, train_separate_node2vec, train_complete_node2vec, \
    train_author_abstract_ngram_weights, train_author_org_ngram_weights, train_author_keywords_ngram_weights, \
    train_author_coauthors_ngram_weights, train_author_title_ngram_weights = \
        load_data_features('train')

    test_data, test_feature_data, test_sim_data, test_author_org_weights, test_author_keywords_weights, \
    test_author_coauthor_weights, test_embed_similarity, test_separate_node2vec, test_complete_node2vec, \
    test_author_abstract_ngram_weights, test_author_org_ngram_weights, test_author_keywords_ngram_weights, \
    test_author_coauthors_ngram_weights, test_author_title_ngram_weights = \
        load_data_features('test')

    data_index = list(range(len(train_data)))
    print('all have {} datas'.format(len(data_index)))
    random.seed(42)
    random.shuffle(data_index)

    chunk_size = len(data_index) // k
    for fold_i in range(k):
        print('===============================================================')
        print('start training fold ', fold_i)
        start = fold_i * chunk_size
        if fold_i == k - 1:
            end = max((fold_i + 1) * chunk_size, len(data_index))
        else:
            end = (fold_i + 1) * chunk_size
        cur_test_data_index = data_index[start:end]
        cur_train_data_index = data_index[:start] + data_index[end:]

        fold_train_x, fold_train_y, _ = \
            get_fold_data(train_data, train_feature_data, train_sim_data, train_author_org_weights,
                          train_author_keywords_weights, train_author_coauthor_weights, train_embed_similarity,
                          train_separate_node2vec, train_complete_node2vec, train_author_abstract_ngram_weights,
                          train_author_org_ngram_weights, train_author_keywords_ngram_weights,
                          train_author_coauthors_ngram_weights, train_author_title_ngram_weights,
                          test_data, test_feature_data, test_sim_data, test_author_org_weights,
                          test_author_keywords_weights, test_author_coauthor_weights, test_embed_similarity,
                          test_separate_node2vec, test_complete_node2vec, test_author_abstract_ngram_weights,
                          test_author_org_ngram_weights, test_author_keywords_ngram_weights,
                          test_author_coauthors_ngram_weights, test_author_title_ngram_weights,
                          cur_train_data_index)
        fold_test_x, fold_test_y, fold_test_data = \
            get_fold_data(train_data, train_feature_data, train_sim_data, train_author_org_weights,
                          train_author_keywords_weights, train_author_coauthor_weights, train_embed_similarity,
                          train_separate_node2vec, train_complete_node2vec, train_author_abstract_ngram_weights,
                          train_author_org_ngram_weights, train_author_keywords_ngram_weights,
                          train_author_coauthors_ngram_weights, train_author_title_ngram_weights,
                          test_data, test_feature_data, test_sim_data, test_author_org_weights,
                          test_author_keywords_weights, test_author_coauthor_weights, test_embed_similarity,
                          test_separate_node2vec, test_complete_node2vec, test_author_abstract_ngram_weights,
                          test_author_org_ngram_weights, test_author_keywords_ngram_weights,
                          test_author_coauthors_ngram_weights, test_author_title_ngram_weights,
                          cur_test_data_index)

        print('train shape: ', fold_train_x.shape, fold_train_y.shape)
        print('test shape: ', fold_test_x.shape, fold_test_y.shape)

        # ===================================== Xgboost ======================================
        params = {'max_depth': 7, 'learning_rate': 0.01, 'n_estimators': 1000, 'subsample': 0.8, 'n_jobs': -1,
                  'min_child_weights': 6, 'random_state': 666, 'tree_method': 'gpu_hist', 'gpu_id': 0}
        s_t = time.time()
        xgb_model = xgb.XGBClassifier(**params)
        xgb_model.fit(fold_train_x, fold_train_y)
        os.makedirs("models/{}_fold/".format(k), exist_ok=True)
        xgb_model.save_model("models/{}_fold/xgboost_{}_fold_{}.json".format(k, model_save_name, fold_i))
        end_t = time.time()
        print("finish training. cost {:.5f} s".format(end_t - s_t))

        s_t = time.time()
        predict_scores = xgb_model.predict_proba(fold_test_x)[:, 1]
        predict_scores = np.reshape(predict_scores, (len(fold_test_data), -1))
        top_k, ratio_top_k, mrr = eval_hits(predict_scores, len(fold_test_data))
        end_t = time.time()
        print("hits@{} = {} mrr: {} cost: {}".format(top_k, ratio_top_k, mrr, round(end_t - s_t, 6)))

        # ===================================== Catboost ====================================
        s_t = time.time()
        cab_model = cab.CatBoostClassifier(iterations=1000,
                                           depth=8,
                                           learning_rate=0.01,
                                           loss_function='Logloss',
                                           eval_metric='AUC',
                                           verbose=False,
                                           random_seed=42,
                                           task_type="GPU",
                                           devices='0:1')
        cab_model.fit(fold_train_x, fold_train_y)
        cab_model.save_model("models/{}_fold/caboost_{}_fold_{}.json".format(k, model_save_name, fold_i))
        print("finish training. cost {:.5f} s".format(end_t - s_t))

        s_t = time.time()
        predict_scores = cab_model.predict_proba(fold_test_x)[:, 1]
        predict_scores = np.reshape(predict_scores, (len(fold_test_data), -1))
        top_k, ratio_top_k, mrr = eval_hits(predict_scores, len(fold_test_data))
        end_t = time.time()
        print("hits@{} = {} mrr: {} cost: {}".format(top_k, ratio_top_k, mrr, round(end_t - s_t, 6)))


def get_unass_predict_features(data_types, model_name, feature_name, k=5):
    unass_author_org_weights, unass_author_keywords_weights, unass_author_coauthor_weights, unass_embed_similarity, \
    unass_author_abstract_ngram_weights, unass_author_org_ngram_weights, unass_author_keywords_ngram_weights, \
    unass_author_coauthors_ngram_weights, unass_author_title_ngram_weights = \
        load_pred_features(data_types)
    unassCandiAuthor = pickle.load(
        open(os.path.join(raw_feature_dir, "{}_unass_CandiAuthor_add_sim.pkl".format(data_types)), 'rb'))
    featureData = pickle.load(
        open(os.path.join(raw_feature_dir, "{}_unass_featData_add_sim.pkl".format(data_types)), 'rb'))
    simData = pickle.load(
        open(os.path.join(raw_feature_dir, "{}_unass_simData_add_sim.pkl".format(data_types)), 'rb'))
    separate_node2vec_datas = KeyedVectors.load_word2vec_format(
        os.path.join(node2vec_dir, "node2vec_fast_{}_unass.bin".format(data_types)))
    complete_node2vec_datas = KeyedVectors.load_word2vec_format(
        os.path.join(node2vec_dir, "node2vec_fast_all_50_walklen.bin"))

    all_features = []
    for insNum in trange(len(featureData)):
        candiFeat = featureData[insNum][0]
        candiSim = simData[insNum]
        _, unassPid, candiAuthors = unassCandiAuthor[insNum]
        pid = unassPid.split('-')[0]
        if pid not in separate_node2vec_datas:
            separate_node2vec_paper = np.zeros(32)
        else:
            separate_node2vec_paper = separate_node2vec_datas[pid]

        if pid not in complete_node2vec_datas:
            complete_node2vec_paper = np.zeros(32)
        else:
            complete_node2vec_paper = complete_node2vec_datas[pid]

        features = []
        separate_node2vec_candi, complete_node2vec_candi = [], []
        for author_i, aid in enumerate(candiAuthors):
            org_weight = unass_author_org_weights[insNum][aid]
            keywords_weight = unass_author_keywords_weights[insNum][aid]
            coauthor_weight = unass_author_coauthor_weights[insNum][aid]
            embed_sim = unass_embed_similarity[insNum][aid]
            abstract_ngram_weights = unass_author_abstract_ngram_weights[insNum][aid][:2]
            org_ngram_weights = unass_author_org_ngram_weights[insNum][aid]
            keywords_ngram_weights = unass_author_keywords_ngram_weights[insNum][aid]
            coauthors_ngram_weights = unass_author_coauthors_ngram_weights[insNum][aid]
            title_ngram_weights = unass_author_title_ngram_weights[insNum][aid]

            cur_features = [org_weight, keywords_weight, coauthor_weight, embed_sim[3]]
            cur_features.extend(abstract_ngram_weights)
            cur_features.append(org_ngram_weights)
            cur_features.extend(keywords_ngram_weights)
            cur_features.append(coauthors_ngram_weights)
            cur_features.append(title_ngram_weights)
            features.append(cur_features)

            if aid in separate_node2vec_datas:
                separate_node2vec_author = separate_node2vec_datas[aid]
            elif pid not in separate_node2vec_datas:
                separate_node2vec_author = np.ones(32)
            else:
                separate_node2vec_author = np.zeros(32)
            separate_node2vec_candi.append(separate_node2vec_author)

            if aid in complete_node2vec_datas:
                complete_node2vec_author = complete_node2vec_datas[aid]
            elif pid not in complete_node2vec_datas:
                complete_node2vec_author = np.ones(32)
            else:
                complete_node2vec_author = np.zeros(32)
            complete_node2vec_candi.append(complete_node2vec_author)

        separate_node2vec_sim = cosine_similarity(np.array([separate_node2vec_paper]),
                                                  np.array(separate_node2vec_candi)).T
        separate_node2vec_paper = np.array([separate_node2vec_paper]).repeat(len(separate_node2vec_candi), axis=0)
        separate_node2vec_features = np.concatenate((separate_node2vec_sim, separate_node2vec_paper), axis=1)

        complete_node2vec_sim = cosine_similarity(np.array([complete_node2vec_paper]),
                                                  np.array(complete_node2vec_candi)).T
        complete_node2vec_paper = np.array([complete_node2vec_paper]).repeat(len(complete_node2vec_candi), axis=0)
        complete_node2vec_features = np.concatenate((complete_node2vec_sim, complete_node2vec_paper), axis=1)

        # print(np.shape(candiSim), np.shape(candiFeat), np.shape(features), np.shape(separate_node2vec_features), np.shape(complete_node2vec_features))
        features = np.concatenate(
            [candiSim, candiFeat, features, separate_node2vec_features, complete_node2vec_features], axis=1)
        all_features.append(features)

    all_features = np.vstack(all_features)
    xgboost_unass_predict_scores, catboost_unass_predict_scores = [], []
    for fold_i in trange(k):
        params = {'max_depth': 7, 'learning_rate': 0.01, 'n_estimators': 1000, 'subsample': 0.8, 'n_jobs': -1,
                  'min_child_weights': 6, 'random_state': 666, 'tree_method': 'gpu_hist', 'gpu_id': 0}
        xgb_model = xgb.XGBClassifier(**params)
        xgb_model.load_model("models/{}_fold/xgboost_{}_fold_{}.json".format(k, model_name, fold_i))

        predict_scores = xgb_model.predict_proba(all_features)[:, 1]
        xgboost_unass_predict_scores.append(predict_scores)

        cab_model = cab.CatBoostClassifier(iterations=1000,
                                           depth=8,
                                           learning_rate=0.01,
                                           loss_function='Logloss',
                                           eval_metric='AUC',
                                           verbose=False,
                                           random_seed=42,
                                           task_type="GPU",
                                           devices='0:1')
        cab_model.load_model("models/{}_fold/caboost_{}_fold_{}.json".format(k, model_name, fold_i))

        predict_scores = cab_model.predict_proba(all_features)[:, 1]
        catboost_unass_predict_scores.append(predict_scores)

    unass_predict_scores = xgboost_unass_predict_scores + catboost_unass_predict_scores
    unass_predict_scores = np.mean(unass_predict_scores, axis=0)

    unass_predict_features = []
    j = 0
    for insNum in trange(len(featureData)):
        _, unassPid, candiAuthors = unassCandiAuthor[insNum]
        cur_features = unass_predict_scores[j:j + len(candiAuthors)]
        unass_predict_features.append(cur_features)
        j += len(candiAuthors)

    assert len(unass_predict_features) == len(featureData)
    pickle.dump(unass_predict_features,
                open(os.path.join(add_feature_dir, '{}_unass_{}_predict_features.pkl'.format(data_types, feature_name)), 'wb'))


def fold_predict(data_types, feature_name, submit_name):
    unassCandiAuthor = pickle.load(
        open(os.path.join(raw_feature_dir, "{}_unass_CandiAuthor_add_sim.pkl".format(data_types)), 'rb'))
    featureData = pickle.load(
        open(os.path.join(raw_feature_dir, "{}_unass_featData_add_sim.pkl".format(data_types)), 'rb'))
    predictData = pickle.load(
        open(os.path.join(add_feature_dir, '{}_unass_{}_predict_features.pkl'.format(data_types, feature_name)), 'rb'))

    candiScore = defaultdict(list)
    for insNum in trange(len(featureData)):
        _, unassPid, candiAuthors = unassCandiAuthor[insNum]

        predict_score = predictData[insNum]
        rank = np.argsort(-np.array(predict_score))
        preAuthor = candiAuthors[rank[0]]
        tmp = []
        for i in rank:
            pAuthor = candiAuthors[i]
            pScore = str(predict_score[i])
            tmp.append((pAuthor, pScore))
        candiScore[unassPid] = tmp

    os.makedirs("results/task1/{}".format(data_types), exist_ok=True)
    with open("results/task1/{}/{}_resultScore.json".format(data_types, submit_name), 'w') as files:
        json.dump(candiScore, files, indent=4, ensure_ascii=False)

    count = 0
    thres = 0.7
    authorPid = defaultdict(list)
    for pid, pres in tqdm(candiScore.items()):
        preAuthor, preScore = pres[0]
        if float(preScore) >= thres:
            authorPid[preAuthor].append(pid.split('-')[0])
            count += 1
    print(count)

    with open("results/task1/{}/{}_threshold.json".format(data_types, submit_name), 'w') as files:
        json.dump(authorPid, files, indent=4, ensure_ascii=False)


def pineline(data_types, model_save_name, feature_name, submit_name, k, train=False, get_features=False):
    if train:
        k_fold_train(model_save_name, k=k)
    if get_features:
        get_unass_predict_features(data_types, model_save_name, feature_name, k=k)
    fold_predict(data_types, feature_name, submit_name)


if __name__ == '__main__':
    pineline(data_types='test',
             model_save_name='snv_cnv_ngram_weights',
             feature_name='xgb_cab_snv_cnv_ngram_weights',
             submit_name='test_xgb_cab_snv_cnv_ngram_weights',
             k=5, train=True, get_features=True)

    pineline(data_types='valid',
             model_save_name='snv_cnv_ngram_weights',
             feature_name='xgb_cab_snv_cnv_ngram_weights',
             submit_name='valid_xgb_cab_snv_cnv_ngram_weights',
             k=5, train=False, get_features=True)

