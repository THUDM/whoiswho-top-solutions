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
import lightgbm as lgb
import catboost as cab
from tqdm import tqdm, trange
from collections import defaultdict
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# data_dir = r'baseline/datas'
data_dir = r'datas'
add_feature_dir = r'resource/add_features_to_official_features'


def load_data_features(data_types):
    start = time.time()

    data = pickle.load(open(os.path.join(data_dir, '{}_data.pkl'.format(data_types)), 'rb'))
    feature_data = pickle.load(open(os.path.join(data_dir, '{}_feature_data.pkl'.format(data_types)), 'rb'))
    sim_data = pickle.load(open(os.path.join(data_dir, '{}_sim_data.pkl'.format(data_types)), 'rb'))
    author_org_weights = pickle.load(
        open(os.path.join(add_feature_dir, '{}_author_org_weights.pkl'.format(data_types)), 'rb'))
    author_keywords_weights = pickle.load(
        open(os.path.join(add_feature_dir, '{}_author_keywords_weights.pkl'.format(data_types)), 'rb'))
    author_coauthor_weights = pickle.load(
        open(os.path.join(add_feature_dir, '{}_author_coauthor_weights.pkl'.format(data_types)), 'rb'))
    embed_similarity = pickle.load(
        open(os.path.join(add_feature_dir, '{}_glove_similarity.pkl'.format(data_types)), 'rb'))
    print('load datas cost {:.7f} s'.format(time.time() - start))
    return data, feature_data, sim_data, author_org_weights, author_keywords_weights, author_coauthor_weights, embed_similarity


def load_pred_features(data_types):
    start = time.time()
    author_org_weights = pickle.load(
        open(os.path.join(add_feature_dir, '{}_unass_author_org_weights_official_recall.pkl'.format(data_types)), 'rb'))
    author_keywords_weights = pickle.load(
        open(os.path.join(add_feature_dir, '{}_unass_author_keywords_weights_official_recall.pkl'.format(data_types)), 'rb'))
    author_coauthor_weights = pickle.load(
        open(os.path.join(add_feature_dir, '{}_unass_author_coauthor_weights_official_recall.pkl'.format(data_types)), 'rb'))
    embed_similarity = pickle.load(
        open(os.path.join(add_feature_dir, '{}_unass_glove_similarity_official_recall.pkl'.format(data_types)), 'rb'))
    print('load datas cost {:.7f} s'.format(time.time() - start))
    return author_org_weights, author_keywords_weights, author_coauthor_weights, embed_similarity


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
        mrr += 1/(true_index +1)

        for k in range(len(top_k)):
            if true_index < top_k[k]:
                top_k_metric[k] += 1

    mrr = round(mrr/test_len, 3)
    ratio_top_k = np.array([0 for i in top_k], dtype = np.float32)

    for i in range(len(ratio_top_k)):
        ratio_top_k[i] = round(top_k_metric[i] / test_len, 3)

    return top_k, ratio_top_k, mrr


def fscore(predictions, test_datas):
    author_preds = {}
    candi_author_ids = []
    for i in range(len(test_datas)):
        data_infos, _, _, negs = test_datas[i]
        pos_author_id, neg_author_id = data_infos[1], data_infos[-1]
        if pos_author_id not in author_preds:
            author_preds[pos_author_id] = {'pred': [], 'true': []}
        author_preds[pos_author_id]['true'].append(i)

        neg_author_id = list(map(lambda x: x.split('-')[0], neg_author_id))
        valid_neg_author_id = []
        for cur_id in neg_author_id:
            if cur_id in valid_neg_author_id:
                continue
            valid_neg_author_id.append(cur_id)
            if cur_id not in author_preds:
                author_preds[cur_id] = {'pred': [], 'true': []}

        if len(valid_neg_author_id) != 19:
            valid_neg_author_id += ['0'] * (19 - len(valid_neg_author_id))

        candi_author_id = valid_neg_author_id + [pos_author_id]
        candi_author_ids.append(candi_author_id)

    for i in range(len(predictions)):
        cur_candidates = candi_author_ids[i]
        tmp_pre = np.array(predictions[i])
        rank = np.argsort(-tmp_pre)
        pred_author = cur_candidates[rank[0]]
        if pred_author not in author_preds:
            author_preds[pred_author] = {'pred': [], 'true': []}
        author_preds[pred_author]['pred'].append(i)

    precisions, recalls = [], []
    for author, infos in author_preds.items():
        truth = set(infos['true'])
        pred = set(infos['pred'])
        if len(truth) == 0 and len(pred) == 0:
            continue

        intersec = truth & pred
        precision = len(intersec) / (len(truth) + 1e-20)
        recall = len(intersec) / (len(pred) + 1e-20)
        precisions.append(precision)
        recalls.append(recall)

    precisions = np.mean(precisions)
    recalls = np.mean(recalls)
    f = 2 * precisions * recalls / (precisions + recalls + 1e-20)
    return f


def get_train_datas(datas, feature_datas, sim_datas, author_org_weights, author_keywords_weights,
                    author_coauthor_weights, embed_similarity):
    whole_x, whole_y = [], []
    for i in trange(len(datas)):
        features = feature_datas[i][0]
        sims = sim_datas[i]
        neg_features, pos_features = features[:-1], features[-1]
        neg_sim, pos_sim = sims[:-1], sims[-1]

        data_infos, paper_pro, pos_pro, neg_pro = datas[i]
        name, aid, pid_order, pos_list, neg_list = data_infos[:5]

        j = 0
        for each_i, each in enumerate(neg_pro[0]):
            if len(each) == 0:
                features = neg_sim[each_i] + neg_features[each_i] + [-1, -1, 0., 0.]  #+ [-1, -1, 0.] + [0.] * 8
                whole_x.append(features)
                whole_y.append(0)
            else:
                cur_neg_list = list(map(lambda x: x.split('-')[0], neg_list[j:j + len(each)]))
                author_id = cur_neg_list[0]

                org_weight = author_org_weights[i][author_id]
                keywords_weight = author_keywords_weights[i][author_id]
                coauthor_weight = author_coauthor_weights[i][author_id]
                embed_sim = embed_similarity[i][author_id]

                features = neg_sim[each_i] + neg_features[each_i] + [org_weight, keywords_weight, coauthor_weight, embed_sim[3]] # + [org_weight, keywords_weight, coauthor_weight] + embed_sim

                whole_x.append(features)
                whole_y.append(0)

                j += len(each)

        if aid in author_org_weights[i]:
            org_weight = author_org_weights[i][aid]
            keywords_weight = author_keywords_weights[i][aid]
            coauthor_weight = author_coauthor_weights[i][aid]
            embed_sim = embed_similarity[i][aid]
            features = pos_sim + pos_features + [org_weight, keywords_weight, coauthor_weight, embed_sim[3]]  # + [org_weight, keywords_weight, coauthor_weight] + embed_sim
        else:
            features = pos_sim + pos_features + [-1, -1, 0., 0.]  #+ [-1, -1, 0.] + [0.] * 8
        whole_x.append(features)
        whole_y.append(1)

    whole_x = np.array(whole_x)
    whole_y = np.array(whole_y)

    return whole_x, whole_y


def train():
    train_data, train_feature_data, train_sim_data, train_author_org_weights, train_author_keywords_weights,\
        train_author_coauthor_weights, train_embed_similarity = load_data_features('train')

    train_x, train_y = get_train_datas(train_data, train_feature_data, train_sim_data, train_author_org_weights,
                                       train_author_keywords_weights, train_author_coauthor_weights,
                                       train_embed_similarity)
    print(train_x.shape)
    print(train_y.shape)

    st = time.time()
    params = {'max_depth': 7, 'learning_rate': 0.01, 'n_estimators': 1000, 'subsample': 0.8, 'n_jobs': -1,
              'min_child_weights': 6, 'random_state': 666, 'tree_method': 'gpu_hist', 'gpu_id': 0}
    xgb_model = xgb.XGBClassifier(**params)
    xgb_model.fit(train_x, train_y)
    os.makedirs("models/", exist_ok=True)
    xgb_model.save_model("models/xgboost.json")

    # cab_model = cab.CatBoostClassifier(iterations=1000,
    #                                     depth=8,
    #                                     learning_rate=0.01,
    #                                     loss_function='Logloss',
    #                                     eval_metric='AUC',
    #                                     verbose=False,
    #                                     random_seed=42,
    #                                     task_type="GPU",
    #                                     devices='0:1')
    # cab_model.fit(train_x, train_y)
    # cab_model.save_model("models/caboost.json")

    print("Train Complete! Cost: %.6f" % (time.time() - st))


def test():
    test_data, test_feature_data, test_sim_data, test_author_org_weights, test_author_keywords_weights, \
        test_author_coauthor_weights, test_embed_similarity = load_data_features('test')

    test_x, test_y = get_train_datas(test_data, test_feature_data, test_sim_data, test_author_org_weights,
                                     test_author_keywords_weights, test_author_coauthor_weights,
                                     test_embed_similarity)
    print(test_x.shape)
    print(test_y.shape)

    params = {'max_depth': 7, 'learning_rate': 0.01, 'n_estimators': 1000, 'subsample': 0.8, 'n_jobs': -1,
              'min_child_weights': 6, 'random_state': 666, 'tree_method': 'gpu_hist', 'gpu_id': 0}
    xgb_model = xgb.XGBClassifier(**params)
    xgb_model.load_model("models/xgboost.json")

    # cab_model = cab.CatBoostClassifier(iterations=1000,
    #                                     depth=8,
    #                                     learning_rate=0.01,
    #                                     loss_function='Logloss',
    #                                     eval_metric='AUC',
    #                                     verbose=False,
    #                                     random_seed=42,
    #                                     task_type="GPU",
    #                                     devices='0:1')
    # cab_model.load_model("models/caboost.json")

    s_t = time.time()
    predict_scores = xgb_model.predict_proba(test_x)[:, 1]
    predict_scores = np.reshape(predict_scores, (len(test_data), -1))
    fs = fscore(predict_scores, test_data)
    top_k, ratio_top_k, mrr = eval_hits(predict_scores, len(test_data))
    end_t = time.time()
    print('fscore: {}'.format(fs))
    print("hits@{} = {} mrr: {} cost: {}".format(top_k, ratio_top_k, mrr, round(end_t - s_t, 6)))


def predict(data_types):
    unass_author_org_weights, unass_author_keywords_weights, unass_author_coauthor_weights, unass_embed_similarity = \
        load_pred_features(data_types)

    data_dir = 'datas'
    unassCandiAuthor = pickle.load(open(os.path.join(data_dir, "{}_unass_CandiAuthor_add_sim.pkl".format(data_types)), 'rb'))
    featureData = pickle.load(open(os.path.join(data_dir, "{}_unass_featData_add_sim.pkl".format(data_types)), 'rb'))
    simData = pickle.load(open(os.path.join(data_dir, "{}_unass_simData_add_sim.pkl".format(data_types)), 'rb'))

    # load model
    print("load model.")
    params = {'max_depth': 7, 'learning_rate': 0.01, 'n_estimators': 1000, 'subsample': 0.8, 'n_jobs': -1,
              'min_child_weights': 6, 'random_state': 666, 'tree_method': 'gpu_hist', 'gpu_id': 0}
    xgb_model = xgb.XGBClassifier(**params)
    xgb_model.load_model("models/xgboost.json")

    # cab_model = cab.CatBoostClassifier(iterations=1000,
    #                                     depth=8,
    #                                     learning_rate=0.01,
    #                                     loss_function='Logloss',
    #                                     eval_metric='AUC',
    #                                     verbose=False,
    #                                     random_seed=42,
    #                                     task_type="GPU",
    #                                     devices='0:1')
    # cab_model.load_model("models/caboost.json")

    all_features = []
    for insNum in trange(len(featureData)):
        candiFeat = featureData[insNum][0]
        candiSim = simData[insNum]
        _, unassPid, candiAuthors = unassCandiAuthor[insNum]

        features = []
        for author_i, aid in enumerate(candiAuthors):
            org_weight = unass_author_org_weights[insNum][aid]
            keywords_weight = unass_author_keywords_weights[insNum][aid]
            coauthor_weight = unass_author_coauthor_weights[insNum][aid]
            embed_sim = unass_embed_similarity[insNum][aid]
            # cur_features = [org_weight, keywords_weight, coauthor_weight] + embed_sim
            cur_features = [org_weight, keywords_weight, coauthor_weight, embed_sim[3]]
            features.append(cur_features)

        features = np.concatenate([candiSim, candiFeat, features], axis=1)
        # features = np.array(candiFeat)
        all_features.append(features)

    all_features = np.vstack(all_features)

    authorUnass = defaultdict(list)
    candiScore = defaultdict(list)
    Score = xgb_model.predict_proba(np.array(all_features))[:, 1]
    j = 0
    for insNum in trange(len(featureData)):
        _, unassPid, candiAuthors = unassCandiAuthor[insNum]

        tmpScore = Score[j:j+len(candiAuthors)]
        assert len(tmpScore) == len(candiAuthors)
        rank = np.argsort(-np.array(tmpScore))
        # print(rank)
        preAuthor = candiAuthors[rank[0]]
        # print("Paper: %s Pre: %s Score: %.6f"%(unassPid, preAuthor, tmpScore[rank[0]]))
        authorUnass[preAuthor].append(unassPid.split('-')[0])
        tmp = []
        for i in rank:
            pAuthor = candiAuthors[i]
            pScore = str(tmpScore[i])
            tmp.append((pAuthor, pScore))
        candiScore[unassPid] = tmp
        j += len(candiAuthors)

    out_dir = "results/task1/{}".format(data_types)
    os.makedirs(out_dir, exist_ok=True)
    with open("results/task1/{}/xgboost.json".format(data_types), 'w') as files:
        json.dump(authorUnass, files, indent=4, ensure_ascii=False)

    with open("results/task1/{}/xgboost_resultScore.json".format(data_types), 'w') as files:
        json.dump(candiScore, files, indent=4, ensure_ascii=False)


def postprocess(data_types):
    with open("results/task1/{}/xgboost_resultScore.json".format(data_types), 'r') as files:
        preScore = json.load(files)

    print(len(preScore))
    count = 0
    thres = 0.8
    authorPid = defaultdict(list)
    for pid, pres in tqdm(preScore.items()):
        preAuthor, preScore = pres[0]
        if float(preScore) >= thres:
            authorPid[preAuthor].append(pid.split('-')[0])
            count += 1
    with open("results/task1/{}/xgboost_threshold.json".format(data_types), 'w') as files:
        json.dump(authorPid, files, indent=4, ensure_ascii=False)
    print(count)


if __name__ == '__main__':
    train()
    test()
    predict('test')
    postprocess('test')
