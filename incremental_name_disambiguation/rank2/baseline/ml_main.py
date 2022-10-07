# import sys
# sys.path.append("./")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import pickle
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import os
import time
import numpy as np
from data_process import raw_data
# from semantic.model import bertEmbeddingLayer, matchingModel, learning2Rank
from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
# from catboost import CatBoostClassifier
# from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
# from feature_model import l2rModel
import logging
import random
import json
from whole_config import configs
from character.feature_process import featureGeneration
from tqdm import tqdm
from time import time
torch.backends.cudnn.benchmark = True

def eval_hits(predictions, test_len):
    top_k = [1, 3, 5]
    mrr = 0
    top_k_metric = np.array([0 for k in top_k])
    # print(predictions)
    predictions = np.array(predictions).reshape((test_len, configs["test_neg_sample"] + 1))
    # print(predictions)
    # print("predict: ", predictions.shape)
    lengths = []
    for i in range(len(predictions)):
        tmp_pre = np.array(predictions[i])
        rank = np.argsort(-tmp_pre)
        # print(len(tmp_pre))
        true_index = np.where(rank == (len(tmp_pre) - 1))[0][0]
        # true_index = np.where(rank == 0)[0][0]
        # if(len(rank) == 2):
            # print(rank)
            # print("total: {} true: {}".format(len(predictions[i]), true_index))
        lengths.append(len(rank))
        mrr += 1/(true_index +1)
        # top_k:[1, 5, 10, 50]
        for k in range(len(top_k)):
            if true_index < top_k[k]:
                top_k_metric[k] += 1

    mrr = round(mrr/test_len, 3)
    ratio_top_k = np.array([0 for i in top_k], dtype = np.float32)

    for i in range(len(ratio_top_k)):
        ratio_top_k[i] = round(top_k_metric[i] / test_len, 3)

    # print("hits@{} = {} mrr: {}".format(top_k, ratio_top_k, mrr))
    # print(np.mean(lengths))
    return top_k, ratio_top_k, mrr


def intergrate(whole_sim, feature):
    # mean_sim = list(np.mean(np.array(each_sim), 0))
    # whole_sim.extend(mean_sim)
    whole_sim.extend(feature)
    feature = whole_sim
    return feature


def fscore(predictions, test_datas):
    author_preds = {}
    candi_author_ids = []
    for i in range(len(test_datas)):
        data_infos, _, _, negs = test_datas[i]
        pos_author_id, neg_author_id = data_infos[-2], data_infos[-1]
        pos_author_id = list(set(list(map(lambda x: x.split('-')[0], pos_author_id))))[0]
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

    predictions = np.array(predictions).reshape((len(predictions), configs["test_neg_sample"] + 1))
    for i in range(len(predictions)):
        cur_candidates = candi_author_ids[i]
        tmp_pre = np.array(predictions[i])
        rank = np.argsort(-tmp_pre)
        # true_index = np.where(rank == (len(tmp_pre) - 1))[0][0]
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


if __name__ == "__main__":

    # # Training with the prepared data
    data_dir = "./compData/"
    with open(data_dir + "prepared_train_data_1.pkl", 'rb') as files:
        total_train_data = pickle.load(files)
    with open(data_dir + "prepared_test_data_1.pkl", 'rb') as files:
        total_test_data = pickle.load(files)

    whole_train_x = []
    whole_train_y = []
    for batch_num in tqdm(range(len(total_train_data))):
        batch_data = total_train_data[batch_num]
        batch_pos_score = []
        batch_neg_score = []
        # random.shuffle(batch_data)
        #  = generate_embedings(embedding_model, batch_data)
        # print(len(batch_data))
        for ins_num in range(len(batch_data)):
            instance = batch_data[ins_num]
            pos_data, neg_data_list = instance
            pos_whole_sim, pos_feature = pos_data
            # print(pos_whole_sim.size(), pos_feature.size())
            pos_whole_sim = list(pos_whole_sim.cpu().numpy())
            pos_feature = list(pos_feature.cpu().numpy()[0])
            # pos_total_feature =
            # print(pos_whole_sim)
            # exit()
            # print(np.array(pos_whole_sim).shape, np.array(pos_each_sim).shape, len(pos_feature))
            whole_feature = intergrate(pos_whole_sim, pos_feature)
            # print(len(whole_feature))
            # exit()
            whole_train_x.append(whole_feature)
            whole_train_y.append(1)
            for each in neg_data_list:
                each_whole_sim, each_feature = each
                # print(each_whole_sim.size(), each_feature.size())
                each_whole_sim = list(each_whole_sim.cpu().numpy())
                each_feature = list(each_feature.cpu().numpy())

                whole_feature = intergrate(each_whole_sim,each_feature)
                whole_train_x.append(whole_feature)
                whole_train_y.append(0)

    whole_train_x = np.array(whole_train_x)
    whole_train_y = np.array(whole_train_y)
    print(whole_train_x.shape)
    print(whole_train_y.shape)

    # whole_test_x = []
    # whole_test_y = []
    # whole_test_x = np.array(whole_test_x)
    # whole_test_y = np.array(whole_test_y)
    # print(whole_test_x.shape)
    # print(whole_test_y.shape)
    st = time()
    xg_model = XGBClassifier(
        max_depth=7, learning_rate=0.01, n_estimators=1000, subsample=0.8,
        n_jobs=-1, min_child_weight=6, random_state=666, tree_method='gpu_hist', gpu_id='0'
        )
    xg_model.fit(whole_train_x, whole_train_y)
    # print("")
    xg_model.save_model("xgboost_add_sim.json")
    print("Train Complete! Cost: %.6f"%(time() - st))

    print("load model.")
    xg_model = XGBClassifier(
        max_depth=7, learning_rate=0.01, n_estimators=1000, subsample=0.8,
        n_jobs=-1, min_child_weight=6, random_state=666
        )
    xg_model.load_model("xgboost_all.json")
    s_t = time()
    total_matching_score = []
    for ins_num in tqdm(range(len(total_test_data))):
        tmp_matching_score = []
        instance = total_test_data[ins_num]
        pos_data, neg_data_list = instance  
        pos_whole_sim, pos_feature = pos_data
        # pos_total_feature = 
        
        pos_whole_sim, pos_feature = pos_data
        pos_whole_sim = list(pos_whole_sim.cpu().numpy())
        pos_feature = list(pos_feature.cpu().numpy()[0])
        # pos_total_feature =
        # print(pos_whole_sim)
        # exit()
        whole_feature = intergrate(pos_whole_sim, pos_feature)
        pos_score = xg_model.predict_proba(np.array(whole_feature)[np.newaxis, :])[:, 1]
        for each in neg_data_list:
            each_whole_sim, each_feature = each

            each_whole_sim = list(each_whole_sim.cpu().numpy())
            each_feature = list(each_feature.cpu().numpy())
            
            whole_feature = intergrate(each_whole_sim, each_feature)
            neg_score = xg_model.predict_proba(np.array(whole_feature)[np.newaxis, :])[:, 1]     
            tmp_matching_score.append(neg_score)
        tmp_matching_score.append(pos_score)
        total_matching_score.append(tmp_matching_score)

    test_datas = pickle.load(open(r'./datas/test_data.pkl', 'rb'))
    fs = fscore(total_matching_score, test_datas)
    top_k, ratio_top_k, mrr = eval_hits(total_matching_score, len(total_matching_score))
    end_t = time()
    print('fscore: {}'.format(fs))
    print("hits@{} = {} mrr: {} cost: {}".format(top_k, ratio_top_k, mrr, round(end_t - s_t, 6)))