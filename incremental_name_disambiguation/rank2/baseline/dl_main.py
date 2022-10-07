import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# import sys
# sys.path.append("./")
import torch
import pickle
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import os
import time
import numpy as np
from data_process import raw_data
from semantic.model import bertEmbeddingLayer, matchingModel, learning2Rank
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
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
from cogdl import oagbert

torch.backends.cudnn.benchmark = True


def cluster_data2torch(embed_ins, feature_ins, device):
    # feature based
    # print(np.array(feature_ins).shape)
    pos_feature = torch.tensor(feature_ins[-1:]).to(device, non_blocking=True)
    neg_feature = torch.tensor(feature_ins[:-1]).to(device, non_blocking=True)

    paper_pro, pos_pro, neg_pro = embed_ins
    # print(len(paper_pro))
    # print(len(pos_pro))
    # print(len(neg_pro))
    # paper
    paper_inputs = torch.tensor(paper_pro[0]).to(device, non_blocking=True)
    paper_attention_masks = torch.tensor(paper_pro[1]).to(device, non_blocking=True)
    # print(paper_semi_inputs.size())
    # print(paper_semantic_inputs.size())
    # pos author
    pos_inputs = torch.tensor(pos_pro[0]).to(device, non_blocking=True)
    pos_attention_masks = torch.tensor(pos_pro[1]).to(device, non_blocking=True)

    # print(pos_semi_inputs.size())
    # print(pos_semantic_inputs.size())
    pos_per_inputs = torch.tensor(pos_pro[2]).to(device, non_blocking=True)
    pos_per_attention_masks = torch.tensor(pos_pro[3]).to(device, non_blocking=True)

    # print(pos_per_semi_inputs.size())
    # print(pos_per_semantic_inputs.size())
    # neg_author
    neg_inputs_list = []
    neg_attention_masks_list = []
    neg_per_inputs_list = []
    neg_per_attention_masks_list = []

    for (each_per_in, each_per_masks) in zip(neg_pro[2], neg_pro[3]):
        # # print(tmp_semi_inputs.size())
        # if(len(each_per_semi_in) < max_paper_semi):
        #     padding_len = max_paper_semi - len(each_per_semi_in)
        #     padding_semi = np.zeros((padding_len, configs["train_max_semi_len"]), dtype = np.int)
        #     padding_semantic = np.zeros((padding_len, configs["train_max_semantic_len"]), dtype = np.int)

        #     each_per_semi_in.extend(padding_semi)
        #     each_per_semi_masks.extend(padding_semi)
        #     each_per_semantic_in.extend(padding_semantic)
        #     each_per_semantic_masks.extend(padding_semantic
        each_per_in = torch.tensor(each_per_in).to(device, non_blocking=True)
        each_per_masks = torch.tensor(each_per_masks).to(device, non_blocking=True)

        neg_per_inputs_list.append(each_per_in)
        neg_per_attention_masks_list.append(each_per_masks)

    neg_inputs_list = torch.tensor(neg_pro[0]).to(device, non_blocking=True)
    neg_attention_masks_list = torch.tensor(neg_pro[1]).to(device, non_blocking=True)

    return paper_inputs, paper_attention_masks, \
           pos_inputs, pos_attention_masks, pos_per_inputs, pos_per_attention_masks, \
           neg_inputs_list, neg_attention_masks_list, neg_per_inputs_list, neg_per_attention_masks_list, \
           pos_feature, neg_feature


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
        mrr += 1 / (true_index + 1)
        # top_k:[1, 5, 10, 50]
        for k in range(len(top_k)):
            if true_index < top_k[k]:
                top_k_metric[k] += 1

    mrr = round(mrr / test_len, 3)
    ratio_top_k = np.array([0 for i in top_k], dtype=np.float32)

    for i in range(len(ratio_top_k)):
        ratio_top_k[i] = round(top_k_metric[i] / test_len, 3)

    # print("hits@{} = {} mrr: {}".format(top_k, ratio_top_k, mrr))
    # print(np.mean(lengths))
    return top_k, ratio_top_k, mrr


def test_model(test_cluster):
    l2r.eval()
    # test_loss = []
    total_matching_score = []
    batch_pos_score = []
    batch_neg_score = []
    s_t = time.time()
    with torch.no_grad():
        for test_ins_num in range(len(test_cluster)):
            tmp_matching_score = []
            instance = test_cluster[test_ins_num]
            pos_data, neg_data_list = instance
            pos_whole_sim, pos_feature = pos_data
            pos_whole_sim.to(bert_device)
            # pos_each_sim.to(bert_device)
            pos_feature.to(bert_device)
            pos_score = l2r(pos_whole_sim, pos_feature[0])
            for each_data in neg_data_list:
                each_whole_sim, each_feature = each_data
                each_whole_sim.to(bert_device)
                each_feature.to(bert_device)
                # each_feature = each_feature.unsqueeze(0)
                # print(each_feature.size())
                neg_score = l2r(each_whole_sim, each_feature)
                batch_pos_score.append(pos_score)
                batch_neg_score.append(neg_score)

                tmp_matching_score.append(neg_score.item())

            tmp_matching_score.append(pos_score.item())
            total_matching_score.append(tmp_matching_score)

        batch_pos_score = torch.cat(batch_pos_score)
        batch_neg_score = torch.cat(batch_neg_score)

        marginLoss = criterion(batch_pos_score, batch_neg_score, rank_y)

    top_k, ratio_top_k, mrr = eval_hits(total_matching_score, len(test_cluster))
    end_t = time.time()
    print("paper: test_loss: {:.3f} hits@{} = {} mrr: {} cost: {}".format(marginLoss.item(), top_k, ratio_top_k, mrr,
                                                                          round(end_t - s_t, 6)))
    return ratio_top_k


def generate_data_batch(emb_data, fea_data, batch_size):
    batch_embed_data = []
    batch_fea_data = []
    assert len(emb_data) == len(fea_data)
    data_len = len(emb_data)
    for i in range(0, data_len, batch_size):
        batch_embed_data.append(emb_data[i:min(i + batch_size, data_len)])
        batch_fea_data.append(fea_data[i:min(i + batch_size, data_len)])
    return batch_embed_data, batch_fea_data


def extract_features(ori_data):
    feature_data = []
    embedding_data = []

    for ins_index in range(len(ori_data)):
        ins = ori_data[ins_index]
        paper_pro, pos_pro, neg_pro = ins
        # embedding_data
        paper_embed = paper_pro[1:]
        pos_embed = pos_pro[1:]
        neg_embed = neg_pro[1:]
        embedding_data.append((paper_embed, pos_embed, neg_embed))

        # feature_data
        paper_str = paper_pro[0]
        pos_str = pos_pro[0]
        neg_str_list = neg_pro[0]

        pos_ins = (paper_str[0], pos_str)
        tmp_neg_list = []
        for each in neg_str_list:
            neg_ins = (paper_str[0], each)
            tmp_neg_list.append(neg_ins)
        tmp_neg_list.append(pos_ins)
        # new_train_data.append(pos_ins, neg_str_list)
        feature_data.append((ins_index, tmp_neg_list))

    return feature_data, embedding_data


if __name__ == "__main__":
    output_dir = "/home/chenbo/oagbert_code/saved/"
    _, bertModel = oagbert(output_dir + "oagbert-v2-sim")
    global bert_device
    bert_device = torch.device("cuda:0")

    # # extract raw_data
    # data_generation = raw_data(bertModel)

    # # generate_embedding_feature
    # embedding_model = bertEmbeddingLayer(bertModel)
    # embedding_model.to(bert_device)
    # embedding_model.eval()

    # matching_model = matchingModel(bert_device)
    # matching_model.to(bert_device)
    # # matching_model.train()

    # # generate_character_feature
    # gen_character_features = featureGeneration()

    # learning2Rank & Machine Learning
    l2r = learning2Rank()
    l2r.to(bert_device)

    criterion = nn.MarginRankingLoss(margin=0.5)
    rank_y = torch.tensor([1.0], device=bert_device)

    optimizer = torch.optim.Adam([{'params': l2r.parameters(), 'lr': configs["train_knrm_learning_rate"]}])
    l2r.train()

    # Training with the prepared data
    data_dir = "/ssd/chenbo/compData/"
    with open(data_dir + "prepared_train_data_1.pkl", 'rb') as files:
        total_train_data = pickle.load(files)
    with open(data_dir + "prepared_test_data_1.pkl", 'rb') as files:
        total_test_data = pickle.load(files)

    # for batch_num in tqdm(range(len(total_train_data))):
    #     batch_train_data = total_train_data[batch_num]
    #     for instance in batch_train_data:
    #         pos_data, neg_data_list = instance
    #         pos_whole_sim, pos_each_sim, pos_feature = pos_data
    #         pos_whole_sim.to(bert_device)
    #         pos_each_sim.to(bert_device)
    #         pos_feature.to(bert_device)

    #         pos_score = l2r(pos_whole_sim, pos_each_sim, pos_feature)

    max_hits = 0
    min_test_loss = 10.0
    # file_name = "./l2_3_adversarial_checkpoints/"
    for epoch in range(configs["n_epoch"]):
        l2r.train()
        epoch_total_loss = []
        epoch_matching_loss = []

        batch_total_loss = []

        optimizer.zero_grad()
        s_time = time.time()
        random.shuffle(total_train_data)
        for batch_num in tqdm(range(len(total_train_data))):
            batch_data = total_train_data[batch_num]
            batch_pos_score = []
            batch_neg_score = []
            random.shuffle(batch_data)
            #  = generate_embedings(embedding_model, batch_data)
            for ins_num in range(len(batch_data)):
                instance = batch_data[ins_num]
                pos_data, neg_data_list = instance
                pos_whole_sim, pos_feature = pos_data
                pos_whole_sim.to(bert_device)
                pos_feature.to(bert_device)
                pos_score = l2r(pos_whole_sim, pos_feature[0])

                for each_data in neg_data_list:
                    each_whole_sim, each_feature = each_data
                    # print(each_whole_sim.size())
                    each_whole_sim.to(bert_device)
                    each_feature.to(bert_device)
                    # each_feature = each_feature.unsqueeze(0)
                    # print(each_feature.size())
                    neg_score = l2r(each_whole_sim, each_feature)
                    batch_pos_score.append(pos_score)
                    batch_neg_score.append(neg_score)
                    # exit()

            batch_pos_score = torch.cat(batch_pos_score)
            batch_neg_score = torch.cat(batch_neg_score)
            # batch_pos_shape = batch_pos_score.size().item()
            # batch_neg_shape = batch_neg_score.size().item()
            # assert batch_pos_shape[0] == batch_neg_score().size()[0] == (configs["local_accum_step"] * configs["train_neg_sample"])
            # print(batch_pos_score)
            # print(batch_neg_score)
            # print(batch_pos_score.size())
            # print(batch_neg_score.size())
            # exit()
            # time.sleep(3)
            marginLoss = criterion(batch_pos_score, batch_neg_score, rank_y)
            marginLoss.backward()
            optimizer.step()
            optimizer.zero_grad()
            batch_total_loss.append(marginLoss.item())
        e_t = time.time()
        epoch_loss = np.array(batch_total_loss)
        avg_epoch_loss = np.mean(epoch_loss)
        print("Epoch: {} loss: {} cost: {}".format(epoch + 1, round(avg_epoch_loss, 6), round(e_t - s_time, 6)))
        if ((epoch + 1) % configs["show_step"] == 0):
            optimizer.zero_grad()
            # test_loss = []
            # matching_score = []
            ratio_top_k = test_model(total_test_data)
            # embedding_model.eval()
            l2r.train()
            # if(ratio_top_k[0] > max_hits):
            #     max_hits = ratio_top_k[0]
            #     print("Save checkpoint!")
            #     state = {'l2r_model': l2r.state_dict()}
            #     torch.save(state, saved_file + "model_" + str(epoch))
            # shared_encoder.train()
        # print("Epoch: {} loss: {:.3f} cost: {:.3f}".format(epoch, avg_epoch_loss, time.time() - s_time))

