import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import os
import torch
import pickle
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import os
import time
import numpy as np
from data_process import raw_data
from semantic.model import bertEmbeddingLayer, matchingModel, learning2Rank 
import logging
import random
import json
from whole_config import configs
from character.feature_process import featureGeneration
from tqdm import tqdm
from cogdl.oag import oagbert
torch.backends.cudnn.benchmark = True


random.seed(42)
np.random.seed(42)

# def get_batch_emb(info):
#     # nonlocal model
#     ins_sent_emb = []
#     for i, ins in enumerate(info):
#         ins = ins[0]
#         input_ids, input_masks, token_type_ids, masked_lm_labels, position_ids, position_ids_second, masked_positions, num_spans = ins
#         _, output_encoder = embedding_model(
#             # torch.LongTensor(input_ids).unsqueeze(0).cuda(),
#             # torch.LongTensor(token_type_ids).unsqueeze(0).cuda(),
#             # torch.LongTensor(input_masks).unsqueeze(0).cuda(),
#             # torch.LongTensor(position_ids).unsqueeze(0).cuda(),
#             # torch.LongTensor(position_ids_second).unsqueeze(0).cuda()
#             torch.LongTensor(input_ids).unsqueeze(0).to(bert_device),
#             torch.LongTensor(token_type_ids).unsqueeze(0).to(bert_device),
#             torch.LongTensor(input_masks).unsqueeze(0).to(bert_device),
#             torch.LongTensor(position_ids).unsqueeze(0).to(bert_device),
#             torch.LongTensor(position_ids_second).unsqueeze(0).to(bert_device)
#             )
#         ins_sent_emb.append(output_encoder)
#     # print(pooled_output.size())
#     ins_sent_emb = torch.cat(ins_sent_emb)
#
#     return ins_sent_emb


def get_batch_emb(info):
    batch_input_ids, batch_token_type_ids, batch_input_masks, batch_position_ids, batch_position_ids_seconds = [], [], [], [], []
    for i, ins in enumerate(info):
        ins = ins[0]
        try:
            input_ids, input_masks, token_type_ids, masked_lm_labels, position_ids, position_ids_second, masked_positions, num_spans = ins
        except:
            input_ids, input_masks, token_type_ids, masked_lm_labels, position_ids, position_ids_second, masked_positions, num_spans = [1], [1], [0], [-1], [0], [0], [], 1
        batch_input_ids.append(input_ids)
        batch_token_type_ids.append(token_type_ids)
        batch_input_masks.append(input_masks)
        batch_position_ids.append(position_ids)
        batch_position_ids_seconds.append(position_ids_second)

    if len(batch_input_ids) == 0:
        batch_input_ids = [[1]]
        batch_token_type_ids = [[0]]
        batch_input_masks = [[1]]
        batch_position_ids = [[0]]
        batch_position_ids_seconds = [[0]]

    max_len = max(map(len, batch_input_ids))
    batch_input_ids = list(map(lambda x: x + [0] * (max_len - len(x)), batch_input_ids))
    batch_token_type_ids = list(map(lambda x: x + [0] * (max_len - len(x)), batch_token_type_ids))
    batch_input_masks = list(map(lambda x: x + [0] * (max_len - len(x)), batch_input_masks))
    batch_position_ids = list(map(lambda x: x + [0] * (max_len - len(x)), batch_position_ids))
    batch_position_ids_seconds = list(map(lambda x: x + [0] * (max_len - len(x)), batch_position_ids_seconds))
    _, output_encoder = embedding_model(
        torch.LongTensor(batch_input_ids).to(bert_device),
        torch.LongTensor(batch_token_type_ids).to(bert_device),
        torch.LongTensor(batch_input_masks).to(bert_device),
        torch.LongTensor(batch_position_ids).to(bert_device),
        torch.LongTensor(batch_position_ids_seconds).to(bert_device)
        )

    return output_encoder


def cluster_data2torch(embed_ins, feature_ins, device):
    # feature based
    pos_feature = torch.tensor(feature_ins[-1:]).to(device, non_blocking=True)
    neg_feature = torch.tensor(feature_ins[:-1]).to(device, non_blocking=True)

    paper_pro, pos_pro, neg_pro = embed_ins
    # paper_pro[0]        [((input_ids, mask, ...), )]
    # paper_pro[0][0]     ((input_ids, mask, ...), )
    # paper_pro[0][0][0]  (input_ids, mask, ...)

    # pos_pro[0]        [((input_ids, mask, ...), ), ((input_ids, mask, ...), ), ...]
    # pos_pro[0][0]     ((input_ids, mask, ...), )
    # pos_pro[0][0][0]  (input_ids, mask, ...)

    # neg_pro[0]            [[(input_ids, mask, ...), ), ((input_ids, mask, ...), ), ...]]  # 负采样作者数
    # neg_pro[0][0]         [(input_ids, mask, ...), ), ((input_ids, mask, ...), ), ...]
    # neg_pro[0][0][0]      ((input_ids, mask, ...), )
    # neg_pro[0][0][0][0]   (input_ids, mask, ...)

    paper_embedding = get_batch_emb(paper_pro[0])
    pos_per_embedding = get_batch_emb(pos_pro[0])
    neg_per_embedding_list = []
    for each in neg_pro[0]:
        per_embed = get_batch_emb(each)
        neg_per_embedding_list.append(per_embed)

    return paper_embedding, pos_per_embedding, neg_per_embedding_list, pos_feature, neg_feature


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
        _, paper_pro, pos_pro, neg_pro = ins
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
    # output_dir = "/home/chenbo/oagbert_code/saved/"
    saved_dir = "./compData/"
    os.makedirs(saved_dir, exist_ok = True)
    _, bertModel = oagbert("path_to_oagbert/oagbert-v2-sim")

    # extract raw_data
    data_generation = raw_data(bertModel)

    # generate_character_feature
    gen_character_features = featureGeneration()
    
    # print("Generate {} round data".format(round_num))
    print("Generate data")
    start = time.time()
    train_ins = data_generation.generate_train_data(configs["train_ins"])
    train_data = data_generation.multi_thread_processed_training_data(train_ins)
    # train_data = data_generation.processed_training_data(train_ins)
    mid = time.time()
    print("Training data cost: ", round(mid - start, 6))
    # exit()
    # gen_features.multi_thread_processed_training_data()
    train_feature_data, train_embedding_data = extract_features(train_data)
    # train_feature_data = gen_character_features.process_data(train_feature_data)
    train_feature_data = gen_character_features.multi_process_data(train_feature_data)
    end_time = time.time()
    print("Generate train feature: ", round(end_time - start, 6))

    # pickle.dump(train_data, open(r'./datas/train_data_all.pkl', 'wb'))
    pickle.dump(train_data, open(r'./datas/train_data.pkl', 'wb'))
    pickle.dump(train_embedding_data, open(r'./datas/train_embedding_data_all.pkl', 'wb'))
    pickle.dump(train_feature_data, open(r'./datas/train_feature_data.pkl', 'wb'))
    # pickle.dump(train_feature_data, open(r'./baseline/datas/train_feature_data.pkl', 'wb'))
    # start = time.time()
    # train_data = pickle.load(open(r'./datas/train_data.pkl', 'rb'))
    # train_embedding_data = pickle.load(open(r'./datas/train_embedding_data.pkl', 'rb'))
    # train_feature_data = pickle.load(open(r'./datas/train_feature_data.pkl', 'rb'))
    # print("Load train feature: ", round(time.time() - start, 6))

    s_t = time.time()
    test_ins = data_generation.generate_test_data(configs["test_ins"])
    test_data = data_generation.multi_thread_processed_training_data(test_ins)
    # test_data = data_generation.processed_training_data(test_ins)
    end = time.time()
    print("Testing data cost: ", round(end - s_t, 6))

    test_feature_data, test_embedding_data = extract_features(test_data)
    # test_feature_data = gen_character_features.process_data(test_feature_data)
    test_feature_data = gen_character_features.multi_process_data(test_feature_data)
    end_time = time.time()
    print("Generate test feature: ", round(end_time - s_t, 6))

    pickle.dump(test_data, open(r'./datas/test_data.pkl', 'wb'))
    pickle.dump(test_embedding_data, open(r'./datas/test_embedding_data.pkl', 'wb'))
    pickle.dump(test_feature_data, open(r'./datas/test_feature_data.pkl', 'wb'))
    # start = time.time()
    # test_data = pickle.load(open(r'./datas/test_data.pkl', 'rb'))
    # test_embedding_data = pickle.load(open(r'./datas/test_embedding_data.pkl', 'rb'))
    # test_feature_data = pickle.load(open(r'./datas/test_feature_data.pkl', 'rb'))
    # print("Load test feature: ", round(time.time() - start, 6))

    batch_train_embedding_data, batch_train_feature_data= generate_data_batch(train_embedding_data, train_feature_data, configs["local_accum_step"])
    end = time.time()
    # print("#batch, train: {} | embed-{} feature-{} batch-{} cost: {:.6f}".format(len(train_data), len(train_embedding_data), len(train_feature_data), len(batch_train_embedding_data), round(end-start, 6)))
    print("#batch, train: {} | embed-{} feature-{} batch-{} test: embed-{} feature-{} cost: {:.6f}".format(len(train_data), len(train_embedding_data), len(train_feature_data), len(batch_train_embedding_data), len(test_embedding_data), len(test_feature_data), round(end-start, 6)))

    # global bert_device
    # bert_device = torch.device("cuda:0")
    # # bert_device = 'cpu'
    #
    # # generate_embedding_feature
    # embedding_model = bertEmbeddingLayer(bertModel)
    # embedding_model.to(bert_device)
    # embedding_model.eval()
    #
    # matching_model = matchingModel(bert_device)
    # matching_model.to(bert_device)
    # # matching_model.train()
    #
    #
    # # transfer training data to GPU
    # torch_train_data = []
    # torch_test_data = []
    # # train_pid2ratio = []
    # with torch.no_grad():
    #     for batch_num in tqdm(range(len(batch_train_embedding_data))):
    #         tmp_data = []
    #         # tmp_pid2ratio = []
    #         batch_embed_data = batch_train_embedding_data[batch_num]
    #         batch_feature_data = batch_train_feature_data[batch_num]
    #         #  = generate_embedings(embedding_model, batch_data)
    #         for ins_num in range(len(batch_embed_data)):
    #             embed_ins = batch_embed_data[ins_num]
    #             # pids = batch_feature_data[ins_num][0]
    #             feature_ins = batch_feature_data[ins_num][0]
    #             # ratio = batch_feature_data[ins_num][2]
    #             # tmp_pid2ratio.append((pids, ratio))
    #             # tmp = cluster_data2torch(embed_ins, feature_ins, bert_device)
    #             tmp = cluster_data2torch(embed_ins, feature_ins, bert_device)
    #             tmp_data.append(tmp)
    #         torch_train_data.append(tmp_data)
    #         # pid2ratio.append(tmp_pid2ratio)


    # transfer testing data to GPU

    # coauthor_ratio = []

    # with torch.no_grad():
    #     for ins_num in tqdm(range(len(test_embedding_data))):
    #         embed_ins = test_embedding_data[ins_num]
    #         feature_ins = test_feature_data[ins_num][0]
    #         # ratio = test_feature_data[ins_num][1]
    #         tmp = cluster_data2torch(embed_ins, feature_ins, bert_device)
    #         torch_test_data.append(tmp)

    # # Prepared Data
    # s_time = time.time()
    # total_train_data = []
    # # total_generate_feature_data = []
    # for batch_num in tqdm(range(len(torch_train_data))):
    #     batch_data = torch_train_data[batch_num]
    #     batch_train_data = []
    #     # batch_generate_feature_data = []
    #     # random.shuffle(batch_data)
    #     #  = generate_embedings(embedding_model, batch_data)
    #     for ins_num in range(len(batch_data)):
    #         instance = batch_data[ins_num]
    #         paper_embedding, pos_per_embedding, neg_per_embedding_list, pos_feature, neg_feature = instance
    #
    #         whole_sim  = matching_model(paper_embedding, pos_per_embedding)
    #         # pos_score = l2r(whole_sim, each_sim)
    #         pos_data = (whole_sim, pos_feature)
    #         neg_data_list =[]
    #         for (each_embed, each_feature) in zip(neg_per_embedding_list, neg_feature):
    #             whole_sim = matching_model(paper_embedding, each_embed)
    #             # neg_score = l2r(whole_sim, each_sim)
    #             neg_data_list.append((whole_sim,each_feature))
    #         batch_train_data.append((pos_data, neg_data_list))
    #     total_train_data.append(batch_train_data)
    # # exit()
    # with open(saved_dir + "prepared_train_data_1" + ".pkl", 'wb') as files:
    #     pickle.dump(total_train_data, files)

    # total_test_data = []
    # for test_ins_num in tqdm(range(len(torch_test_data))):
    #     # tmp_matching_score = []
    #     instance = torch_test_data[test_ins_num]
    #     paper_embedding, pos_per_embedding, neg_per_embedding_list, pos_feature, neg_feature = instance
    #
    #     whole_sim  = matching_model(paper_embedding, pos_per_embedding)
    #     pos_data = (whole_sim, pos_feature)
    #     neg_data_list = []
    #     for (each_embed, each_feature) in zip(neg_per_embedding_list, neg_feature):
    #         whole_sim = matching_model(paper_embedding, each_embed)
    #         neg_data_list.append((whole_sim, each_feature))
    #     total_test_data.append((pos_data, neg_data_list))
    #
    # with open(saved_dir + "prepared_test_data_" + str(round_num+1) + ".pkl", 'wb') as files:
    #     pickle.dump(total_test_data, files)
    # ---------------------------------------------------
