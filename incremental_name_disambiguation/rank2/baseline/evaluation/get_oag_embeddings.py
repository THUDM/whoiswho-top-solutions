import sys
import os
sys.path.append('..')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import os
import torch
import pickle
import numpy as np
from incremental_name_disambiguation.rank2.baseline.semantic.model import bertEmbeddingLayer, matchingModel, learning2Rank
from incremental_name_disambiguation.rank2.baseline.character.feature_process import featureGeneration
import logging
import random
import json
from tqdm import tqdm, trange
from cogdl.oag import oagbert
torch.backends.cudnn.benchmark = True

maxPapers = 256


def get_batch_emb(info, embed_model, bert_device):
    batch_input_ids, batch_token_type_ids, batch_input_masks, batch_position_ids, batch_position_ids_seconds = [], [], [], [], []
    for i, ins in enumerate(info):
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
    _, output_encoder = embed_model(
        torch.LongTensor(batch_input_ids).to(bert_device),
        torch.LongTensor(batch_token_type_ids).to(bert_device),
        torch.LongTensor(batch_input_masks).to(bert_device),
        torch.LongTensor(batch_position_ids).to(bert_device),
        torch.LongTensor(batch_position_ids_seconds).to(bert_device)
        )

    return output_encoder


def getPaperAtter(pids, pubDict):
    split_info = pids.split('-')
    pid = str(split_info[0])
    author_index = int(split_info[1])
    papers_attr = pubDict[pid]
    name_info = set()
    org_str = ""
    keywords_info = set()
    try:
        title = papers_attr["title"].strip().lower()
    except:
        title = ""

    try:
        venue = papers_attr["venue"].strip().lower()
    except:
        venue = ""
    try:
        abstract = papers_attr["abstract"]
    except:
        abstract = ""

    try:
        keywords = papers_attr["keywords"]
    except:
        keywords = []

    for ins in keywords:
        keywords_info.add(ins.strip().lower())

    paper_authors = papers_attr["authors"]
    for ins_author_index in range(len(paper_authors)):
        ins_author = paper_authors[ins_author_index]
        if (ins_author_index == author_index):
            try:
                orgnizations = ins_author["org"].strip().lower()
            except:
                orgnizations = ""

            if (orgnizations.strip().lower() != ""):
                org_str = orgnizations
        else:
            try:
                name = ins_author["name"].strip().lower()
            except:
                name = ""
            if (name != ""):
                name_info.add(name)
    keywords_info = list(keywords_info)
    keywords_str = " ".join(keywords_info).strip()
    return (name_info, org_str, venue, keywords_str, title, keywords_info, abstract)


def tokenizer(pid, paper_infos, bertTokenizer):
    name_info, org_str, venue, keywords_str, title, keywords_info, abs_info = getPaperAtter(pid, paper_infos)
    bert_token = bertTokenizer.build_inputs(title=title, abstract=abs_info, venue=venue, authors=name_info,
                                            concepts=keywords_info, affiliations=org_str)
    return bert_token


def get_oag_embedding(paper_tokens, embed_model, batch_size, bert_device):
    paper_ids, tokens = list(zip(*paper_tokens.items()))
    paper_embeds = []
    chunk = int(len(tokens) // batch_size) + 1
    with torch.no_grad():
        for j in tqdm(range(chunk), desc='get embedding'):
            cur_tokens = tokens[j * batch_size:(j + 1) * batch_size]
            cur_embeds = get_batch_emb(cur_tokens, embed_model, bert_device)
            cur_embeds = cur_embeds.detach().cpu().numpy().tolist()
            paper_embeds.extend(cur_embeds)

    assert len(paper_embeds) == len(paper_ids)

    paper_oag_embeds = {}
    for pid, embeds in zip(paper_ids, paper_embeds):
        paper_oag_embeds[pid] = np.array(embeds)
    return paper_oag_embeds

    
def get_feat_sim_data(unassCandi, nameAidPid, unass_paper_infos, ass_paper_infos, unass_embeddings, ass_embeddings, bert_device):
    tmpFeature = []
    tmpMatching = []
    tmpCandi = []
    for insIndex in tqdm(range(len(unassCandi))):
        unassPid, candiName = unassCandi[insIndex]
        unassAttr = getPaperAtter(unassPid, unass_paper_infos)[:5]
        candiAuthors = list(nameAidPid[candiName].keys())
        paper_embeds = unass_embeddings[unassPid]
        paper_embeds = torch.tensor([paper_embeds]).to(bert_device)

        tmpCandiAuthor = []
        tmpFeat = []
        tmpSim = []
        for each in candiAuthors:
            totalPubs = nameAidPid[candiName][each]
            # samplePubs = random.sample(totalPubs, min(len(totalPubs), maxPapers))
            candiAttrList, candiEmbList = [], []
            for insPub in totalPubs:
                candiAttrList.append(getPaperAtter(insPub, ass_paper_infos)[:5])
                candiEmbList.append(ass_embeddings[insPub.split('-')[0]])

            if len(candiEmbList) == 0:
                whole_sim = [0.] * 41
            else:
                candiEmbList = torch.tensor(candiEmbList).to(bert_device)
                whole_sim = matching_model(paper_embeds, candiEmbList)
                whole_sim = whole_sim.detach().cpu().numpy().tolist()
            tmpSim.append(whole_sim)
            tmpFeat.append((unassAttr, candiAttrList))
            tmpCandiAuthor.append(each)

        tmpFeature.append((insIndex, tmpFeat))
        tmpMatching.append(tmpSim)
        tmpCandi.append((insIndex, unassPid, tmpCandiAuthor))
    return tmpFeature, tmpMatching, tmpCandi


def get_sim_data(embedding_data):
    sim_data = []
    for ins in trange(len(embedding_data)):
        embed_ins = embedding_data[ins]
        paper_pro, pos_pro, neg_pro = embed_ins
        paper_embedding = get_batch_emb(paper_pro[0], embedding_model, bert_device)
        pos_per_embedding = get_batch_emb(pos_pro[0], embedding_model, bert_device)
        pos_whole_sim = matching_model(paper_embedding, pos_per_embedding)
        pos_whole_sim = pos_whole_sim.detach().cpu().numpy().tolist()
        neg_whole_sim_list = []
        for each in neg_pro[0]:
            per_embed = get_batch_emb(each, embedding_model, bert_device)
            whole_sim = matching_model(paper_embedding, per_embed)
            whole_sim = whole_sim.detach().cpu().numpy().tolist()
            neg_whole_sim_list.append(whole_sim)

        neg_whole_sim_list.append(pos_whole_sim)
        sim_data.append(neg_whole_sim_list)
    return sim_data


if __name__ == '__main__':
    
    _, bertModel = oagbert("./path_to_oagbert/oagbert-v2-sim")

    bert_device = torch.device("cuda:0")
    embedding_model = bertEmbeddingLayer(bertModel)
    embedding_model.to(bert_device)
    embedding_model.eval()

    matching_model = matchingModel(bert_device)
    matching_model.to(bert_device)

    genFeatures = featureGeneration()

    # ============================== get oag embedding for each dataset ============================
    # whole oag embedding
    dataDir = "./datas/Task1/cna-valid/"
    nameAidPid = json.load(open(r'./datas/proNameAuthorPubs.json',encoding='utf-8'))
    whole_paper_infos = json.load(open(dataDir + "whole_author_profiles_pub.json", 'r',encoding='utf-8'))
    paper_tokens = {}
    for name, author_ids in tqdm(nameAidPid.items(), desc='tokenize'):
        for aid, pids in author_ids.items():
            for paper_ids in pids:
                pid = paper_ids.split('-')[0]
                bert_token = tokenizer(paper_ids, whole_paper_infos, bertModel)
                if pid not in paper_tokens:
                    paper_tokens[pid] = bert_token

    whole_oag_embeddings = get_oag_embedding(paper_tokens, embedding_model, 128, bert_device)
    pickle.dump(whole_oag_embeddings, open('./datas/whole_oag_embeddings.pkl', 'wb'))

    # valid unass oag embedding
    dataDir = "./datas/Task1/cna-valid/"
    unass_pid_infos = json.load(open(dataDir + "cna_valid_unass.json",encoding='utf-8'))
    unass_paper_infos = json.load(open(dataDir + "cna_valid_unass_pub.json",encoding='utf-8'))
    paper_tokens = {}
    for pid in tqdm(unass_pid_infos):
        bert_token = tokenizer(pid, unass_paper_infos, bertModel)
        if pid not in paper_tokens:
            paper_tokens[pid] = bert_token

    valid_unass_oag_embeddings = get_oag_embedding(paper_tokens, embedding_model, 128, bert_device)
    pickle.dump(valid_unass_oag_embeddings, open('./datas/valid_unass_oag_embeddings.pkl', 'wb'))

    # test unass oag embedding
    dataDir = "./datas/Task1/cna-test/"
    unass_pid_infos = json.load(open(dataDir + "cna_test_unass.json",encoding='utf-8'))
    unass_paper_infos = json.load(open(dataDir + "cna_test_unass_pub.json",encoding='utf-8'))
    paper_tokens = {}
    for pid in tqdm(unass_pid_infos):
        bert_token = tokenizer(pid, unass_paper_infos, bertModel)
        if pid not in paper_tokens:
            paper_tokens[pid] = bert_token

    test_unass_oag_embeddings = get_oag_embedding(paper_tokens, embedding_model, 128, bert_device)
    pickle.dump(test_unass_oag_embeddings, open('./datas/test_unass_oag_embeddings.pkl', 'wb'))

    # train oag embedding
    dataDir = "./datas/Task1/train/"
    nameAidPid = json.load(open(r'./datas/train_proNameAuthorPubs.json',encoding='utf-8'))
    train_paper_infos = json.load(open(dataDir + "train_pub.json", 'r',encoding='utf-8'))
    paper_tokens = {}
    for name, author_ids in tqdm(nameAidPid.items(), desc='tokenize'):
        for aid, pids in author_ids.items():
            for paper_ids in pids:
                pid = paper_ids.split('-')[0]
                bert_token = tokenizer(paper_ids, train_paper_infos, bertModel)
                if pid not in paper_tokens:
                    paper_tokens[pid] = bert_token

    train_oag_embeddings = get_oag_embedding(paper_tokens, embedding_model, 128, bert_device)
    pickle.dump(train_oag_embeddings, open('./datas/train_oag_embeddings.pkl', 'wb'))

    # test oag embedding
    dataDir = "./datas/Task1/train/"
    nameAidPid = json.load(open(r'./datas/test_proNameAuthorPubs.json'))
    test_paper_infos = json.load(open(dataDir + "train_pub.json", 'r',encoding='utf-8'))
    paper_tokens = {}
    for name, author_ids in tqdm(nameAidPid.items(), desc='tokenize'):
        for aid, pids in author_ids.items():
            for paper_ids in pids:
                pid = paper_ids.split('-')[0]
                bert_token = tokenizer(paper_ids, test_paper_infos, bertModel)
                if pid not in paper_tokens:
                    paper_tokens[pid] = bert_token

    test_oag_embeddings = get_oag_embedding(paper_tokens, embedding_model, 128, bert_device)
    pickle.dump(test_oag_embeddings, open('./datas/test_oag_embeddings.pkl', 'wb'))

    # ===================== get features and oag similarity for each dataset ===========================
    # valid unass
    dataDir = "./datas/Task1/cna-valid/"
    nameAidPid = json.load(open(r'./datas/proNameAuthorPubs.json',encoding='utf-8'))
    whole_paper_infos = json.load(open(dataDir + "whole_author_profiles_pub.json", 'r',encoding='utf-8'))
    unass_pid_infos = json.load(open(dataDir + "cna_valid_unass.json",encoding='utf-8'))
    unass_paper_infos = json.load(open(dataDir + "cna_valid_unass_pub.json",encoding='utf-8'))
    unassCandi = json.load(open(r'./datas/valid_unassCandi.json',encoding='utf-8'))
    unass_embeddings = pickle.load(open('./datas/valid_unass_oag_embeddings.pkl', 'rb'))
    ass_embeddings = pickle.load(open(r'./datas/whole_oag_embeddings.pkl', 'rb'))
    rawFeatData, simData, unassCandiAuthor = get_feat_sim_data(unassCandi, nameAidPid, unass_paper_infos,
                                                               whole_paper_infos, unass_embeddings, ass_embeddings,
                                                               bert_device)
    # featureData = genFeatures.process_data(rawFeatData)
    featureData = genFeatures.multi_process_data(rawFeatData)

    pickle.dump(unassCandiAuthor, open("./datas/valid_unass_CandiAuthor_add_sim.pkl", 'wb'))
    pickle.dump(featureData, open("./datas/valid_unass_featData_add_sim.pkl", 'wb'))
    pickle.dump(simData, open('./datas/valid_unass_simData_add_sim.pkl', 'wb'))

    # test unass
    dataDir = "./datas/Task1/cna-test/"
    nameAidPid = json.load(open(r'./datas/proNameAuthorPubs.json',encoding='utf-8'))
    whole_paper_infos = json.load(open("./datas/Task1/cna-valid/" + "whole_author_profiles_pub.json", 'r',encoding='utf-8'))
    unass_pid_infos = json.load(open(dataDir + "cna_test_unass.json",encoding='utf-8'))
    unass_paper_infos = json.load(open(dataDir + "cna_test_unass_pub.json",encoding='utf-8'))
    unassCandi = json.load(open(r'./datas/test_unassCandi.json',encoding='utf-8'))
    unass_embeddings = pickle.load(open('./datas/test_unass_oag_embeddings.pkl', 'rb'))
    ass_embeddings = pickle.load(open(r'./datas/whole_oag_embeddings.pkl', 'rb'))
    rawFeatData, simData, unassCandiAuthor = get_feat_sim_data(unassCandi, nameAidPid, unass_paper_infos,
                                                               whole_paper_infos, unass_embeddings, ass_embeddings,
                                                               bert_device)
    # featureData = genFeatures.process_data(rawFeatData)
    featureData = genFeatures.multi_process_data(rawFeatData)

    pickle.dump(unassCandiAuthor, open("./datas/test_unass_CandiAuthor_add_sim.pkl", 'wb'))
    pickle.dump(featureData, open("./datas/test_unass_featData_add_sim.pkl", 'wb'))
    pickle.dump(simData, open('./datas/test_unass_simData_add_sim.pkl', 'wb'))
    
    # train
    train_embedding_data = pickle.load(open(r'./datas/train_embedding_data_all.pkl', 'rb'))
    train_sim_data = get_sim_data(train_embedding_data)
    pickle.dump(train_sim_data, open(r'./datas/train_sim_data.pkl', 'wb'))

    # test
    test_embedding_data = pickle.load(open(r'./datas/test_embedding_data.pkl', 'rb'))
    test_sim_data = get_sim_data(test_embedding_data)
    pickle.dump(test_sim_data, open(r'./datas/test_sim_data.pkl', 'wb'))
