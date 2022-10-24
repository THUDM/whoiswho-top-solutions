import re


from gensim.models import word2vec
# from sklearn.cluster import DBSCAN
from sklearn.cluster import DBSCAN
import numpy as np
from utils import load_json, dump_json, dump_data, load_data, tanimoto, generate_pair, pairwise_evaluate, \
    MetaPathGenerator, save_relation
from sklearn.metrics.pairwise import pairwise_distances
import os
# from cogdl import oagbert
from cogdl.oag import oagbert
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import torch


def save_oagbertEmb(name_pubs_raw, name):  # 语义特征的oagbertEmbedding
    name_pubs_raw = load_json('saveName', name_pubs_raw)

    tokenizer, model = oagbert("oagbert-v2-sim")
    model.eval()

    ptext_emb = {}
    for i, pid in tqdm(enumerate(name_pubs_raw)):
        pub = name_pubs_raw[pid]

        name_info = set()
        org_str = ""
        keywords_info = set()
        try:
            title = pub["title"].strip().lower()
        except:
            title = ""

        try:
            venue = pub["venue"].strip().lower()
        except:
            venue = ""

        # try:
        #     year = int(papers_attr["year"])
        # except:
        #     year = 0

        try:
            abstract = pub["abstract"]
        except:
            abstract = ""

        try:
            keywords = pub["keywords"]
        except:
            keywords = []

        for ins in keywords:
            keywords_info.add(ins.strip().lower())

        paper_authors = pub["authors"]

        for ins_author_index in range(len(paper_authors)):
            ins_author = paper_authors[ins_author_index]
            try:
                orgnizations = ins_author["org"].strip().lower()
            except:
                orgnizations = ""

            if (orgnizations.strip().lower() != ""):
                org_str = orgnizations

            try:
                name_a = ins_author["name"].strip().lower()
            except:
                name_a = ""
            if (name_a != ""):
                name_info.add(name_a)

        # name_str = " ".join(name_info).strip()

        # org_str = " ".join(org_info).strip()
        keywords_info = list(keywords_info)
        keywords_str = " ".join(keywords_info).strip()

        semi_str = org_str + venue
        semantic_str = title + " " + keywords_str

        input_ids, input_masks, token_type_ids, masked_lm_labels, position_ids, position_ids_second, masked_positions, num_spans = model.build_inputs(
            title=title, abstract=abstract, venue=venue, authors=name_info, concepts=keywords_info,
            affiliations=org_str)
        # encode thrid paper
        _, paper_embed = model.bert.forward(
            input_ids=torch.LongTensor(input_ids).unsqueeze(0),
            token_type_ids=torch.LongTensor(token_type_ids).unsqueeze(0),
            attention_mask=torch.LongTensor(input_masks).unsqueeze(0),
            output_all_encoded_layers=False,
            checkpoint_activations=False,
            position_ids=torch.LongTensor(position_ids).unsqueeze(0),
            position_ids_second=torch.LongTensor(position_ids_second).unsqueeze(0)
        )

        ptext_emb[pid] = paper_embed.detach().numpy().squeeze()
    #  ptext_emb: key is paper id, and the value is the paper's text embedding
    dump_data(ptext_emb, 'save', name + 'ptext_emb_OAGbert_train.pkl')


##将关系保存为路径，随机游走，并训练word2vec向量
def randomWalk(rw_num, numwalks, walklength, pubs, name):
    mpg = MetaPathGenerator()
    mpg.read_data("save")

    ##论文关系表征向量
    all_embs = []
    rw_num = rw_num
    cp = set()
    for k in range(rw_num):
        mpg.generate_WMRW("save/RW_train.txt", numwalks, walklength)
        print('random walk done')
        sentences = word2vec.Text8Corpus(r'save/RW_train.txt')
        model = word2vec.Word2Vec(sentences, size=768, negative=25, min_count=1, window=10)
        #将保存的路径表示为embedding
        embs = []
        for i, pid in enumerate(pubs):
            if pid in model:
                embs.append(model[pid])
            else:
                cp.add(i)
                embs.append(np.zeros(768))
        all_embs.append(embs)
    all_embs = np.array(all_embs)
    dump_data(all_embs, 'save', name + 'walk_emb_train.pkl')
    dump_data(cp, 'save', name + 'cp_train.pkl')


def validation(rw_num=10, numwalks=5, walklength=20,  # 随机游走参数
               eps=0.16, min_samples=4,  # DBSCAN聚类参数
               # output_path="result_train.json"   #验证集生成文件
               outTh=1.5  # 判断离群点的阈值
               ):
    #load数据
    # base = '/home/chengyq/projects/whoiswho/data'
    base = "../data/"
    pubs_raw = load_json(base, "train/train_pub.json")
    name_pubs1 = load_json(base, "train/train_author.json")

    result = {}
    score = []
    #记录prec和recall值
    pairwise_precision = []
    pairwise_recall = []
    for n, name in tqdm(enumerate(name_pubs1)):
        #对每一个name（表示待消歧的作者姓名）进行处理
        ilabel = 0
        pubs = []  # all papers
        labels = []  # ground truth

        for author in name_pubs1[name]:
            iauthor_pubs = name_pubs1[name][author]
            for pub in iauthor_pubs:
                pubs.append(pub)
                labels.append(ilabel)
            ilabel += 1

        print(n, name, len(pubs))
        if len(pubs) == 0:
            result[name] = []
            score.append(0)
            continue

        ##将该名字下的论文与train_pubs中论文对应，并保存对应关系
        name_pubs_raw = {}
        for i, pid in enumerate(pubs):
            name_pubs_raw[pid] = pubs_raw[pid]

        dump_json(name_pubs_raw, 'saveName', name + '.json', indent=4)
        save_relation(name + '.json', name)

        # 提取语义特征，保存语义OAGBertembedding
        save_oagbertEmb(name + '.json', name)


        # 提取关系特征，构建异质图，并随机游走得到关系embedding
        randomWalk(rw_num, numwalks, walklength, pubs, name)

        ##load刚保存的语义OAGBertembedding，并计算语义相似性矩阵
        ptext_emb = load_data('save', name + 'ptext_emb_OAGbert_train.pkl')

        tembs = []
        for i, pid in enumerate(pubs):
            tembs.append(ptext_emb[pid])

        ##load刚保存的关系embedding，并计算关系相似性矩阵
        all_embs = load_data('save', name + 'walk_emb_train.pkl')
        sk_sim = np.zeros((len(pubs), len(pubs)))
        for k in range(rw_num):
            sk_sim = sk_sim + pairwise_distances(all_embs[k], metric="cosine")
        sk_sim = sk_sim / rw_num

        tembs = pairwise_distances(tembs, metric="cosine")
        # print(sk_sim.shape)
        # print(tembs.shape)

        # 计算最终相似性矩阵
        w = 1
        sim = (np.array(sk_sim) + w * np.array(tembs)) / (1 + w)

        ##evaluate

        # DBSCAN聚类
        pre = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed").fit_predict(sim)
        pre = np.array(pre)

        outlier = set()
        for i in range(len(pre)):
            if pre[i] == -1:
                outlier.add(i)
        # load随机游走离群点
        cp = load_data('save', name + 'cp_train.pkl')
        for i in cp:
            outlier.add(i)

        ##离群点，基于阈值的相似性匹配

        paper_pair = generate_pair(pubs, outlier)
        paper_pair1 = paper_pair.copy()

        K = len(set(pre))
        for i in range(len(pre)):
            if i not in outlier:
                continue
            j = np.argmax(paper_pair[i])
            while j in outlier:
                paper_pair[i][j] = -1
                j = np.argmax(paper_pair[i])
            if paper_pair[i][j] >= outTh:
                pre[i] = pre[j]
            else:
                pre[i] = K
                K = K + 1

        for ii, i in enumerate(outlier):
            for jj, j in enumerate(outlier):
                if jj <= ii:
                    continue
                else:
                    if paper_pair1[i][j] >= outTh:
                        pre[j] = pre[i]

        result[name] = []
        for i in set(pre):
            oneauthor = []
            for idx, j in enumerate(pre):
                if i == j:
                    oneauthor.append(pubs[idx])
            result[name].append(oneauthor)

        # 计算分数用于调参
        labels = np.array(labels)
        pre = np.array(pre)


        pairwise_precision_one, pairwise_recall_one, pairwise_f1 = pairwise_evaluate(labels, pre)
        pairwise_precision.append(pairwise_precision_one)
        pairwise_recall.append(pairwise_recall_one)
        # print (pairwise_precision, pairwise_recall, pairwise_f1)
        score.append(pairwise_f1)

        print('avg_f1:', np.mean(score))
    print('==========================================================')
    print('rw_num=%d, numwalks=%d, walklength=%d, eps=%.2f, min_samples=%d, outTh=%.2f' % (rw_num,numwalks,walklength,eps,min_samples,outTh))
    print('prec:', np.mean(pairwise_precision))
    print('recall:', np.mean(pairwise_recall))
    print('mean_score:', np.mean(score))
    # dump_json(result, "saveResult", output_path, indent =4)


if __name__ == '__main__':
    validation()
