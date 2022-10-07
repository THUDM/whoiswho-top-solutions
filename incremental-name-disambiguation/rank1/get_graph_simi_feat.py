import sys
import os
import re
import random
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
from gensim.models import word2vec
import numpy as np

from utils import load_json, save_json, load_pickle, save_pickle
from whole_config import FilePathConfig, processed_data_root, graph_feat_root, raw_data_root


gene_root = os.path.join(graph_feat_root, 'gene')
genename_root = os.path.join(graph_feat_root, 'genename')


class MetaPathGenerator:
    def __init__(self):
        self.paper_author = dict()
        self.author_paper = dict()
        self.paper_org = dict()
        self.org_paper = dict()
        self.paper_conf = dict()
        self.conf_paper = dict()

    def read_data(self, dirpath):
        temp = set()
        with open(dirpath + "/paper_org.txt", 'r', encoding='utf-8') as pafile:
            for line in pafile:
                if line == '':
                    continue
                temp.add(line)
        for line in temp:
            toks = line.strip().split("\t")
            if len(toks) == 2:
                p, a = toks[0], toks[1]
                if p not in self.paper_org:
                    self.paper_org[p] = []
                self.paper_org[p].append(a)
                if a not in self.org_paper:
                    self.org_paper[a] = []
                self.org_paper[a].append(p)
        temp.clear()

        with open(dirpath + "/paper_author.txt", encoding='utf-8') as pafile:
            for line in pafile:
                if line == '':
                    continue
                temp.add(line)
        for line in temp:
            toks = line.strip().split("\t")
            if len(toks) == 2:
                p, a = toks[0], toks[1]
                if p not in self.paper_author:
                    self.paper_author[p] = []
                self.paper_author[p].append(a)
                if a not in self.author_paper:
                    self.author_paper[a] = []
                self.author_paper[a].append(p)
        temp.clear()

        with open(dirpath + "/paper_conf.txt", encoding='utf-8') as pcfile:  # 期刊和会议信息
            for line in pcfile:
                if line == '':
                    continue
                temp.add(line)
        for line in temp:
            toks = line.strip().split("\t")
            if len(toks) == 2:
                p, a = toks[0], toks[1]
                if p not in self.paper_conf:
                    self.paper_conf[p] = []
                self.paper_conf[p].append(a)
                if a not in self.conf_paper:
                    self.conf_paper[a] = []
                self.conf_paper[a].append(p)
        temp.clear()

        print("#papers ", len(self.paper_conf))
        print("#authors", len(self.author_paper))
        print("#org_words", len(self.org_paper))
        print("#confs  ", len(self.conf_paper))

    def generate_WMRW(self, outfilename, numwalks, walklength, open_mod='w'):
        outfile = open(outfilename, open_mod)
        for paper0 in self.paper_conf:
            for j in range(0, numwalks):  # wnum walks
                paper = paper0
                outline = ""
                i = 0
                while i < walklength:
                    i = i + 1
                    if paper in self.paper_author:
                        authors = self.paper_author[paper]
                        numa = len(authors)
                        authorid = random.randrange(numa)
                        author = authors[authorid]

                        papers = self.author_paper[author]
                        nump = len(papers)
                        if nump > 1:
                            paperid = random.randrange(nump)
                            paper1 = papers[paperid]
                            while paper1 == paper:
                                paperid = random.randrange(nump)
                                paper1 = papers[paperid]
                            paper = paper1
                            outline += " " + paper

                    if paper in self.paper_org:
                        words = self.paper_org[paper]
                        numw = len(words)
                        wordid = random.randrange(numw)
                        word = words[wordid]

                        papers = self.org_paper[word]
                        nump = len(papers)
                        if nump > 1:
                            paperid = random.randrange(nump)
                            paper1 = papers[paperid]
                            while paper1 == paper:
                                paperid = random.randrange(nump)
                                paper1 = papers[paperid]
                            paper = paper1
                            outline += " " + paper

                outfile.write(outline + "\n")
        outfile.close()
        print("walks done")


def save_relation(name_pubs_raw_filepath, name):  # 保存论文的各种语义feature
    '''

    Args:
        name_pubs_raw_filepath:
        name:

    Returns:

    '''
    r = '[!“”"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~—～’]+'
    stopword = ['at', 'based', 'in', 'of', 'for', 'on', 'and', 'to', 'an', 'using', 'with', 'the', 'by', 'we', 'be',
                'is', 'are', 'can']
    stopword1 = ['university', 'univ', 'china', 'department', 'dept', 'laboratory', 'lab', 'school', 'al', 'et',
                 'institute', 'inst', 'college', 'chinese', 'beijing', 'journal', 'science', 'international']

    f1 = open(f'{gene_root}/paper_author.txt', 'w', encoding='utf-8')
    f2 = open(f'{gene_root}/paper_conf.txt', 'w', encoding='utf-8')
    f3 = open(f'{gene_root}/paper_word.txt', 'w', encoding='utf-8')
    f4 = open(f'{gene_root}/paper_org.txt', 'w', encoding='utf-8')

    taken = name.split("_")
    if len(taken) > 2:
        name = taken[0] + taken[1] + taken[2]
        name_reverse = taken[2] + taken[0] + taken[1]
    else:
        name = taken[0] + taken[1]
        name_reverse = taken[1] + taken[0]

    authorname_dict = {}

    # 读取 name_pubs_raw
    name_pubs_raw = load_json(name_pubs_raw_filepath)
    for i, pid in enumerate(name_pubs_raw):
        pub = name_pubs_raw[pid]
        # save authors
        org = ""
        for author in pub["authors"]:
            authorname = re.sub(r, '', author["name"]).lower()
            taken = authorname.split(" ")
            if len(taken) == 2:  # 检测目前作者名是否在作者词典中
                authorname = taken[0] + taken[1]
                authorname_reverse = taken[1] + taken[0]

                if authorname not in authorname_dict:
                    if authorname_reverse not in authorname_dict:
                        authorname_dict[authorname] = 1
                    else:
                        authorname = authorname_reverse
            else:
                authorname = authorname.replace(" ", "")

            if authorname != name and authorname != name_reverse:
                f1.write(pid + '\t' + authorname + '\n')

            else:
                if "org" in author:
                    org = author["org"]

        # save org 待消歧作者的机构名
        pstr = org.strip()
        pstr = pstr.lower()  # 小写
        pstr = re.sub(r, ' ', pstr)  # 去除符号
        pstr = re.sub(r'\s{2,}', ' ', pstr).strip()  # 去除多余空格
        pstr = pstr.split(' ')
        pstr = [word for word in pstr if len(word) > 1]
        pstr = [word for word in pstr if word not in stopword1]
        pstr = [word for word in pstr if word not in stopword]
        pstr = set(pstr)
        for word in pstr:
            f4.write(pid + '\t' + word + '\n')

        # save venue
        pstr = pub["venue"].strip()
        pstr = pstr.lower()
        pstr = re.sub(r, ' ', pstr)
        pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
        pstr = pstr.split(' ')
        pstr = [word for word in pstr if len(word) > 1]
        pstr = [word for word in pstr if word not in stopword1]
        pstr = [word for word in pstr if word not in stopword]
        for word in pstr:
            f2.write(pid + '\t' + word + '\n')
        if len(pstr) == 0:
            f2.write(pid + '\t' + 'null' + '\n')

        # save text
        pstr = ""
        keyword = ""
        if "keywords" in pub:
            for word in pub["keywords"]:
                keyword = keyword + word + " "
        pstr = pstr + pub["title"]
        pstr = pstr.strip()
        pstr = pstr.lower()
        pstr = re.sub(r, ' ', pstr)
        pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
        pstr = pstr.split(' ')
        pstr = [word for word in pstr if len(word) > 1]
        pstr = [word for word in pstr if word not in stopword]
        for word in pstr:
            f3.write(pid + '\t' + word + '\n')

        # save all words' embedding
        pstr = keyword + " " + pub["title"] + " " + pub["venue"] + " " + org
        if "year" in pub:
            pstr = pstr + " " + str(pub["year"])
        pstr = pstr.strip()
        pstr = pstr.lower()
        pstr = re.sub(r, ' ', pstr)
        pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
        pstr = pstr.split(' ')
        pstr = [word for word in pstr if len(word) > 2]
        pstr = [word for word in pstr if word not in stopword]
        pstr = [word for word in pstr if word not in stopword1]

    f1.close()
    f2.close()
    f3.close()
    f4.close()


def get_pid_graph_embedding():
    ''' 通过构建文档图，随机游走采样元路径，获取每篇论文pid对应的图表征 '''
    os.makedirs(genename_root, exist_ok=True)
    os.makedirs(gene_root, exist_ok=True)
    # 处理测试集的数据
    # 加入测试集中待分配的论文
    unass_candi_v1 = load_json(processed_data_root, FilePathConfig.unass_candi_v1_path)
    unass_candi_v2 = load_json(processed_data_root, FilePathConfig.unass_candi_v2_path)

    pubs_info_v1 = load_json(raw_data_root, "cna-valid/cna_valid_unass_pub.json")
    pubs_info_v2 = load_json(raw_data_root, "cna-test/cna_test_unass_pub.json")
    unass_name2pinfo = {}
    unass_names = set()  # 保存所有涉及到的作者名
    for pub, name in unass_candi_v1:
        unass_names.add(name)
        pid = pub.split('-')[0]
        if name not in unass_name2pinfo:
            unass_name2pinfo[name] = {}
        unass_name2pinfo[name][pid] = pubs_info_v1[pid]

    for pub, name in unass_candi_v2:
        unass_names.add(name)
        pid = pub.split('-')[0]
        if name not in unass_name2pinfo:
            unass_name2pinfo[name] = {}
        unass_name2pinfo[name][pid] = pubs_info_v2[pid]
    # 处理训练集的数据
    pubs_raw_info = load_json(raw_data_root, FilePathConfig.train_pubs)
    name_pubs = load_json(raw_data_root, FilePathConfig.train_name2aid2pid)  # {'name': {'aid': [pid, ] } }
    clear_RW = open(gene_root + "/RW.txt", 'w')
    clear_RW.close()
    for n, name in enumerate(name_pubs):
        pubs = []
        for aid in name_pubs[name]:
            pubs.extend(name_pubs[name][aid])
        assert len(pubs) > 0
        name_pubs_raw = {}
        for i, pid in enumerate(pubs):
            name_pubs_raw[pid] = pubs_raw_info[pid]
        if name in unass_names:
            name_pubs_raw.update(unass_name2pinfo[name])
            unass_names.remove(name)  # 移除训练集中的名字
        name_pubs_raw_path = os.path.join(genename_root, name + '.json')
        save_json(name_pubs_raw, name_pubs_raw_path)
        save_relation(name_pubs_raw_path, name)
        mpg = MetaPathGenerator()
        mpg.read_data(gene_root)
        mpg.generate_WMRW(gene_root + "/RW.txt", 5, 20, 'a')  # 生成路径集
    # 处理剩余的测试集中涉及到的作者名
    whole_name2aid2pid = load_json(processed_data_root, FilePathConfig.whole_name2aid2pid)
    whole_pubs_info = load_json(processed_data_root, FilePathConfig.whole_pubsinfo)
    for name in unass_names:
        print(name)
        name_pubs_raw = {}
        name_pubs_raw.update(unass_name2pinfo[name])
        for aid, pubs in whole_name2aid2pid[name].items():
            for pub in pubs:
                pid = pub.split('-')[0]
                name_pubs_raw[pid] = whole_pubs_info[pid]
        name_pubs_raw_path = os.path.join(genename_root, name + '.json')
        save_json(name_pubs_raw, name_pubs_raw_path)
        save_relation(name_pubs_raw_path, name)
        mpg = MetaPathGenerator()
        mpg.read_data(gene_root)
        mpg.generate_WMRW(gene_root + "/RW.txt", 5, 20, 'a')  # 生成路径集

    # 训练论文 pid 词向量
    sentences = word2vec.Text8Corpus(gene_root + "/RW.txt")
    model = word2vec.Word2Vec(sentences, size=100, negative=25, min_count=1, window=10)
    os.makedirs(graph_feat_root + 'word2vec', exist_ok=True)
    model.save(os.path.join(graph_feat_root, FilePathConfig.pid_model_path))


def kernal_mus(n_kernels):
    """
    get the mu for each guassian kernel. Mu is the middle of each bin
    :param n_kernels: number of kernels (including exact match). first one is exact match
    :return: l_mu, a list of mu.
    """
    l_mu = [1]
    if n_kernels == 1:
        return l_mu

    bin_size = 2.0 / (n_kernels - 1)  # score range from [-1, 1]
    l_mu.append(1 - bin_size / 2)  # mu: middle of the bin
    for i in range(1, n_kernels - 1):
        l_mu.append(l_mu[i] - bin_size)
    #     print(l_mu)
    return l_mu


def kernel_sigmas(n_kernels):
    """
    get sigmas for each guassian kernel.
    :param n_kernels: number of kernels (including exactmath.)
    :param lamb:
    :param use_exact:
    :return: l_sigma, a list of simga
    """
    bin_size = 2.0 / (n_kernels - 1)
    l_sigma = [0.001]  # for exact match. small variance -> exact match
    if n_kernels == 1:
        return l_sigma

    l_sigma += [0.1] * (n_kernels - 1)
    #     print(l_sigma)
    return l_sigma


class GetMetrics:
    def __init__(self, sim_metric_type='cos', dim=41):
        ''' 根据待分配论文表征和候选者所有论文表征获取每个待分配论文的表征

        Args:
            dim:
        '''
        self.mu = np.array(kernal_mus(dim))
        self.sigma = np.array(kernel_sigmas(dim))
        self.sim_metric_type = sim_metric_type
        assert sim_metric_type in ['cos', 'euc', 'inner'], "metric type only can be ('cos'|'euc'|'inner')"

    def get_metric(self, paper_embedding, per_embedding):
        ''' 根据待分配论文表征和候选者所有论文表征获取每个待分配论文的表征

        Args:
            paper_embedding: (paper_num, paper_embed_dim)
            per_embedding: (author_paper_num, paper_embed_dim)

        Returns:

        '''
        if self.sim_metric_type == 'cos':
            sim_vec = cosine_similarity(paper_embedding, per_embedding)
        elif self.sim_metric_type == 'euc':
            sim_vec = cdist(paper_embedding, per_embedding, metric='euclidean')
        else:
            sim_vec = paper_embedding @ per_embedding.T

        sim_vec = sim_vec[:, :, np.newaxis]
        pooling_value = np.exp((-((sim_vec - self.mu) ** 2) / (self.sigma ** 2) / 2))
        pooling_sum = np.sum(pooling_value, axis=1)
        log_pooling_sum = np.log(np.clip(pooling_sum, a_min=1e-10, a_max=100)) * 0.01
        return log_pooling_sum


def deal_with_train(author_profile, author_unass, unass_pid_aidx2aid2embed, paper_mp_model, gen_metric_type_list,
                    pid_word2vec_model):
    for candiName in author_unass:
        unass_embeds = []
        unass_pids_w_aidx = []
        for aid in author_unass[candiName]:
            unass_pids_w_aidx.extend(author_unass[candiName][aid])
        unass_pids = [pid.split('-')[0] for pid in unass_pids_w_aidx]

        for pid in unass_pids:
            if pid in pid_word2vec_model:
                emb = pid_word2vec_model[pid]
            else:
                emb = np.zeros(100)
            unass_embeds.append(emb)
        assert len(unass_pids_w_aidx) == len(unass_embeds)
        unass_embeds = np.array(unass_embeds)
        ''' 处理person '''
        candi_aids_dict = author_profile[candiName]  # {aid:[pid, ]}
        aids = list(candi_aids_dict.keys())
        aid2embed = []
        for aid in aids:
            candi_papers_embed = []  # 该候选人的所有论文的表征
            for candi_pid in candi_aids_dict[aid]:
                c_pid = candi_pid.split('-')[0]
                if c_pid in pid_word2vec_model:
                    emb = pid_word2vec_model[c_pid]
                else:
                    emb = np.zeros(100)
                #                     print(c_pid, ' not found')
                candi_papers_embed.append(emb)
            candi_papers_embed = np.array(candi_papers_embed)
            for metric_type in gen_metric_type_list:
                sim = paper_mp_model[metric_type].get_metric(unass_embeds, candi_papers_embed)
                assert len(sim) == len(unass_pids_w_aidx)
                for temp_i, pid_aidx in enumerate(unass_pids_w_aidx):
                    unass_pid_aidx2aid2embed[metric_type][pid_aidx][aid] = sim[temp_i]


def get_paper_cos_euc_feature(unass_candi_path, online_name2aid2pid_path, unass_pid_aidx2aid2emb, paper_MP_model,
                              pid_word2vec_model, gen_metric_type_list=('cos', 'euc', 'inner')):
    unassCandi = load_json(unass_candi_path)
    online_nameAidPid = load_json(online_name2aid2pid_path)

    unass_name2pid = {}
    for unass in unassCandi:
        unassPid1, candiName = unass
        #     unassPid = unassPid1.split('-')[0]
        assert candiName in online_nameAidPid, f"{candiName} not found"
        if candiName not in unass_name2pid:
            unass_name2pid[candiName] = []
        unass_name2pid[candiName].append(unassPid1)

    for name in unass_name2pid:
        candiName = name
        unass_pids_w_aidx = unass_name2pid[name]
        unass_pids = [pid.split('-')[0] for pid in unass_pids_w_aidx]
        #     print(unass_pids)
        unass_embeds = []  # 待分配论文的表征
        pids = []
        for pid in unass_pids:
            if pid in pid_word2vec_model:
                emb = pid_word2vec_model[pid]
            else:
                emb = np.zeros(100)
            unass_embeds.append(emb)
        assert len(unass_pids_w_aidx) == len(unass_embeds)
        unass_embeds = np.array(unass_embeds)

        candi_aids_dict = online_nameAidPid[name]  # {aid:[pid, ]}
        aids = list(candi_aids_dict.keys())
        aid2embed = []
        for aid in aids:
            candi_papers_embed = []  # 该候选人的所有论文的表征
            for candi_pid in candi_aids_dict[aid]:
                c_pid = candi_pid.split('-')[0]
                if c_pid in pid_word2vec_model:
                    emb = pid_word2vec_model[c_pid]
                else:
                    emb = np.zeros(100)
                candi_papers_embed.append(emb)
            candi_papers_embed = np.array(candi_papers_embed)
            for metric_type in gen_metric_type_list:
                sim = paper_MP_model[metric_type].get_metric(unass_embeds, candi_papers_embed)
                assert len(sim) == len(unass_pids_w_aidx)
                for temp_i, pid_aidx in enumerate(unass_pids_w_aidx):
                    unass_pid_aidx2aid2emb[metric_type][pid_aidx][aid] = sim[temp_i]


def get_graph_sim_feat():
    # 加载之前训练出的 word2vec 模型
    pid_word2vec_model = word2vec.Word2Vec.load(os.path.join(graph_feat_root, FilePathConfig.pid_model_path))

    gen_metric_type_list = ['cos', 'euc', 'inner']
    paper_MP_model = {}
    unass_pid_aidx2aid2emb = {}
    for metric_type in gen_metric_type_list:
        unass_pid_aidx2aid2emb[metric_type] = defaultdict(dict)  # {'pid-aidx': {'aid': emb}}
        paper_MP_model[metric_type] = GetMetrics(metric_type)

    config = {
        'unass_candi_path'        : processed_data_root + FilePathConfig.unass_candi_v1_path,
        'online_name2aid2pid_path': processed_data_root + FilePathConfig.whole_name2aid2pid,
        'paper_MP_model'          : paper_MP_model,
        'unass_pid_aidx2aid2emb'  : unass_pid_aidx2aid2emb,
        'gen_metric_type_list'    : gen_metric_type_list,
        'pid_word2vec_model'      : pid_word2vec_model
    }
    get_paper_cos_euc_feature(**config)

    config = {
        'unass_candi_path'        : processed_data_root + FilePathConfig.unass_candi_v2_path,
        'online_name2aid2pid_path': processed_data_root + FilePathConfig.whole_name2aid2pid,
        'paper_MP_model'          : paper_MP_model,
        'unass_pid_aidx2aid2emb'  : unass_pid_aidx2aid2emb,
        'gen_metric_type_list'    : gen_metric_type_list,
        'pid_word2vec_model'      : pid_word2vec_model
    }
    get_paper_cos_euc_feature(**config)
    # 处理训练集中的特征
    train_author_profile = load_json(processed_data_root, 'train/offline_profile.json')
    train_author_unass = load_json(processed_data_root, 'train/offline_unass.json')
    deal_with_train(train_author_profile, train_author_unass, unass_pid_aidx2aid2emb, paper_MP_model,
                    gen_metric_type_list, pid_word2vec_model)

    print('saving')
    for metric_type in gen_metric_type_list:
        save_pickle(dict(unass_pid_aidx2aid2emb[metric_type]), graph_feat_root,
                    FilePathConfig.cos_path[metric_type])
    print('saved')


if __name__ == '__main__':
    get_pid_graph_embedding()
    get_graph_sim_feat()
