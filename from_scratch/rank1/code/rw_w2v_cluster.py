from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from datetime import datetime
try:
    from .utils import *
except Exception:
    from utils import *

try:
    from .text_process_test import TextProcesser
except Exception:
    from text_process_test import TextProcesser
from tqdm import tqdm


class RwW2vClusterMaster:
    def __init__(self, rw_conf, rule_conf, mode='train', w2v_model='tvt', text_weight=1,
                 db_eps=0.2, db_min=4, res_name='res', comment='', idf_file='',
                 text_processed=False, add_abstract=False, if_local_idf=False,
                 local_idf_ab=False, local_idf_pc=False, base=None):
        base = base
        if mode == 'valid':
            # pid: paper_infos
            self.pubs = load_json(join(base, "sna_valid", "sna_valid_pub.json"))
            # name: [pids]
            self.raw_pubs = load_json(join(base, "sna_valid", "sna_valid_raw.json"))
        elif mode == 'test':
            self.pubs = load_json(join(base, 'sna_test', 'sna_test_pub.json'))
            self.raw_pubs = load_json(join(base, 'sna_test', 'sna_test_raw.json'))
        elif mode == 'train':
            self.pubs = load_json(join(base, "train", "train_pub.json"))
            # name: {aid: [pids]}
            self.raw_pubs = load_json(join(base, "train", "train_author.json"))
        else:
            raise ValueError('choose right mode')
        self.mode = mode
        cur_time = datetime.now().strftime("%m%d%H%M")
        self.w2v_model_file = f'./word2vec/{w2v_model}.model'
        check_mkdir('./res/log_summary')
        check_mkdir(f'./res/output/{mode}')
        check_mkdir(f'./res/log/{mode}')
        self.out_file = f'./res/output/{mode}/{res_name}_{cur_time}.json'
        self.log_file = f'./res/log/{mode}/{res_name}_{cur_time}.txt'
        self.All_log_file = f'./res/log_summary/Alog_{mode}.txt'
        self.idf_file = f'./TFIDF/{idf_file}_idf.pkl'
        self.local_idf_dir = f'./TFIDF/{mode}/ab_{int(local_idf_ab)}_pc_{int(local_idf_pc)}'
        if len(idf_file) > 0:
            self.if_idf = True
        else:
            self.if_idf = False
        self.if_local_idf = if_local_idf
        self.text_processed = text_processed
        self.tp = None
        self.add_abstract = add_abstract
        if text_processed:
            self.tp = TextProcesser()
        self.comment = comment

        # 超参数
        self.rw_conf = rw_conf
        self.rule_conf = rule_conf
        self.text_weight = text_weight
        self.db_eps = db_eps
        self.db_min = db_min

        self.puncs = '[!“”"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~—～’]+'
        self.stopwords = ['at', 'based', 'in', 'of', 'for', 'on', 'and', 'to', 'an', 'using', 'with',
                          'the', 'by', 'we', 'be', 'is', 'are', 'can']
        # self.stopwords_extend = ['university', 'univ', 'china', 'department', 'dept', 'laboratory', 'lab',
        #                          'school', 'al', 'et', 'institute', 'inst', 'college', 'chinese', 'beijing',
        #                          'journal', 'science', 'international']
        self.stopwords_extend = ['university', 'univ', 'china', 'department', 'dept', 'laboratory', 'lab',
                                 'school', 'al', 'et', 'institute', 'inst', 'college', 'chinese', 'beijing',
                                 'journal', 'science', 'international', 'key', 'sciences', 'research',
                                 'academy', 'state', 'center']

        self.stopwords_check = ['a', 'was', 'were', 'that', '2', 'key', '1', 'technology', '0', 'sciences', 'as',
                                 'from', 'r', '3', 'academy', 'this', 'nanjing', 'shanghai', 'state', 's', 'research',
                                'p', 'results', 'peoples', '4', 'which', '5', 'high', 'materials', 'study', 'control',
                                'method', 'group', 'c', 'between', 'or', 'it', 'than', 'analysis', 'system',  'sci',
                                'two', '6', 'has', 'h', 'after', 'different', 'n', 'national', 'japan', 'have', 'cell',
                                'time', 'zhejiang', 'used', 'data', 'these']

    def run(self):
        # if not os.listdir(join('gen_names', self.mode)):
        #     print('Generating name, relation cache files...')
        #     self.gene_files()
        #     print('Finish file generating')
        self.gene_files()
        print('Finish file generating')
        result = {}
        if self.mode == 'train':
            precisions, recalls, f1s = [], [], []
        for n, name in enumerate(self.raw_pubs):
            if self.mode == 'valid' or self.mode == 'test':
                pubs = self.raw_pubs[name]
            else:
                pubs = []
                ilabel = 0
                labels = []
                for aid in self.raw_pubs[name]:
                    pubs.extend(self.raw_pubs[name][aid])
                    labels.extend([ilabel] * len(self.raw_pubs[name][aid]))
                    ilabel += 1

            ##元路径游走类
            ###############################################################r
            mpg = MetaPathGenerator()
            mpg.read_data(join("gen_relations", self.mode, name))

            ##论文关系表征向量
            ###############################################################
            all_embs = []
            # paper id not occur in RW(关系缺失)
            cp = set()
            for k in range(self.rw_conf['repeat_num']):
                check_mkdir(f'gen_rw/{self.mode}')
                # Rw.txt 仅作为cache
                mpg.generate_WMRW(f"gen_rw/{self.mode}/RW.txt", self.rw_conf['num_walk'], self.rw_conf['walk_len'])
                sentences = word2vec.Text8Corpus(f'gen_rw/{self.mode}/RW.txt')
                model = word2vec.Word2Vec(sentences, size=self.rw_conf['rw_dim'], negative=self.rw_conf['neg'],
                                          min_count=1, window=self.rw_conf['window'])
                embs = []
                for i, pid in enumerate(pubs):
                    if pid in model:
                        embs.append(model[pid])
                    else:
                        cp.add(i)
                        embs.append(np.zeros(100))
                all_embs.append(embs)
            all_embs = np.array(all_embs)

            # add from train
            tcp = load_data(join('other_gen', self.mode, name, 'tcp.pkl'))
            # print('semantic outlier:', tcp)

            ##论文语义表征向量
            ###############################################################
            if self.if_idf or self.if_local_idf:
                ptext_emb = load_data(join('other_gen', self.mode, name, 'ptext_emb_idf.pkl'))
            else:
                ptext_emb = load_data(join('other_gen', self.mode, name, 'ptext_emb.pkl'))
            tembs = []
            # pdb.set_trace()
            for i, pid in enumerate(pubs):
                tembs.append(ptext_emb[pid])

            sk_dis = np.zeros((len(pubs), len(pubs)))
            for k in range(self.rw_conf['repeat_num']):
                sk_dis = sk_dis + pairwise_distances(all_embs[k], metric="cosine")
            sk_dis = sk_dis / self.rw_conf['repeat_num']
            tembs_dis = pairwise_distances(tembs, metric="cosine")

            ##用拼接矩阵计算相似度，权值设为1：1
            ###############################################################
            dis = (np.array(sk_dis) + self.text_weight * np.array(tembs_dis)) / (1 + self.text_weight)
            ##聚类操作，返回cluster labels
            ###############################################################
            pred = DBSCAN(eps=self.db_eps, min_samples=self.db_min, metric='precomputed').fit_predict(dis)
            pred = np.array(pred)

            outlier = set()
            for i in range(len(pred)):
                if pred[i] == -1:
                    outlier.add(i)
            for i in cp:
                outlier.add(i)
            for i in tcp:
                outlier.add(i)

            ## 基于阈值的相似性匹配
            paper_pair = self.generate_pair(pubs, name, outlier)
            paper_pair1 = paper_pair.copy()
            # top num
            K = len(set(pred))
            for i in range(len(pred)):
                if i not in outlier:
                    continue
                j = np.argmax(paper_pair[i])
                while j in outlier:
                    paper_pair[i][j] = -1
                    j = np.argmax(paper_pair[i])
                if paper_pair[i][j] >= 1.5:
                    pred[i] = pred[j]
                # 单独成簇
                else:
                    pred[i] = K
                    K = K + 1

            # 尝试在离群点中找同类(之前修改了paper_pair, paper_pair1保留)
            for ii, i in enumerate(outlier):
                for jj, j in enumerate(outlier):
                    if jj <= ii:
                        continue
                    else:
                        if paper_pair1[i][j] >= 1.5:
                            pred[j] = pred[i]

            result[name] = []
            for i in set(pred):
                oneauthor = []
                for idx, j in enumerate(pred):
                    if i == j:
                        oneauthor.append(pubs[idx])
                result[name].append(oneauthor)

            ##训练集上的分数评估
            ###############################################################
            if self.mode == 'train':
                labels = np.array(labels)
                pred = np.array(pred)
                pred_label_num = len(set(pred))
                true_label_num = len(set(labels))
                pairwise_precision, pairwise_recall, pairwise_f1 = pairwise_evaluate(labels, pred)
                with open(self.log_file, 'a') as f:
                    f.write(f'name: {name}, prec: {pairwise_precision: .4}, recall: {pairwise_recall: .4}, '
                            f'f1: {pairwise_f1: .4}, pred label num : {pred_label_num}/{true_label_num}\n')
                precisions.append(pairwise_precision)
                recalls.append(pairwise_recall)
                f1s.append(pairwise_f1)

        ##存储预测结果
        dump_json(result, self.out_file, indent=4)

        ##存储评估log
        if self.mode == 'train':
            with open(self.log_file, 'a') as f:
                f.write(f'AVG, prec: {np.mean(precisions): .4}, recall: {np.mean(recalls): .4}, '
                        f'f1: {np.mean(f1s): .4}\n')
            with open(self.All_log_file, 'a') as f:
                f.write(f'Detail log file: {self.log_file}, res: {self.out_file}\n')
                f.write('Parameters:\n')
                f.write(str(self.rw_conf) + '\n')
                f.write(str(self.rule_conf) + '\n')
                f.write(f'text_weight: {self.text_weight}, '
                        f'db eps:{self.db_eps}, db_min: {self.db_min}\n')
                if len(self.comment) > 0:
                    f.write(f'Comment: {self.comment}\n')
                f.write(f'AVG, prec: {np.mean(precisions): .4}, recall: {np.mean(recalls): .4}, '
                        f'f1: {np.mean(f1s): .4}\n')
                f.write('--------------------------------------\n')

    def gene_files(self):
        '''
        保存 name.json 存储同名下的papers
        更新：保存至 ./gen_names/mode/

        :return:
        '''
        if not os.path.exists(join('gen_names', self.mode)):
            check_mkdir(join('gen_names', self.mode))
            for name in tqdm(self.raw_pubs):
                name_pubs_raw = {}
                if self.mode == 'valid' or self.mode == 'test':
                    for i, pid in enumerate(self.raw_pubs[name]):
                        name_pubs_raw[pid] = self.pubs[pid]
                else:
                    pubs = []
                    for aid in self.raw_pubs[name]:
                        pubs.extend(self.raw_pubs[name][aid])
                    for pid in pubs:
                        name_pubs_raw[pid] = self.pubs[pid]
                dump_json(name_pubs_raw, join('gen_names', self.mode, name + '.json'), indent=4)

        for name in self.raw_pubs:
            self.save_relation_files(name)
            # print(f'Finish {name} file gen.')

    def save_relation_files(self, name):
        name_pubs = load_json(join('gen_names', self.mode, name + '.json'))
        w2v_model = word2vec.Word2Vec.load(self.w2v_model_file)

        # 每行格式均为：pid \t content
        # author_name(不包括需要消歧的名字)
        check_mkdir(f'gen_relations/{self.mode}/{name}')
        f1 = open(f'gen_relations/{self.mode}/{name}/paper_author.txt', 'w', encoding='utf-8')
        # venue word
        f2 = open(f'gen_relations/{self.mode}/{name}/paper_conf.txt', 'w', encoding='utf-8')
        # 仅包含title word
        f3 = open(f'gen_relations/{self.mode}/{name}/paper_word.txt', 'w', encoding='utf-8')
        # org word(仅待消歧的作者名的机构)
        f4 = open(f'gen_relations/{self.mode}/{name}/paper_org.txt', 'w', encoding='utf-8')

        text_feature_path = join('other_gen', self.mode, name)
        check_mkdir(text_feature_path)

        if self.if_idf:
            idf_weight = load_data(self.idf_file)
        elif self.if_local_idf:
            idf_weight = load_data(f'{self.local_idf_dir}/{name}_idf.pkl')
        ori_name = name
        taken = name.split("_")
        name = taken[0] + taken[1]
        name_reverse = taken[1] + taken[0]
        if len(taken) > 2:
            name = taken[0] + taken[1] + taken[2]
            name_reverse = taken[2] + taken[0] + taken[1]

        authorname_dict = {}
        # ptext_emb: key is paper id, and the value is the paper's text embedding
        ptext_emb = {}
        # the paper index that lack text information
        tcp = set()
        for i, pid in enumerate(name_pubs):

            pub = name_pubs[pid]

            # save authors
            org = ""
            find_author = False
            for author in pub["authors"]:
                # authorname = re.sub(self.puncs, '', author["name"]).lower()
                authorname = ''.join(filter(str.isalpha, author['name'])).lower()
                # 太迷了，authorname就不可能有空格，一定会进else分支的

                taken = authorname.split(" ")
                if len(taken) == 2:  ##检测目前作者名是否在作者词典中
                    authorname = taken[0] + taken[1]
                    authorname_reverse = taken[1] + taken[0]

                    if authorname not in authorname_dict:
                        if authorname_reverse not in authorname_dict:
                            authorname_dict[authorname] = 1
                        else:
                            authorname = authorname_reverse
                else:  # len(taken)==3 ??
                    authorname = authorname.replace(" ", "")

                # 非待消歧的作者名
                if authorname != name and authorname != name_reverse:
                    f1.write(pid + '\t' + authorname + '\n')
                # 是待消歧的作者名
                else:
                    if "org" in author:
                        org = author["org"]
                        find_author = True
            ##如果没匹配到，再试试name_match
            ###################################
            if not find_author:
                for author in pub['authors']:
                    if match_name(author['name'], ori_name):
                        org = author['org']
                        break
            # save org 待消歧作者的机构名
            pstr = org.strip()
            pstr = pstr.lower()  # 小写
            pstr = re.sub(self.puncs, ' ', pstr)  # 去除符号
            pstr = re.sub(r'\s{2,}', ' ', pstr).strip()  # 去除多余空格
            pstr = pstr.split(' ')
            pstr = [word for word in pstr if len(word) > 1]
            pstr = [word for word in pstr if word not in self.stopwords]
            pstr = [word for word in pstr if word not in self.stopwords_extend]

            # pstr = [word for word in pstr if word not in self.stopwords_check]
            pstr = set(pstr)
            for word in pstr:
                f4.write(pid + '\t' + word + '\n')

            # save venue
            if pub["venue"]:
                pstr = pub["venue"].strip()
                pstr = pstr.lower()
                pstr = re.sub(self.puncs, ' ', pstr)
                pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
                pstr = pstr.split(' ')
                pstr = [word for word in pstr if len(word) > 1]
                pstr = [word for word in pstr if word not in self.stopwords]
                pstr = [word for word in pstr if word not in self.stopwords_extend]

                pstr = [word for word in pstr if word not in self.stopwords_check]
                for word in pstr:
                    f2.write(pid + '\t' + word + '\n')
                # 处理后无 venue的存null
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
            pstr = re.sub(self.puncs, ' ', pstr)
            pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
            pstr = pstr.split(' ')
            pstr = [word for word in pstr if len(word) > 1]
            pstr = [word for word in pstr if word not in self.stopwords]
            pstr = [word for word in pstr if word not in self.stopwords_check]
            for word in pstr:
                f3.write(pid + '\t' + word + '\n')

            # save all words' embedding
            if pub["venue"]:
                pstr = pub["title"] + " " + keyword + " " + pub["venue"] + " " + org
            else:
                pstr = pub["title"] + " " + keyword + " " + org

            if self.add_abstract and pub['abstract']:
                pstr = pstr + " " + pub["abstract"]

            if "year" in pub:
                pstr = pstr + " " + str(pub["year"])
            pstr = pstr.strip()
            if not self.text_processed:
                pstr = pstr.lower()
                pstr = re.sub(self.puncs, ' ', pstr)
                pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
                pstr = pstr.split(' ')
                pstr = [word for word in pstr if len(word) > 2]
                pstr = [word for word in pstr if word not in self.stopwords]
                pstr = [word for word in pstr if word not in self.stopwords_extend]

                pstr = [word for word in pstr if word not in self.stopwords_check]
            else:
                pstr = self.tp.process(pstr)
                pstr = pstr.split()

            if (not self.if_idf) and (not self.if_local_idf):
                words_vec = []
                for word in pstr:
                    if word in w2v_model:
                        words_vec.append(w2v_model[word])
                if len(words_vec) < 1:
                    words_vec.append(np.zeros(100))
                    tcp.add(i)
                    # print ('outlier:',pid,pstr)

                ptext_emb[pid] = np.mean(words_vec, 0)
            else:
                idf_sum = 0
                words_vec = np.zeros(100)
                for word in pstr:
                    if word in w2v_model and word in idf_weight:
                        words_vec += w2v_model[word] * idf_weight[word]
                        idf_sum += idf_weight[word]
                if np.array_equal(words_vec, np.zeros(100)):
                    idf_sum = 1
                    tcp.add(i)
                    # print ('outlier:',pid,pstr)

                ptext_emb[pid] = words_vec / idf_sum

        # ptext_emb: key is paper id, and the value is the paper's text embedding
        if (not self.if_idf) and (not self.if_local_idf):
            dump_data(ptext_emb, join(text_feature_path, 'ptext_emb.pkl'))
        else:
            dump_data(ptext_emb, join(text_feature_path, 'ptext_emb_idf.pkl'))
        # the paper index that lack text information
        dump_data(tcp, join(text_feature_path, 'tcp.pkl'))

        f1.close()
        f2.close()
        f3.close()
        f4.close()

    def generate_pair(self, pubs, name, outlier):  ##求匹配相似度
        dirpath = join('gen_relations', self.mode, name)

        paper_org = {}
        paper_conf = {}
        paper_author = {}
        paper_word = {}

        temp = set()
        with open(dirpath + "/paper_org.txt", encoding='utf-8') as pafile:
            for line in pafile:
                temp.add(line)
        for line in temp:
            toks = line.strip().split("\t")
            if len(toks) == 2:
                p, a = toks[0], toks[1]
                if p not in paper_org:
                    paper_org[p] = []
                paper_org[p].append(a)
        temp.clear()

        with open(dirpath + "/paper_conf.txt", encoding='utf-8') as pafile:
            for line in pafile:
                temp.add(line)
        for line in temp:
            toks = line.strip().split("\t")
            if len(toks) == 2:
                p, a = toks[0], toks[1]
                if p not in paper_conf:
                    paper_conf[p] = []
                paper_conf[p] = a
        temp.clear()

        with open(dirpath + "/paper_author.txt", encoding='utf-8') as pafile:
            for line in pafile:
                temp.add(line)
        for line in temp:
            toks = line.strip().split("\t")
            if len(toks) == 2:
                p, a = toks[0], toks[1]
                if p not in paper_author:
                    paper_author[p] = []
                paper_author[p].append(a)
        temp.clear()

        with open(dirpath + "/paper_word.txt", encoding='utf-8') as pafile:
            for line in pafile:
                temp.add(line)
        for line in temp:
            toks = line.strip().split("\t")
            if len(toks) == 2:
                p, a = toks[0], toks[1]
                if p not in paper_word:
                    paper_word[p] = []
                paper_word[p].append(a)
        temp.clear()

        paper_paper = np.zeros((len(pubs), len(pubs)))
        for i, pid in enumerate(pubs):
            if i not in outlier:
                continue
            for j, pjd in enumerate(pubs):
                if j == i:
                    continue
                ca = 0
                cv = 0
                co = 0
                ct = 0

                if pid in paper_author and pjd in paper_author:
                    ca = len(set(paper_author[pid]) & set(paper_author[pjd])) * self.rule_conf['w_author']
                if pid in paper_conf and pjd in paper_conf and 'null' not in paper_conf[pid]:
                    cv = tanimoto(set(paper_conf[pid]), set(paper_conf[pjd])) * self.rule_conf['w_venue']
                if pid in paper_org and pjd in paper_org:
                    co = tanimoto(set(paper_org[pid]), set(paper_org[pjd])) * self.rule_conf['w_org']
                if pid in paper_word and pjd in paper_word:
                    ct = len(set(paper_word[pid]) & set(paper_word[pjd])) * self.rule_conf['w_word']

                paper_paper[i][j] = ca + cv + co + ct

        return paper_paper
