import os
import random
import copy
from collections import defaultdict
import multiprocessing

from utils import load_json, save_json, get_author_index, dname_l_dict
from character.name_match.tool.is_chinese import cleaning_name
from character.name_match.tool.interface import FindMain
from whole_config import FilePathConfig, configs, raw_data_root, processed_data_root


def printInfo(dicts):
    aNum = 0
    pNum = 0
    for name, aidPid in dicts.items():
        aNum += len(aidPid)
        for aid, pids in aidPid.items():
            pNum += len(pids)

    print("#Name %d, #Author %d, #Paper %d" % (len(dicts), aNum, pNum))


def split_train2dev(train_author_filepath, train_pub_filepath, unass_ratio=0.2):
    '''将 train 划分为训练集和测试集'''

    def _get_last_n_paper(name, paper_ids, paper_info, ratio=0.2):
        cnt_unfind_author_num = 0  # 未找到作者 index 的数量
        name = cleaning_name(name)
        years = set()
        now_years = defaultdict(list)
        for pid in paper_ids:
            year = paper_info[pid].get('year', '0')
            if year == '':
                year = 0
            else:
                year = int(year)
            if year < 1500 or year > 2022:
                year = 0
            years.add(year)
            authors = paper_info[pid].get('authors', [])
            author_names = [a['name'] for a in authors]
            author_res = FindMain(name, author_names)[0]
            if len(author_res) > 0:
                aids = author_res[0][1]
            else:
                aids = get_author_index(name, author_names, False)
                if aids < 0:
                    aids = len(authors)
                    cnt_unfind_author_num += 1
            assert aids >= 0
            # if aids == len(authors):
            #     cnt_unfind_author_num += 1
            # assert aids >= 0, f"{name} 's paper {pid}"
            now_years[year].append((pid, aids,))
        years = list(years)
        years.sort(reverse=False)
        papers_list = []
        assert len(years) > 0
        for y in years:
            papers_list.extend(now_years[y])
        # 取后 ratio 的作为未分配论文
        split_gap = int((1 - ratio) * len(papers_list))
        unass_list = papers_list[split_gap:]
        prof_list = papers_list[0:split_gap]
        assert len(unass_list) > 0
        assert len(prof_list) > 0
        assert len(unass_list) + len(prof_list) == len(papers_list)
        return prof_list, unass_list, cnt_unfind_author_num

    def _split_unass(names, authors_info, papers_info, unass_info, dump_info):
        sum_unfind_author_num = 0
        unass_candi_list = []
        for name in names:
            unass_info[name] = {}
            dump_info[name] = {}
            for aid in authors_info[name]:
                papers = authors_info[name][aid]
                prof_list, unass_list, cnt_unfind_num = _get_last_n_paper(name, papers, papers_info, unass_ratio)
                sum_unfind_author_num += cnt_unfind_num
                # for pid in pid_rm_list:
                #     authors_info[name][aid].remove(pid[0])
                unass_info[name][aid] = [f"{p[0]}-{p[1]}" for p in unass_list if
                                         'authors' in papers_info[p[0]] and 0 <= p[1] < len(
                                             papers_info[p[0]]['authors'])]
                dump_info[name][aid] = [f"{p[0]}-{p[1]}" for p in prof_list]
                for pid in unass_info[name][aid]:
                    unass_candi_list.append((pid, name))
        print('未找到作者序号的论文篇数 : ', sum_unfind_author_num)
        return unass_candi_list

    papers_info = load_json(raw_data_root, train_pub_filepath)
    authors_info = load_json(raw_data_root, train_author_filepath)
    names = []
    # author_ids = []
    for name in authors_info:
        names.append(name)
    random.shuffle(names)
    # 随机将训练集中的名字划分为 train 和 dev
    train_threshold = int(len(names) * 0.8)
    train_names = names[:train_threshold]
    test_names = names[train_threshold:]
    train_unass_info = {}
    train_dump_info = {}
    train_unass_candi = _split_unass(train_names, authors_info, papers_info, train_unass_info, train_dump_info)
    save_json(train_dump_info, processed_data_root, "train/train_author_profile.json")
    save_json(train_unass_info, processed_data_root, "train/train_author_unass.json")
    save_json(train_unass_candi, processed_data_root, 'train/unass_candi.train.json')

    test_unass_info = {}
    test_dump_info = {}
    test_unass_candi = _split_unass(test_names, authors_info, papers_info, test_unass_info, test_dump_info)
    save_json(test_dump_info, processed_data_root, "train/test_author_profile.json")
    save_json(test_unass_info, processed_data_root, "train/test_author_unass.json")
    save_json(test_unass_candi, processed_data_root, 'train/unass_candi.test.json')


def get_author_index_father(params):
    ''' 为多线程封装的函数 '''
    unass_pid, name, dnames = params
    author_res = FindMain(name, dnames)[0]
    if len(author_res) > 0:
        return unass_pid, author_res[0][1], 'find', name
    res = get_author_index(name, dnames, True)
    return unass_pid, res, 'doudi', name


def get_name2aid2pid(name2aid2pids_path):
    ''' 产生 已有的所有名字到aid2pids的字典 '''
    # 加载比赛给出的 whole_author_profiles.json 文件
    whole_pros = load_json(raw_data_root, FilePathConfig.database_name2aid2pid)
    whole_pubs_info = load_json(raw_data_root, FilePathConfig.database_pubs)
    # 加载训练集中的 作者名字信息
    train_pros = load_json(raw_data_root, FilePathConfig.train_name2aid2pid)
    train_pubs_info = load_json(raw_data_root, FilePathConfig.train_pubs)

    whole_pubs_info.update(train_pubs_info)
    save_json(whole_pubs_info, processed_data_root, FilePathConfig.whole_pubsinfo)

    this_year = 2022
    # Merge all authors under the same name.
    name_aid_pid = defaultdict(dict)
    for aid, ainfo in whole_pros.items():
        name = ainfo['name']
        pubs = ainfo['pubs']
        name_aid_pid[name][aid] = pubs
    # print(name_aid_pid)
    # Find the main author index for each paper
    key_names = list(name_aid_pid.keys())
    new_name2aid2pids = defaultdict(dict)

    for i in range(len(key_names)):
        name = key_names[i]
        aid2pid = name_aid_pid[name]
        for aid, pids in aid2pid.items():
            tmp_pubs = []
            for pid in pids:
                coauthors = [tmp['name'] for tmp in whole_pubs_info[pid]['authors']]
                coauthors = [n.replace('.', ' ').lower() for n in coauthors]
                if 'year' in whole_pubs_info[pid]:
                    year = whole_pubs_info[pid]['year']
                    year = int(year) if year != '' else this_year
                else:
                    year = this_year

                aidx = get_author_index_father((pid, name, coauthors))[1]
                if aidx < 0:
                    aidx = len(coauthors)
                new_pid = pid + '-' + str(aidx)
                tmp_pubs.append((new_pid, year))
            tmp_pubs.sort(key=lambda x: x[1], reverse=True)
            tmp_pubs = [p[0] for p in tmp_pubs]
            new_name2aid2pids[name][aid] = tmp_pubs
    printInfo(new_name2aid2pids)

    for name, aid2pid in train_pros.items():
        assert name not in key_names
        for aid, pids in aid2pid.items():
            tmp_pubs = []
            for pid in pids:
                coauthors = [tmp['name'].lower() for tmp in train_pubs_info[pid]['authors']]
                coauthors = [n.replace('.', ' ').lower() for n in coauthors]
                if 'year' in train_pubs_info[pid]:
                    year = train_pubs_info[pid]['year']
                    year = int(year) if year != '' else this_year
                else:
                    year = this_year

                aidx = get_author_index_father((pid, name, coauthors))[1]
                if aidx < 0:
                    aidx = len(coauthors)
                new_pid = pid + '-' + str(aidx)

                tmp_pubs.append((new_pid, year))
            tmp_pubs.sort(key=lambda x: x[1], reverse=True)
            tmp_pubs = [p[0] for p in tmp_pubs]
            new_name2aid2pids[name][aid] = tmp_pubs
    new_name2aid2pids = dict(new_name2aid2pids)
    printInfo(new_name2aid2pids)
    # 保存所有的名字到aid2pid的字典
    save_json(new_name2aid2pids, processed_data_root, name2aid2pids_path)


def pretreat_unass(unass_candi_path, unass_list_path, unass_paper_info_path):
    ''' 处理在线测试集

    Args:
        unass_candi_path: 处理后的 candidate [(pid_aidx, name)] 保存路径
        unass_list_path: 待分配论文路径 [pid_aidx]
        unass_paper_info_path: 待分配论文信息路径

    Returns:

    '''
    name_aid_pid = load_json(processed_data_root, FilePathConfig.whole_name2aid2pid)
    #
    unass_list = load_json(raw_data_root, unass_list_path)
    unass_paper_info = load_json(raw_data_root, unass_paper_info_path)
    whole_candi_names = list(name_aid_pid.keys())
    print("#Unass: %d #candiNames: %d" % (len(unass_list), len(whole_candi_names)))
    for dname in whole_candi_names:
        if dname not in dname_l_dict.keys():
            dname_l_dict[dname] = cleaning_name(dname).split()
    # ----------
    unass_candi = []
    not_match = 0
    # ----------

    num_thread = int(multiprocessing.cpu_count() / 1.3)
    pool = multiprocessing.Pool(num_thread)

    ins = []
    for unass_pid in unass_list:
        pid, aidx = unass_pid.split('-')
        candi_name = unass_paper_info[pid]['authors'][int(aidx)]['name']

        ins.append((unass_pid, candi_name, whole_candi_names))

    multi_res = pool.map(get_author_index_father, ins)
    pool.close()
    pool.join()
    for i in multi_res:
        pid, aidx, typ, name = i
        if aidx >= 0:
            unass_candi.append((pid, whole_candi_names[aidx]))
        else:
            not_match += 1
            print(i)
    print("Matched: %d Not Match: %d" % (len(unass_candi), not_match))

    # print(unass_candi_path)
    save_json(unass_candi, processed_data_root, unass_candi_path)
    save_json(whole_candi_names, processed_data_root, 'whole_candi_names.json')


def split_list2kfold(s_list, k, start_index=0):
    # 将输入列表划分为 k 份
    num = len(s_list)
    each_l = int(num / k)
    result = []
    remainer = num % k
    random.shuffle(s_list)
    last_index = 0
    for i in range(k):
        if (k + i - start_index) % k < remainer:
            result.append(s_list[last_index:last_index + each_l + 1])
            last_index += each_l + 1
        else:
            result.append(s_list[last_index:last_index + each_l])
            last_index += each_l
    return result, (start_index + remainer) % k


def kfold_main_func(offline_whole_profile, offline_whole_unass, k=5):
    kfold_path = f"{processed_data_root}/train/kfold_dataset/"
    os.makedirs(kfold_path, exist_ok=True)
    # 获取训练集的作者名字+候选者数量列表
    name_weight = []
    for name, aid2pids in offline_whole_profile.items():
        assert len(aid2pids.keys()) == len(offline_whole_unass[name].keys())
        name_weight.append((name, len(aid2pids.keys())))
    # name_weight.sort(key=lambda x: x[1])
    # 尽量保持各折的平衡
    both_name_weight = []
    unused_name_weight = []
    for name, weight in name_weight:
        if weight < 20:
            unused_name_weight.append((name, weight))
        else:
            both_name_weight.append((name, weight))
    # 将名字的集合分为 k 组
    start_index = 0
    split_res = [[] for i in range(k)]
    tmp, start_index = split_list2kfold(unused_name_weight, k, start_index)
    for i in range(k):
        split_res[i].extend(tmp[i])

    tmp, start_index = split_list2kfold(both_name_weight, k, start_index)
    for i in range(k):
        split_res[i].extend(tmp[i])
    # 产生训练需要的数据集合
    for i in range(k):
        this_root = os.path.join(kfold_path, f'kfold_v{i + 1}')
        os.makedirs(this_root, exist_ok=True)
        dev_names = split_res[i]
        train_names = []
        for j in range(k):
            if j != i:
                train_names.extend(split_res[j])

        train_ins = []
        for na_w in train_names:
            name = na_w[0]
            whole_candi_aids = list(offline_whole_unass[name].keys())
            if len(whole_candi_aids) < configs['train_neg_sample'] + 1:
                continue
            for pos_aid, pids in offline_whole_unass[name].items():
                for pid in pids:
                    neg_aids = copy.deepcopy(whole_candi_aids)
                    neg_aids.remove(pos_aid)
                    neg_aids = random.sample(neg_aids, configs['train_neg_sample'])
                    train_ins.append((name, pid, pos_aid, neg_aids))
        save_json(train_ins, this_root, 'train_ins.json')

        dev_ins = []
        for na_w in dev_names:
            name = na_w[0]
            whole_candi_aids = list(offline_whole_unass[name].keys())
            if len(whole_candi_aids) < configs['test_neg_sample'] + 1:
                continue
            for pos_aid, pids in offline_whole_unass[name].items():
                for pid in pids:
                    neg_aids = copy.deepcopy(whole_candi_aids)
                    neg_aids.remove(pos_aid)
                    neg_aids = random.sample(neg_aids, configs['test_neg_sample'])
                    dev_ins.append((name, pid, pos_aid, neg_aids))
        save_json(dev_ins, this_root, 'test_ins.json')
    print(name_weight)


def main():
    random.seed(66)
    # 划分数据集
    split_train2dev(train_author_filepath=FilePathConfig.train_name2aid2pid,
                    train_pub_filepath=FilePathConfig.train_pubs, unass_ratio=0.2)
    # 合并分割的测试集和验证集
    unass_candi_train = load_json(processed_data_root, 'train/unass_candi.train.json')
    unass_candi_test = load_json(processed_data_root, 'train/unass_candi.test.json')
    unass_candi_train.extend(unass_candi_test)
    save_json(unass_candi_train, processed_data_root, FilePathConfig.unass_candi_offline_path)

    offline_test_profile = load_json(processed_data_root, "train/test_author_profile.json")
    offline_test_unass = load_json(processed_data_root, "train/test_author_unass.json")
    offline_train_profile = load_json(processed_data_root, "train/train_author_profile.json")
    offline_train_unass = load_json(processed_data_root, "train/train_author_unass.json")

    offline_profile = offline_train_profile
    offline_profile.update(offline_test_profile)
    offline_unass = offline_train_unass
    offline_unass.update(offline_test_unass)

    save_json(offline_profile, processed_data_root, "train/offline_profile.json")
    save_json(offline_unass, processed_data_root, "train/offline_unass.json")
    kfold_main_func(offline_profile, offline_unass, 5)

    # 获取测试集需要的name2aid2pid
    get_name2aid2pid(FilePathConfig.whole_name2aid2pid)

    # 获取待分配论文及其需要确定的作者名
    pretreat_unass(FilePathConfig.unass_candi_v1_path, "cna-valid/cna_valid_unass.json",
                   "cna-valid/cna_valid_unass_pub.json")
    pretreat_unass(FilePathConfig.unass_candi_v2_path, "cna-test/cna_test_unass.json",
                   "cna-test/cna_test_unass_pub.json")


if __name__ == '__main__':
    main()
