import os
import random
import sys
import time
from collections import defaultdict
import numpy as np

from whole_config import FilePathConfig, processed_data_root, raw_data_root
from character.feature_process import featureGeneration
from utils import load_json, save_json, load_pickle, save_pickle

# debug_mod = True if sys.gettrace() else False
debug_mod = False


class ProcessFeature:
    def __init__(self, name2aid2pid_path, whole_pub_info_path, unass_candi_path, unass_pubs_path):
        '''

        Args:
            name2aid2pid_path:
            whole_pub_info_path:
            unass_candi_path:
            unass_pubs_path:
        '''
        self.nameAidPid = load_json(name2aid2pid_path)
        self.prosInfo = load_json(whole_pub_info_path)
        self.unassCandi = load_json(unass_candi_path)
        if debug_mod:
            self.unassCandi = self.unassCandi[:5]
        self.validUnassPub = load_json(unass_pubs_path)
        # self.maxNames = 64
        self.maxPapers = 256

    def get_paper_atter(self, pids, pubDict):
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
            if ins_author_index == author_index:
                try:
                    orgnizations = ins_author["org"].strip().lower()
                except:
                    orgnizations = ""

                if orgnizations.strip().lower() != "":
                    org_str = orgnizations
            else:
                try:
                    name = ins_author["name"].strip().lower()
                except:
                    name = ""
                if name != "":
                    name_info.add(name)
        keywords_str = " ".join(keywords_info).strip()
        return name_info, org_str, venue, keywords_str, title

    def getUnassFeat(self):
        tmp = []
        tmpCandi = []
        for insIndex in range(len(self.unassCandi)):
            # if insIndex > 30:
            #     break
            unassPid, candiName = self.unassCandi[insIndex]
            unassAttr = self.get_paper_atter(unassPid, self.validUnassPub)
            candiAuthors = list(self.nameAidPid[candiName].keys())

            tmpCandiAuthor = []
            tmpFeat = []
            for each in candiAuthors:
                totalPubs = self.nameAidPid[candiName][each]
                samplePubs = random.sample(totalPubs, min(len(totalPubs), self.maxPapers))
                candiAttrList = [(self.get_paper_atter(insPub, self.prosInfo)) for insPub in samplePubs]
                tmpFeat.append((unassAttr, candiAttrList))
                tmpCandiAuthor.append(each)

            tmp.append((insIndex, tmpFeat))
            tmpCandi.append((insIndex, unassPid, tmpCandiAuthor))
        return tmp, tmpCandi


def get_hand_feature(config, feat_save_path):
    s_time = time.time()
    genRawFeat = ProcessFeature(**config)
    genFeatures = featureGeneration()
    # 获取待分配论文及候选者信息
    rawFeatData, unassCandiAuthor = genRawFeat.getUnassFeat()
    print('begin multi_process_data')
    hand_feature_list = genFeatures.multi_process_data(rawFeatData)
    print('end multi_process_data')
    assert len(hand_feature_list) == len(unassCandiAuthor)
    pid2aid2cb_feat = defaultdict(dict)
    for hand_feat_item, candi_item in zip(hand_feature_list, unassCandiAuthor):
        ins_index, unass_pid, candi_aids = candi_item
        hand_feat_list, coauthor_ratio = hand_feat_item
        assert len(hand_feat_list) == len(candi_aids)
        for candi_aid, hand_f in zip(candi_aids, hand_feat_list):
            pid2aid2cb_feat[unass_pid][candi_aid] = np.array(hand_f)
    pid2aid2cb_feat = dict(pid2aid2cb_feat)
    print("process data: %.6f" % (time.time() - s_time))
    save_pickle(pid2aid2cb_feat, feat_save_path)


def main():
    # 生成训练集的手工特征
    config = {
        'name2aid2pid_path'  : processed_data_root + 'train/offline_profile.json',
        'whole_pub_info_path': raw_data_root + FilePathConfig.train_pubs,
        'unass_candi_path'   : processed_data_root + FilePathConfig.unass_candi_offline_path,
        'unass_pubs_path'    : raw_data_root + FilePathConfig.train_pubs,
    }
    get_hand_feature(config, FilePathConfig.offline_hand_feat_path)
    # 生成 cna-valid 的手工表征
    config = {
        'name2aid2pid_path'  : processed_data_root + FilePathConfig.whole_name2aid2pid,
        'whole_pub_info_path': processed_data_root + FilePathConfig.whole_pubsinfo,
        'unass_candi_path'   : processed_data_root + FilePathConfig.unass_candi_v1_path,
        'unass_pubs_path'    : raw_data_root + 'cna-valid/cna_valid_unass_pub.json',
    }
    get_hand_feature(config, FilePathConfig.cna_v1_hand_feat_path)
    # 生成 cna-test 的手工表征
    config = {
        'name2aid2pid_path'  : processed_data_root + FilePathConfig.whole_name2aid2pid,
        'whole_pub_info_path': processed_data_root + FilePathConfig.whole_pubsinfo,
        'unass_candi_path'   : processed_data_root + FilePathConfig.unass_candi_v2_path,
        'unass_pubs_path'    : raw_data_root + 'cna-test/cna_test_unass_pub.json',
    }
    get_hand_feature(config, FilePathConfig.cna_v2_hand_feat_path)


if __name__ == '__main__':
    main()
