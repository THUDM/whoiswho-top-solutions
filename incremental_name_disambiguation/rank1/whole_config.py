import time
import os

log_time = time.strftime("%m%d_%H%M%S")


data_root = './data/'  # 存放除特征外的数据集
feat_root = './feat/'  # 存放产生的特征

paper_idf_dir = './paper_idf/'  # 存放 paper_idf 文件
pretrained_oagbert_path = "saved/oagbert-v2-sim"

raw_data_root = os.path.join(data_root, 'raw_data/')
processed_data_root = os.path.join(data_root, 'processed_data/')

graph_feat_root = os.path.join(feat_root, 'graph/')
hand_feat_root = os.path.join(feat_root, 'hand/')
bert_feat_root = os.path.join(feat_root, 'bert/')

configs = {
    "train_neg_sample"              : 19,
    "test_neg_sample"               : 19,

    # "train_ins"                     : 9622,
    # "test_ins"                      : 1480,

    "train_max_papers_each_author"  : 100,
    "train_min_papers_each_author"  : 5,

    "train_max_semi_len"            : 24,
    "train_max_whole_semi_len"      : 256,
    "train_max_per_len"             : 128,

    "train_max_semantic_len"        : 64,
    "train_max_whole_semantic_len"  : 512,
    "train_max_whole_len"           : 512,
    "raw_feature_len"               : 41,
    "feature_len"                   : 36 + 41,
    "bertsimi_graph_handfeature_len": 36 + 41 + 41 + 41,
    "str_len"                       : 36,
    "dl_len"                        : 44,
    # "train_knrm_learning_rate"    : 6e-5,
    "train_knrm_learning_rate"      : 2e-3,
    "local_accum_step"              : 32,

    "hidden_size"                   : 768,
    "n_epoch"                       : 15,
    "show_step"                     : 1,
    "padding_num"                   : 1,
}


class FilePathConfig:
    # 线下训练集
    train_name2aid2pid = "train/train_author.json"
    train_pubs = "train/train_pub.json"
    train_name2aid2pid_4train_bert_smi = "train/offline_profile.json"
    # 库中所有候选作者的信息
    database_name2aid2pid = "cna-valid/whole_author_profiles.json"
    database_pubs = "cna-valid/whole_author_profiles_pub.json"
    # 自行生成的文件相对路径
    # 所有已有的 name2aid2pid 信息
    whole_name2aid2pid = 'database/name2aid2pid.whole.json'
    whole_pubsinfo = 'database/pubs.info.json'
    unass_candi_offline_path = 'train/unass_candi.whole.json'
    unass_candi_v1_path = 'onlinev1/unass_candi.json'
    unass_candi_v2_path = 'onlinev2/unass_candi.json'
    unass_pubs_info_v1_path = 'cna-valid/cna_valid_unass_pub.json'
    unass_pubs_info_v2_path = 'cna-test/cna_test_unass_pub.json'
    # feat_dict
    feat_dict_path = 'feat/'
    pid_model_path = 'word2vec/pid_word2vec.model'
    cos_path = {
        'cos'  : graph_feat_root + 'pid2aid2feat_cos.pkl',
        'inner': graph_feat_root + 'pid2aid2feat_inner.pkl',
        'euc'  : graph_feat_root + 'pid2aid2feat_euc.pkl'
    }

    offline_hand_feat_path = hand_feat_root + 'pid2aid2hand_feat.offline.pkl'
    cna_v1_hand_feat_path = hand_feat_root + 'pid2aid2hand_feat.onlinev1.pkl'
    cna_v2_hand_feat_path = hand_feat_root + 'pid2aid2hand_feat.onlinev2.pkl'

    offline_bert_simi_feat_path = bert_feat_root + 'pid2aid2bert_feat.offline.pkl'
    cna_v1_bert_simi_feat_path = bert_feat_root + 'pid2aid2bert_feat.onlinev1.pkl'
    cna_v2_bert_simi_feat_path = bert_feat_root + 'pid2aid2bert_feat.onlinev2.pkl'

    tmp_offline_bert_emb_save_path = bert_feat_root + 'train/'
    tmp_cna_v1_bert_emb_feat_save_path = bert_feat_root + 'online_testv1/'
    tmp_cna_v2_bert_emb_feat_save_path = bert_feat_root + 'online_testv2/'

    tmp_offline_bert_simi_feat_save_path = bert_feat_root + 'train/'
    tmp_cna_v1_bert_simi_feat_save_path = bert_feat_root + 'online_testv1/'
    tmp_cna_v2_bert_simi_feat_save_path = bert_feat_root + 'online_testv2/'

    # offline_bert_simi_feat_path = 'feat/bert_simi_feat/train/bert_simi_0_15872.pkl'
    # cna_v1_bert_simi_feat_path = 'feat/bert_simi_feat/online_testv1/bert_simi_0_13849.pkl'
    # cna_v2_bert_simi_feat_path = 'feat/bert_simi_feat/online_testv2/bert_simi_0_14560.pkl'


def main():
    pass


if __name__ == '__main__':
    main()
