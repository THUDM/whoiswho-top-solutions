import os
import sys
import copy
import logging
import random
import time
from collections import defaultdict
import multiprocessing

import numpy as np
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from whole_config import FilePathConfig, log_time, processed_data_root
from utils import set_log, load_pickle, save_pickle, load_json, save_json

xgb_njob = int(multiprocessing.cpu_count() / 2)

# debug_mod = True if sys.gettrace() else False
debug_mod = False


def random_select_instance(train_ins, nil_ratio=0.2, min_neg_num=10):
    '''

    Args:
        train_ins: 输入的实例。 会进行深拷贝，不会改变原始值
        nil_ratio: NIL 实例的比例
        min_neg_num: 每个实例最少的候选者个数

    Returns:

    '''
    # 改变传入的集合，从中采样出部分用于处理 nil 问题的分类器
    train_ins_num = len(train_ins)
    res_list = []
    nil_ins_max = int(train_ins_num * nil_ratio)
    random.shuffle(train_ins)
    neg_ins = train_ins[0:nil_ins_max]  # 出现 NIL 问题的示例
    for ins in neg_ins:
        if len(ins[3]) >= min_neg_num:
            ins[2] = ''
            res_list.append(ins)
    pos_ins = train_ins[nil_ins_max:]
    for ins in pos_ins:
        if len(ins[3]) >= min_neg_num:
            ins[3] = random.sample(ins[3], len(ins[3]) - 1)
            res_list.append(ins)
    random.shuffle(res_list)
    return res_list


def get_gbd_model(gbd_type='xgb', njob=xgb_njob, model_args=None):
    '''

    Args:
        gbd_type: ['xgb', 'cat', 'lgbm']
        njob: 多线程数
        model_args: 不按默认方式初始化时传入的参数

    Returns:

    '''
    assert gbd_type in ['xgb', 'cat', 'lgbm']
    gbd_model = None
    if gbd_type == 'xgb':
        default_config = {
            'max_depth'       : 7,
            'learning_rate'   : 0.01,
            'n_estimators'    : 1000,
            'subsample'       : 0.8,
            'n_jobs'          : njob,
            'min_child_weight': 6,
            'random_state'    : 666
        }
        if model_args is not None:
            default_config.update(model_args)
        gbd_model = XGBClassifier(**default_config)
    elif gbd_type == 'cat':
        default_config = {
            'iterations'   : 1000,
            'learning_rate': 0.05,
            'depth'        : 10,
            'loss_function': 'Logloss',
            'eval_metric'  : 'Logloss',
            'random_seed'  : 666,
        }
        if model_args is not None:
            default_config.update(model_args)
        gbd_model = CatBoostClassifier(**default_config)
    elif gbd_type == 'lgbm':
        default_config = {
            'max_depth'    : 10,
            'learning_rate': 0.01,
            'n_estimators' : 1000,
            'objective'    : 'binary',
            'subsample'    : 0.8,
            'n_jobs'       : njob,
            'num_leaves'   : 82,
            'random_state' : 666
        }
        if model_args is not None:
            default_config.update(model_args)
        gbd_model = LGBMClassifier(**default_config)
    return gbd_model


def get_gbd_pred(gbd_model, feat, gbd_type='xgb'):
    assert gbd_type in ['xgb', 'cat', 'lgbm']
    if gbd_type in ['xgb']:
        feat = np.array(feat)
        res = gbd_model.predict_proba(feat)[:, 1]
    elif gbd_type in ['lgbm']:
        feat = np.array(feat)
        res = gbd_model.predict_proba(feat)[:, 1]
    elif gbd_type in ['cat']:
        res = gbd_model.predict_proba(feat)[:, 1]
    else:
        raise NotImplementedError
    return res


def fit_gbd_model(gbd_model, whole_x, whole_y, gbd_type='xgb'):
    assert gbd_type in ['xgb', 'cat', 'lgbm']
    if gbd_type in ['xgb', 'cat', 'lgbm']:
        gbd_model.fit(whole_x, whole_y)
    else:
        raise NotImplementedError


class FeatDataLoader:
    def __init__(self, feat_config):
        self.cos_path = feat_config['cos_path']
        self.euc_path = feat_config['euc_path']
        self.inner_path = feat_config['inner_path']
        self.bert_path = feat_config['bert_path']

        self.hand_dict = load_pickle(feat_config['hand_path'])
        self.bert_dict = None
        self.cos_dict = None
        self.euc_dict = None
        self.inner_dict = None

    def update_feat(self, feat_list):
        if 'bert' in feat_list and self.bert_dict is None:
            self.bert_dict = load_pickle(self.bert_path)
        if 'cos' in feat_list and self.cos_dict is None:
            self.cos_dict = load_pickle(self.cos_path)
        if 'euc' in feat_list and self.euc_dict is None:
            self.euc_dict = load_pickle(self.euc_path)
        if 'inner' in feat_list and self.inner_dict is None:
            self.inner_dict = load_pickle(self.inner_path)

    def get_whole_feat(self, unass_pid, candi_aid, feat_list):
        hand_feat = self.hand_dict[unass_pid][candi_aid]
        if 'bert' in feat_list:
            whole_feature = np.hstack([self.bert_dict[unass_pid][candi_aid], hand_feat])
        else:
            whole_feature = np.array(hand_feat)
        if 'cos' in feat_list:
            whole_feature = np.hstack([whole_feature, self.cos_dict[unass_pid][candi_aid]])
        if 'euc' in feat_list:
            whole_feature = np.hstack([whole_feature, self.euc_dict[unass_pid][candi_aid]])
        if 'inner' in feat_list:
            whole_feature = np.hstack([whole_feature, self.inner_dict[unass_pid][candi_aid]])
        return whole_feature


class CellModel:
    def __init__(self, model_config, kfold):
        # print(model_config)
        assert len(model_config) <= 2
        lv1_model_list = []
        lv2_model_list = []
        lv1_gdb_type = []
        lv2_gdb_type = []
        for i in range(kfold):
            this_fold_model_list = []
            for t_config in model_config[0]:
                if 'params' in t_config and len(t_config['params'].keys()) > 0:
                    this_fold_model_list.append(
                        get_gbd_model(t_config['gbd_type'], njob=xgb_njob, model_args=t_config['params']))
                else:
                    this_fold_model_list.append(get_gbd_model(t_config['gbd_type'], njob=xgb_njob))
                if i == 0:
                    lv1_gdb_type.append(t_config['gbd_type'])
            lv1_model_list.append(this_fold_model_list)
        if len(model_config) == 2 and len(model_config[1]) > 0:
            for t_config in model_config[1]:
                if 'params' in t_config and len(t_config['params'].keys()) > 0:
                    lv2_model_list.append(
                        get_gbd_model(t_config['gbd_type'], njob=xgb_njob, model_args=t_config['params']))
                else:
                    lv2_model_list.append(get_gbd_model(t_config['gbd_type'], njob=xgb_njob))
                lv2_gdb_type.append(t_config['gbd_type'])
        self.lv1_model_list = lv1_model_list
        self.lv2_model_list = lv2_model_list
        self.lv1_gdb_type = lv1_gdb_type
        self.lv2_gdb_type = lv2_gdb_type
        self.kfold = kfold
        self.has_lv2 = True if len(lv2_gdb_type) > 0 else False

    def fit(self, whole_x, whole_y, training_type, fold_i=None):
        # training_type 决定是训练第一层模型还是第二层模型
        print('\tfitting ', training_type)
        print('\t\twhole_x.shape ', whole_x.shape)
        print('\t\twhole_y.shape ', whole_y.shape)
        assert training_type == 'lv2' or fold_i is not None
        if training_type == 'lv1':
            for lv1_model, gdb_type in zip(self.lv1_model_list[fold_i], self.lv1_gdb_type):
                fit_gbd_model(lv1_model, whole_x, whole_y, gdb_type)
            return
        elif training_type == 'lv2':
            assert self.has_lv2
            for lv2_model, gdb_type in zip(self.lv2_model_list, self.lv2_gdb_type):
                fit_gbd_model(lv2_model, whole_x, whole_y, gdb_type)
            return
        raise ValueError('illegal training_type ', training_type)

    def _get_lv1_preds(self, candis_feature, fold_i):
        preds = None
        for lv1_model, gdb_type in zip(self.lv1_model_list[fold_i], self.lv1_gdb_type):
            if preds is None:
                preds = get_gbd_pred(lv1_model, candis_feature, gdb_type)[:, np.newaxis]
            else:
                preds = np.hstack([preds, get_gbd_pred(lv1_model, candis_feature, gdb_type)[:, np.newaxis]])
        return preds

    def train_model(self, train_config_list, train_feat_data, cell_feat_list):
        step_two_train_x = None
        step_two_train_y = []

        for fold_i, train_config in enumerate(train_config_list):
            # 先做第一阶段训练
            log_msg = f'\n\ntraing fold {fold_i + 1}'
            print(log_msg)
            logging.warning(log_msg)
            if debug_mod:
                train_ins = load_json(train_config['train_path'])[:200]
                # train_ins = train_ins
                dev_ins = load_json(train_config['dev_path'])[:200]
            else:
                train_ins = load_json(train_config['train_path'])
                dev_ins = load_json(train_config['dev_path'])
            whole_x = []
            whole_y = []
            for _, unass_pid, pos_aid, neg_aids in train_ins:
                for neg_aid in neg_aids:
                    feat = train_feat_data.get_whole_feat(unass_pid, neg_aid, cell_feat_list)
                    whole_x.append(feat)
                    whole_y.append(0)
                feat = train_feat_data.get_whole_feat(unass_pid, pos_aid, cell_feat_list)
                whole_x.append(feat)
                whole_y.append(1)
            whole_x = np.array(whole_x)
            whole_y = np.array(whole_y)
            self.fit(whole_x, whole_y, 'lv1', fold_i)
            if self.has_lv2:
                # 产生第二阶段训练数据
                tmp_dev_ins = copy.deepcopy(dev_ins)
                new_dev_ins = random_select_instance(tmp_dev_ins, 0.2)
                for _, unass_pid, pos_aid, neg_aids in new_dev_ins:
                    step_one_feat = []
                    for neg_aid in neg_aids:
                        feat = train_feat_data.get_whole_feat(unass_pid, neg_aid, cell_feat_list)
                        step_one_feat.append(feat)
                        step_two_train_y.append(0)
                    if pos_aid != '':
                        feat = train_feat_data.get_whole_feat(unass_pid, pos_aid, cell_feat_list)
                        step_one_feat.append(feat)
                        step_two_train_y.append(1)
                    step_two_feat = self.get_lv2_feat(step_one_feat, fold_i)
                    if step_two_train_x is None:
                        step_two_train_x = step_two_feat
                    else:
                        step_two_train_x = np.vstack([step_two_train_x, step_two_feat])
                del tmp_dev_ins
        # 产生验证集第二阶段使用数据
        if self.has_lv2:
            step_two_train_x = np.array(step_two_train_x)
            step_two_train_y = np.array(step_two_train_y)
            assert len(step_two_train_x) == len(step_two_train_y)
            self.fit(step_two_train_x, step_two_train_y, training_type='lv2')

    def get_lv2_feat(self, candis_feature, fold_i):
        # 获取用于二阶段预测和训练的特征, 传入当前文章所有候选作者拼成的表征
        assert self.has_lv2, '有二阶段时才能调用此函数'
        candi_num = len(candis_feature)
        assert candi_num > 0
        lv1_preds_all = self._get_lv1_preds(candis_feature, fold_i)  # [candi_num, lv_1_model_num]
        score_feat_all = None  # [4 * lv_1_model_num]
        for i in range(len(self.lv1_gdb_type)):
            lv1_preds = lv1_preds_all[:, i]
            max_score = np.max(lv1_preds)
            if candi_num > 1:
                min_score = np.min(lv1_preds)
                mean_score = np.mean(lv1_preds)
                lv1_preds[np.argmax(lv1_preds)] = np.min(lv1_preds)
                second_score = np.max(lv1_preds)
                score_feat = [max_score, mean_score,
                              round((max_score - second_score) / (1e-8 + max_score - mean_score), 5),
                              round((max_score - second_score) / (1e-8 + max_score - min_score), 5)]
            else:
                score_feat = [max_score, max_score, 0, 0]
            score_feat = np.array(score_feat).reshape(1, 4).repeat(candi_num, axis=0)
            if score_feat_all is None:
                score_feat_all = score_feat
            else:
                score_feat_all = np.hstack([score_feat_all, score_feat])

        candis_feature = np.hstack([candis_feature, score_feat_all])
        return candis_feature

    def predict(self, candis_feature):
        ''' 传入所有候选者的表征 '''
        # candis_feature = np.array(candis_feature)
        if self.has_lv2:
            lv2_feat_all = []
            for fold_i in range(self.kfold):
                lv2_feat = self.get_lv2_feat(candis_feature, fold_i)  # [candi_num,feat_dim ]
                # lv2_feat = lv2_feat[:, np.newaxis, :]  # [candi_num,1,feat_dim ]
                lv2_feat_all.append(lv2_feat)
            lv2_feat = np.mean(lv2_feat_all, axis=0)
            preds = None
            for lv2_model, gdb_type in zip(self.lv2_model_list, self.lv2_gdb_type):
                pred = get_gbd_pred(lv2_model, lv2_feat, gdb_type)[:, np.newaxis]
                if preds is None:
                    preds = pred
                else:
                    preds = np.hstack([preds, pred])
            preds = np.mean(preds, axis=1)
            return preds

        preds_all = None
        for fold_i in range(self.kfold):
            preds = self._get_lv1_preds(candis_feature, fold_i)  # [candi_num, step_one_model_num]
            preds = np.mean(preds, axis=1)[:, np.newaxis]  # [candi_num]
            if preds_all is None:
                preds_all = preds
            else:
                preds_all = np.hstack([preds_all, preds])
        preds_all = np.mean(preds_all, axis=1)
        return preds_all


def get_cell_pred(cell_model, unass_pid2aid, eval_feat_data, cell_feat_list):
    unass_pid2aid2score = defaultdict(dict)
    for unass_pid, candi_aids in unass_pid2aid:
        candi_feat = []
        for candi_aid in candi_aids:
            candi_feat.append(eval_feat_data.get_whole_feat(unass_pid, candi_aid, cell_feat_list))
        candi_preds = cell_model.predict(candi_feat)
        for candi_index, candi_aid in enumerate(candi_aids):
            unass_pid2aid2score[unass_pid][candi_aid] = float(candi_preds[candi_index])
    return unass_pid2aid2score


def train_cell_model_as_stacking(cell_config, train_config_list, train_feat_data: FeatDataLoader,
                                 cell_save_root, cell_index: int):
    '''训练每个cell '''
    cell_feat_list = cell_config['feature_list']
    cell_model = CellModel(cell_config['model'], len(train_config_list))
    cell_model.train_model(train_config_list, train_feat_data, cell_feat_list)
    save_pickle(cell_model, cell_save_root, f'cell-{cell_index}.pkl')
    return cell_model


def deal_nil_threshold_new(score_path, save_dir, info, thres=0.7):
    res_unass_aid2score_list = load_json(score_path)
    result = defaultdict(list)
    ass_papers = 0
    max_score = -1
    for pid, aid2score in res_unass_aid2score_list.items():
        tmp_scores = []
        for aid, score in aid2score.items():
            tmp_scores.append((aid, score))
        if len(tmp_scores) == 0:
            continue
        tmp_scores.sort(key=lambda x: x[1], reverse=True)
        max_score = max(max_score, tmp_scores[0][1])
        if tmp_scores[0][1] >= thres:
            ass_papers += 1
            result[tmp_scores[0][0]].append(pid.split('-')[0])
    log_msg = f'ass_papers= {ass_papers}, max_score={max_score}.'
    print(log_msg)
    logging.warning(log_msg)
    save_json(dict(result), save_dir, f'result.{info}.json')
    save_json(dict(result), f'result.{info}.json')


def get_result(cell_model, unass_pid2aid, eval_feat_data, cell_config, cell_i, res_unass_aid2score_list,
               cell_weight_sum, model_save_dir, info):
    this_cell_res = defaultdict(dict)
    unass_pid2aid_cell_i = get_cell_pred(cell_model, unass_pid2aid, eval_feat_data, cell_config['feature_list'])

    for unass_pid, unass_aids in unass_pid2aid:
        for unass_aid in unass_aids:
            this_cell_res[unass_pid][unass_aid] = unass_pid2aid_cell_i[unass_pid][unass_aid]
            if cell_i == 0:
                res_unass_aid2score_list[unass_pid][unass_aid] = unass_pid2aid_cell_i[unass_pid][unass_aid] * \
                                                                 cell_config['cell_weight'] / cell_weight_sum
            else:
                res_unass_aid2score_list[unass_pid][unass_aid] += unass_pid2aid_cell_i[unass_pid][unass_aid] * \
                                                                  cell_config['cell_weight'] / cell_weight_sum
    save_json(dict(this_cell_res), model_save_dir, f'result_score_cell{cell_i}.{info}.json')


def test_config2data(test_config):
    ''' 将测试配置转换为需要的中间变量 '''
    eval_feat_data = FeatDataLoader(test_config)
    unass_list = load_json(test_config['unass_path'])
    unass_name2aid2pid_v1 = load_json(test_config['name2aid2pid'])
    unass_pid2aid = []  # [pid, [candi_aid,]]
    # 获取测试集
    if debug_mod:
        unass_list = unass_list[:40]
    for unass_pid, name in unass_list:
        candi_aids = list(unass_name2aid2pid_v1[name])
        unass_pid2aid.append((unass_pid, candi_aids))
    return eval_feat_data, unass_pid2aid


def train_stack_model(config):
    os.makedirs('logs', exist_ok=True)
    set_log('logs', log_time)
    train_config_list = config['train_config_list']  # 训练数据列表
    train_feature_config = config['train_feature_config']  # 训练特征字典列表

    model_save_dir = config['model_save_dir']  # 模型保存路径
    cell_list_config = config['cell_list_config']  # 模型配置
    os.makedirs(model_save_dir, exist_ok=True)
    save_json(config, model_save_dir, 'models_train_config.json')
    train_feat_data = FeatDataLoader(train_feature_config)
    test_config_v1 = config['test_config_v1']  # 测试配置
    eval_feat_data_v1, unass_pid2aid_v1 = test_config2data(test_config_v1)

    test_config_v2 = config['test_config_v2']  # 测试配置
    eval_feat_data_v2, unass_pid2aid_v2 = test_config2data(test_config_v2)

    # 模型训练
    cell_weight_sum = 0
    for cell_config in cell_list_config:
        cell_weight_sum += cell_config['cell_weight']

    res_unass_aid2score_list_v1 = defaultdict(dict)
    res_unass_aid2score_list_v2 = defaultdict(dict)

    for cell_i, cell_config in enumerate(cell_list_config):
        # 先训练每个cell
        s_time = time.time()
        log_msg = f'\n\nbegin to train cell {cell_i + 1}.'
        print(log_msg)
        logging.warning(log_msg)
        train_feat_data.update_feat(cell_config['feature_list'])
        eval_feat_data_v1.update_feat(cell_config['feature_list'])
        eval_feat_data_v2.update_feat(cell_config['feature_list'])
        if 'train_config_list' in cell_config:
            in_train_config_list = cell_config['train_config_list']
        else:
            in_train_config_list = train_config_list
        cell_model = train_cell_model_as_stacking(cell_config, in_train_config_list, train_feat_data, model_save_dir,
                                                  cell_i + 1)
        log_msg = f'it cost {round(time.time() - s_time, 3)} s to train.'
        print(log_msg)
        logging.warning(log_msg)

        get_result(cell_model, unass_pid2aid_v1, eval_feat_data_v1, cell_config, cell_i, res_unass_aid2score_list_v1,
                   cell_weight_sum, model_save_dir, 'v1')
        get_result(cell_model, unass_pid2aid_v2, eval_feat_data_v2, cell_config, cell_i, res_unass_aid2score_list_v2,
                   cell_weight_sum, model_save_dir, 'v2')

    score_result_path_v1 = os.path.join(model_save_dir, 'result_score_vote.v1.json')
    save_json(dict(res_unass_aid2score_list_v1), score_result_path_v1)
    deal_nil_threshold_new(
        score_result_path_v1, model_save_dir, 'v1', 0.65
    )

    score_result_path_v2 = os.path.join(model_save_dir, 'result_score_vote.v2.json')
    save_json(dict(res_unass_aid2score_list_v2), score_result_path_v2)
    deal_nil_threshold_new(
        score_result_path_v2, model_save_dir, 'v2', 0.65
    )


def main():
    # 训练文件路径配置
    train_config_list = [
        {
            'train_path': processed_data_root + "/train/kfold_dataset/kfold_v1/train_ins.json",
            'dev_path'  : processed_data_root + "/train/kfold_dataset/kfold_v1/test_ins.json",
        },
        {
            'train_path': processed_data_root + "/train/kfold_dataset/kfold_v2/train_ins.json",
            'dev_path'  : processed_data_root + "/train/kfold_dataset/kfold_v2/test_ins.json",
        },
        {
            'train_path': processed_data_root + "/train/kfold_dataset/kfold_v3/train_ins.json",
            'dev_path'  : processed_data_root + "/train/kfold_dataset/kfold_v3/test_ins.json",
        },
        {
            'train_path': processed_data_root + "/train/kfold_dataset/kfold_v4/train_ins.json",
            'dev_path'  : processed_data_root + "/train/kfold_dataset/kfold_v4/test_ins.json",
        },
        {
            'train_path': processed_data_root + "/train/kfold_dataset/kfold_v5/train_ins.json",
            'dev_path'  : processed_data_root + "/train/kfold_dataset/kfold_v5/test_ins.json",
        },
    ]
    # 测试配置
    test_config_v1 = {  # 产生 cna-valid 结果的配置
        'cos_path'    : FilePathConfig.cos_path['cos'],
        'euc_path'    : FilePathConfig.cos_path['euc'],
        'inner_path'  : FilePathConfig.cos_path['inner'],
        'bert_path'   : FilePathConfig.cna_v1_bert_simi_feat_path,
        'hand_path'   : FilePathConfig.cna_v1_hand_feat_path,
        'unass_path'  : processed_data_root + FilePathConfig.unass_candi_v1_path,
        'name2aid2pid': processed_data_root + FilePathConfig.whole_name2aid2pid
    }
    test_config_v2 = {  # 产生 cna-valid 结果的配置
        'cos_path'    : FilePathConfig.cos_path['cos'],
        'euc_path'    : FilePathConfig.cos_path['euc'],
        'inner_path'  : FilePathConfig.cos_path['inner'],
        'bert_path'   : FilePathConfig.cna_v2_bert_simi_feat_path,
        'hand_path'   : FilePathConfig.cna_v2_hand_feat_path,
        'unass_path'  : processed_data_root + FilePathConfig.unass_candi_v2_path,
        'name2aid2pid': processed_data_root + FilePathConfig.whole_name2aid2pid
    }
    # 配置模型结构
    final_cell_list_config = [
        {
            'cell_weight' : 5,
            'score'       : 0.0,
            'feature_list': ['bert'],
            'vote_type'   : 'mean',
            'model'       : [
                [
                    {
                        'gbd_type': 'lgbm',
                        'params'  : {}
                    }
                ],
                [
                ]
            ],
        },
        {
            'cell_weight' : 5,
            'score'       : 0.0,
            'feature_list': ['bert'],
            'vote_type'   : 'mean',
            'model'       : [
                [
                    {
                        'gbd_type': 'cat',
                        'params'  : {'verbose': False}
                    }
                ],
                [
                ]
            ],
        },
        {
            'cell_weight' : 5,
            'score'       : 0.0,
            'feature_list': ['bert'],
            'vote_type'   : 'mean',
            'model'       : [
                [
                    {
                        'gbd_type': 'xgb',
                        'params'  : {}
                    }
                ],
                [
                ]
            ],
        },
        {
            'cell_weight' : 5,
            'score'       : 0.0,
            'feature_list': [],
            'vote_type'   : 'mean',
            'model'       : [
                [
                    {
                        'gbd_type': 'lgbm',
                        'params'  : {}
                    }
                ],
                [
                ]
            ],
        },
        {
            'cell_weight' : 5,
            'score'       : 0.0,
            'feature_list': [],
            'vote_type'   : 'mean',
            'model'       : [
                [
                    {
                        'gbd_type': 'cat',
                        'params'  : {'verbose': False}
                    }
                ],
                [
                ]
            ],
        },
        {
            'cell_weight' : 5,
            'score'       : 0.0,
            'feature_list': [],
            'vote_type'   : 'mean',
            'model'       : [
                [
                    {
                        'gbd_type': 'xgb',
                        'params'  : {}
                    }
                ],
                [
                ]
            ],
        },
        {
            'cell_weight'      : 1,
            'train_config_list': [train_config_list[0]],
            'score'            : 0.0,
            'feature_list'     : ['bert'],
            'vote_type'        : 'mean',
            'model'            : [
                [
                    {
                        'gbd_type': 'cat',
                        'params'  : {'verbose': False}
                    }
                ],
                [
                    {
                        'gbd_type': 'cat',
                        'params'  : {'verbose': False}
                    }
                ]
            ],
        },
        {
            'cell_weight'      : 1,
            'train_config_list': [train_config_list[0]],
            'score'            : 0.0,
            'feature_list'     : ['bert'],
            'vote_type'        : 'mean',
            'model'            : [
                [
                    {
                        'gbd_type': 'lgbm',
                        'params'  : {}
                    }
                ],
                [
                    {
                        'gbd_type': 'lgbm',
                        'params'  : {}
                    }
                ]
            ],
        },
        {
            'cell_weight'      : 1,
            'train_config_list': [train_config_list[0]],
            'score'            : 0.0,
            'feature_list'     : ['bert'],
            'vote_type'        : 'mean',
            'model'            : [
                [
                    {
                        'gbd_type': 'xgb',
                        'params'  : {}
                    }
                ],
                [
                    {
                        'gbd_type': 'xgb',
                        'params'  : {}
                    }
                ]
            ],
        },
        {
            'cell_weight'      : 1,
            'train_config_list': [train_config_list[0]],
            'score'            : 0.0,
            'feature_list'     : [],
            'vote_type'        : 'mean',
            'model'            : [
                [
                    {
                        'gbd_type': 'cat',
                        'params'  : {'verbose': False}
                    }
                ],
                [
                    {
                        'gbd_type': 'cat',
                        'params'  : {'verbose': False}
                    }
                ]
            ],
        },
        {
            'cell_weight'      : 1,
            'train_config_list': [train_config_list[0]],
            'score'            : 0.0,
            'feature_list'     : [],
            'vote_type'        : 'mean',
            'model'            : [
                [
                    {
                        'gbd_type': 'lgbm',
                        'params'  : {}
                    }
                ],
                [
                    {
                        'gbd_type': 'lgbm',
                        'params'  : {}
                    }
                ]
            ],
        },
        {
            'cell_weight'      : 1,
            'train_config_list': [train_config_list[0]],
            'score'            : 0.0,
            'feature_list'     : [],
            'vote_type'        : 'mean',
            'model'            : [
                [
                    {
                        'gbd_type': 'xgb',
                        'params'  : {}
                    }
                ],
                [
                    {
                        'gbd_type': 'xgb',
                        'params'  : {}
                    }
                ]
            ],
        },
    ]

    stack_model_name = 'stack-model'
    models_train_config = {
        'name'                : stack_model_name,
        'train_config_list'   : train_config_list,
        'train_feature_config': {
            'cos_path'  : FilePathConfig.cos_path['cos'],
            'euc_path'  : FilePathConfig.cos_path['euc'],
            'inner_path': FilePathConfig.cos_path['inner'],
            'bert_path' : FilePathConfig.offline_bert_simi_feat_path,
            'hand_path' : FilePathConfig.offline_hand_feat_path,
        },
        'test_config_v1'      : test_config_v1,
        'test_config_v2'      : test_config_v2,
        'model_save_dir'      : os.path.join(f'save_model/{log_time}.{stack_model_name}'),
        'cell_list_config'    : final_cell_list_config,
    }

    print(log_time)
    os.makedirs('save_model', exist_ok=True)
    train_stack_model(models_train_config)


if __name__ == '__main__':
    main()
