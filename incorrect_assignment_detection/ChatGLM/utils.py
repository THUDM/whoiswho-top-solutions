"""
    为了平衡正负样本，在预处理阶段对每一个正样本，采样一个负样本与其对应，对每个负样本也采样一个正样本与其对应（为了保证每条数据不过度拟合，控制负样本采样数量不超过五倍）；
    也可以使用LLM生成outlier样本数据
"""

import json

from sklearn import metrics
import numpy as np
from torch.utils.data import Dataset
import random
import numpy as np
import torch
from dataclasses import dataclass
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers.utils import PaddingStrategy

class INDDataSet(Dataset):
    '''
        iteratively return the profile of each author 
    '''
    def __init__(self, dataset, tokenizer, max_source_length, max_target_length):
        super(INDDataSet, self).__init__()
        self.author, self.pub = dataset  
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.max_seq_length = max_source_length + max_target_length + 1
        author_keys = self.author.keys()
        self.train_keys = []
        for key in author_keys :
            if len(self.author[key]['normal_data']) > len(self.author[key]['outliers']): #用于平衡正负样本
                n = min(len(self.author[key]['normal_data']) - len(self.author[key]['outliers']), 4 *  len(self.author[key]['outliers']))
                neg = np.random.choice(self.author[key]['outliers'], n, replace = True).tolist()

                for i in neg+self.author[key]['outliers']:
                    self.train_keys.append({
                        "pub": i,
                        "author": key
                    }) 
                for i in self.author[key]['normal_data']:
                    self.train_keys.append({
                        "pub": i,
                        "author": key
                    })
            else:
                n = min(len(self.author[key]['outliers']) - len(self.author[key]['normal_data']), 4 *  len(self.author[key]['normal_data']))
                neg = np.random.choice(self.author[key]['normal_data'], n, replace = True).tolist()
                for i in neg+self.author[key]['normal_data']:
                    self.train_keys.append({
                        "pub": i,
                        "author": key   
                    })
                for i in self.author[key]['outliers']:
                    self.train_keys.append({
                        "pub": i,
                        "author": key
                    })
        random.shuffle(self.train_keys)
        for key in author_keys:   
            for pub_key in self.author[key]['normal_data']:
                self.pub[pub_key]['label'] = 1 #表示正样本
                self.pub[pub_key]['author'] = key 

            for pub_key in self.author[key]['outliers']:
                self.pub[pub_key]['label'] = 0
                self.pub[pub_key]['author'] = key 

        with open('/workspace/pangyunhe/source_code/finetune_basemodel_demo/instruction.json','r') as f:
            self.instruct = json.load(f)

        self.yes_token = self.tokenizer.encode(text='yes', add_special_tokens=False, truncation=True,
                                    max_length=self.max_target_length)
        self.no_token = self.tokenizer.encode(text='no', add_special_tokens=False, truncation=True,
                                    max_length=self.max_target_length)
    def __len__(self):
        return len(self.train_keys)

    def __getitem__(self, index):
        profile = self.author[self.train_keys[index]['author']]['normal_data'] +self.author[self.train_keys[index]['author']]['outliers']
        random.shuffle(profile)
        profile = [self.pub[p]['title'] for p in profile]

        #限制context的最长token长度超过max_len-1000则随机忽略一部分样本
        tokenized_profile = [self.tokenizer.tokenize(i) for i in profile]
        len_profile = [len(i) for i in tokenized_profile]
        sum_len = sum(len_profile)
        if sum_len> self.max_source_length-500:
            total_len = 0
            p = 0   
            while total_len < self.max_source_length-1000 and p < sum_len:
                total_len += len_profile[p]
                p += 1
            profile = profile[:p-1]

        profile_text = ' # '.join(profile)
        title = self.pub[self.train_keys[index]['pub']]['title']
        title = title if len(self.tokenizer.tokenize(title))<1000 else '' #防止脏数据导致模型崩溃
        context = self.instruct['instruction_classfication'].format(profile_text,title)

        input_ids = self.tokenizer.encode(text=context, add_special_tokens=True, truncation=True, max_length=self.max_source_length)
        label_ids = self.yes_token if self.pub[self.train_keys[index]['pub']]['label'] == 1 else self.no_token
        input_ids = input_ids + label_ids + [self.tokenizer.eos_token_id]
        labels = [-100] * (len(input_ids)-2) + label_ids + [self.tokenizer.eos_token_id]

        return {
            "input_ids":input_ids,
            "labels":labels,
            "author":self.train_keys[index]['author'],
            "pub":self.train_keys[index]['pub'],
        }

@dataclass
class DataCollatorForIND:
    """
        borrow and modified from transformers.DataCollatorForSeq2Seq
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)
        # breakpoint()
        features = self.tokenizer.pad(
            features,
            padding=True,
            max_length=max_label_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids
        # breakpoint() # [(len(features[i]['input_ids']),len(features[i]['labels'])) for i in range(4)]
        return features    

def split_dict(data_dict, ratio=0.9):
    # 随机打乱字典的键
    keys = list(data_dict.keys())
    random.shuffle(keys)
    
    # 计算划分点
    split_index = int(len(keys) * ratio)
    
    # 分割字典
    train_dict = {key: data_dict[key] for key in keys[:split_index]}
    val_dict = {key: data_dict[key] for key in keys[split_index:]}
    
    return train_dict, val_dict

def weighted_metric(pred:list , label: list) -> float:
    num_pred = [len(i) for i in pred]
    num_label = [len(i) for i in label]
    assert all(len(a) == len(b) for a, b in zip(num_pred, num_label))
    
    acc_pred = [metrics.accuracy_score(l,p) for l,p in zip(label,pred)]

    # abnormal = 0, normal = 1
    num0 = np.array([i.count(0) for i in label])
    weight = num0/np.array(num0.sum())
    weighted_acc = sum(weight * acc_pred)
    
    return weighted_acc

def compute_metrics(ground_truth:dict, res: dict) -> float:
    
    res_list = []
    label_list = []
    
    for author,pubs in ground_truth.items():
        sub_res = res[author]
        keys = pubs['normal_data'] +pubs['outliers']
        label = [1]* len(pubs['normal_data'])+[0]* len(pubs['outliers'])
        
        pred = []
        for i in keys:
            if i in sub_res['normal_data'] and i not in sub_res['outliers']:
                pred.append(1)
            elif i in sub_res['outliers'] and i not in sub_res['normal_data']:
                pred.append(0)
            else:
                # 对于回复异常的文本，直接将其认定为outlier
                pred.append(0)

        res_list.append(pred)
        label_list.append(label)
    acc, f1 = weighted_metric(pred, label)
    return acc, f1

def sanity_check(tokens: List[int], target: List[int], tokenizer: PreTrainedTokenizer):
    print("Sanity Check >>>>>>>>>>>>>")
    for t, m in zip(tokens, target):
        decoded = tokenizer.tokenizer.index_special_tokens[t] \
            if t in tokenizer.tokenizer.index_special_tokens \
            else tokenizer.decode([t])
        if t != 0:
            print("%20s: %6d -> %6d" % (repr(decoded), t, m))
    print("<<<<<<<<<<<<< Sanity Check")
    if not  len(tokens) == len(target):
        breakpoint()
    assert len(tokens) == len(target), f"length mismatch: {len(tokens)} vs {len(target)}"

def preprocess_logits_for_metrics(logits: torch.Tensor, targets: torch.Tensor, return_logits: bool= False) -> torch.Tensor:
   # 计算每个样本中概率最高的一项
   predictions = torch.argmax(logits, dim=-1)
   return predictions


class IND4EVAL(Dataset):
    def __init__(self, dataset, tokenizer, max_source_length, max_target_length):
        super(IND4EVAL, self).__init__()
        self.author,self.pub = dataset  
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.max_seq_length = max_source_length + max_target_length + 1
        author_keys = self.author.keys()

        self.val_set = []
        for key in author_keys:   
            for pub_key in self.author[key]['normal_data']:   
                self.val_set.append({
                    'pub':pub_key,
                    'author':key,
                    'label':1
                }) 
            for pub_key in self.author[key]['outliers']:
                self.val_set.append({
                    'pub':pub_key,
                    'author':key,
                    'label':0
                }) 
        # self.val_keys = []
        # for key in author_keys:   
        #     for pub_key in self.author[key]['normal_data']:
        #         self.pub[pub_key]['label'] = 1 #表示正样本
        #         self.pub[pub_key]['author'] = key 
        #         self.val_keys.append(self.pub[pub_key])
        #     for pub_key in self.author[key]['outliers']:
        #         self.pub[pub_key]['label'] = 0
        #         self.pub[pub_key]['author'] = key
        #         self.val_keys.append(self.pub[pub_key])
        with open('/workspace/pangyunhe/source_code/finetune_basemodel_demo/instruction.json','r') as f:
            self.instruct = json.load(f)    
    def __len__(self):
        return len(self.val_set)
    
    def __getitem__(self, index):
        profile = self.author[self.val_set[index]['author']]['normal_data'] +self.author[self.val_set[index]['author']]['outliers']
        random.shuffle(profile)
        profile = [self.pub[p]['title'] for p in profile]

        #限制context的最长token长度超过max_len-1000则随机忽略一部分样本
        tokenized_profile = [self.tokenizer.tokenize(i) for i in profile]
        len_profile = [len(i) for i in tokenized_profile]
        sum_len = sum(len_profile)
        if sum_len> self.max_source_length-500:
            total_len = 0
            p = 0   
            while total_len < self.max_source_length-1000 and p < sum_len:
                total_len += len_profile[p]
                p += 1
            profile = profile[:p-1]

        profile_text = ' # '.join(profile)
        title = self.pub[self.val_set[index]['pub']]['title']
        title = title if len(self.tokenizer.tokenize(title))<200 else ' '.join(title.split(' ')[:50]) #防止脏数据导致模型崩溃
        context = self.instruct['instruction_classfication'].format(profile_text,title)
        return {
            "input_ids":context,
            "author":self.val_set[index]['author'],
            "pub":self.val_set[index]['pub'],
            "label": self.val_set[index]['label']
        }

# 从数据集中数据中取出author的所有论文
def get_profile(author:[str,list[str]], author_file: dict , pub_file: dict) -> [dict,list[dict]]:
    return_dict = {}

    if isinstance(author, list):
        for a in author:
            return_dict[a] = {
                "normal_data":[pub_file[i] for i in author_file[a]['normal_data']],
                "outliers":[pub_file[i] for i in author_file[a]['outliers']]
            }
    elif isinstance(author, str):
        return_dict[author] = {
            "normal_data":[pub_file[i] for i in author_file[author]['normal_data']],
            "outliers":[pub_file[i] for i in author_file[author]['outliers']]
        }
    else:
        raise TypeError("author must be str or list[str]")
    return return_dict
