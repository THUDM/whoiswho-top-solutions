# -*- coding: utf-8 -*-
import time
import pypinyin
import numpy as np
from collections import defaultdict
from contextlib import contextmanager

import warnings
warnings.filterwarnings("ignore")

@contextmanager
def timer(task_name="timer"):
    print("-- {} started".format(task_name))
    t0 = time.time()
    yield
    print("-- {} done in {:.0f} seconds".format(task_name, time.time() - t0))
    

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage before optimization is: {:.2f} MB'.format(start_mem))
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


def to_pinyin(word):
    s = ''
    for i in pypinyin.pinyin(word, style=pypinyin.NORMAL):
        s += ''.join(i)
    return s
           

def check_chs(c):
    return '\u4e00' <= c <= '\u9fa5'

def convert_name(x):
    # 去除 \xa0
    x = x.translate(dict.fromkeys((ord(c) for c in u"\xa0")))
    
    if "," in x:
        x = x.replace(","," ")
    if x == "qiang 郭强":
        x = "qiang guo"
    if x == "Gang Liu 刘钢":
        x = "Gang Liu"
    if x == "汪之国 Wang Zhiguo":
        x = "Wang Zhiguo"
    if x == "shucai li 李术才":
        x = "shucai li"
    if x == "舒桦 Shu Hua":
        x = "Shu Hua"
    if x == "Lei Zhang 0006":
        x = "Lei Zhang"
    if x == "Min Zhang 0005":
        x = "Min Zhang"
    if x == "Min Zhang 0006":
        x = "Min Zhang"
    if x == "Yong Li 0008":
        x = "Yong Li"
    if x == "ying 胡英":
        x = "ying hu"
    if x == "duan li 李端":
        x = "duan li"
    if x == "peng zhou 周 鹏":
        x = "peng zhou"
    if x == "cheng li 李成":
        x = "cheng li"
    if x == "ユタカ ササキ":
        x = "Yutaka Sakaki"
    if x == "タダシ ナカムラ":
        x = "Tadayoshi Nakamura"
    if x == "タクヤ マツモト":
        x = "Matsumoto takatake"
    
    # 中文翻译成英文
    n = ''.join(filter(str.isalpha, x.lower()))
    if check_chs(n):
        x = to_pinyin(x)
        
    return x

def distance_score(n1, n2):
    n1 = ''.join(filter(str.isalpha, n1.lower()))
    if check_chs(n1):
        n1 = to_pinyin(n1)
    n2 = ''.join(filter(str.isalpha, n2.lower()))
    counter = defaultdict(int)
    score = 0
    for c in n1:
        counter[c] += 1
    for c in n2:
        if (c in counter) and (counter[c] > 0):
            counter[c] -= 1
        else:
            score += 1
    score += np.sum(list(counter.values()))
    return score

