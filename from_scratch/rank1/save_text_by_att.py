import codecs
import json
from os.path import join
import pickle
import os
import re
from tqdm import tqdm
try:
    from .utils import *
except Exception:
    from utils import *


puncs = '[!“”"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~—～’]+'
stopwords = ['at', 'based', 'in', 'of', 'for', 'on', 'and', 'to', 'an', 'using', 'with',
             'the', 'by', 'we', 'be', 'is', 'are', 'can']

def replcae_nr(sen: str) -> str:
    sen = sen.replace('\n', ' ').replace('\r', ' ')
    return sen

def process_org(org):
    # save org 待消歧作者的机构名
    pstr = org.strip()
    pstr = pstr.lower()  # 小写
    pstr = re.sub(puncs, ' ', pstr)  # 去除符号
    pstr = re.sub(r'\s{2,}', ' ', pstr).strip()  # 去除多余空格
    pstr = pstr.split(' ')
    pstr = [word for word in pstr if len(word) > 1]
    pstr = [word for word in pstr if word not in stopwords]
    return ' '.join(pstr)


def save_raw_text(pub_files, out_file, att_name, process=False):
    assert att_name in ['title', 'abstract', 'keywords', 'org', 'venue']
    paper_num = 0
    f_out = open(out_file, 'w', encoding='utf-8')
    if att_name == 'org':
        for file in pub_files:
            pubs = load_json(file)
            for pub in tqdm(pubs.values()):
                for author in pub["authors"]:
                    if "org" in author:
                        org = replcae_nr(author['org'])
                        if process:
                            org = process_org(org)
                        f_out.write(org + '\n')
    elif att_name == 'keywords':
        for file in pub_files:
            pubs = load_json(file)
            for pub in tqdm(pubs.values()):
                paper_num += 1
                if 'keywords' in pub:
                    key_str = ' '.join(pub['keywords'])
                    key_str = replcae_nr(key_str)
                    f_out.write(key_str + '\n')
    else:
        for file in pub_files:
            pubs = load_json(file)
            for pub in tqdm(pubs.values()):
                if att_name in pub and type(pub[att_name]) is str:
                    pub[att_name] = replcae_nr(pub[att_name])
                    f_out.write(pub[att_name] + '\n')
    f_out.close()
    print(f'Finish {att} text extract')
    print(f'paper num: {paper_num}')


def remove_dup(file, out_file):
    f2 = open(out_file, 'w')
    last_line = ''
    with open(file) as f:
        for line in f.readlines():
            if line == last_line:
                continue
            else:
                f2.write(line)
                last_line = line
    f2.close()


if __name__ == '__main__':
    base = join(os.environ['DATABASE'], 'WhoIsWho')
    train_pub = join(base, 'train', 'train_pub.json')
    valid_pub = join(base, 'sna-valid', 'sna_valid_pub.json')
    test_pub = join(base, 'sna-test', 'sna_test_pub.json')
    # train_pub_v1 = join(base, 'previous', 'na_v1_pub.json')
    # train_pub_v2 = join(base, 'previous', 'na_v2_pub.json')
    # pub_files = [train_pub_v1, train_pub_v2, train_pub, valid_pub]
    # pub_files = [train_pub, valid_pub, test_pub]
    check_mkdir('./raw_texts')
    atts = ['org']
    for att in atts:
    #     save_raw_text(pub_files, f'./raw_texts/{att}_processed.txt', att, True)
        remove_dup(f'./raw_texts/{att}_processed.txt', f'./raw_texts/{att}_processed_unique.txt')
