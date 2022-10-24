## SAVE all text in the datasets


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


# 提取的文本包括：每个author的org, title, abstract, venue,
def extract_text_save(pub_files, out_file):
    r = '[!“”"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~—～’]+'
    f_out = open(out_file, 'w', encoding='utf-8')
    for file in pub_files:
        pubs = load_json(file)
        for pub in tqdm(pubs.values()):
            for author in pub["authors"]:
                if "org" in author:
                    org = author["org"]
                    pstr = org.strip()
                    pstr = pstr.lower()
                    pstr = re.sub(r, ' ', pstr)
                    pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
                    f_out.write(pstr + '\n')

            title = pub["title"]
            pstr = title.strip()
            pstr = pstr.lower()
            pstr = re.sub(r, ' ', pstr)
            pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
            f_out.write(pstr + '\n')

            if "abstract" in pub and type(pub["abstract"]) is str:
                abstract = pub["abstract"]
                pstr = abstract.strip()
                pstr = pstr.lower()
                pstr = re.sub(r, ' ', pstr)
                pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
                f_out.write(pstr + '\n')

            if "venue" in pub and type(pub["venue"]) is str:
                venue = pub["venue"]
                pstr = venue.strip()
                pstr = pstr.lower()
                pstr = re.sub(r, ' ', pstr)
                pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
                f_out.write(pstr + '\n')

        print(f'File {file} text extracted.')
    f_out.close()


if __name__ == '__main__':
    # 改成自己存放数据的路径
    # base = join(os.environ['DATABASE'], 'WhoIsWho')
    base = "../data"
    
    train_pub = join(base, 'train', 'train_pub.json')
    valid_pub = join(base, 'sna_valid', 'sna_valid_pub.json')
    test_pub = join(base, 'sna_test', 'sna_test_pub.json')
    # train_pub_v1 = join(base, 'previous', 'na_v1_pub.json')
    # train_pub_v2 = join(base, 'previous', 'na_v2_pub.json')
    pub_files = [train_pub, valid_pub, test_pub]

    texts_dir = './extract_texts'
    check_mkdir(texts_dir)
    extract_text_save(pub_files, join(texts_dir, 'train_valid_test.txt'))
