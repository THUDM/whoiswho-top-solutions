import numpy as np
try:
    from .utils import *
except Exception:
    from utils import *
from glob import glob

try:
    from .text_process_test import TextProcesser
except Exception:
    from text_process_test import TextProcesser
from tqdm import tqdm
import pdb


puncs = '[!“”"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~—～’]+'
stopwords = ['at', 'based', 'in', 'of', 'for', 'on', 'and', 'to', 'an', 'using', 'with',
              'the', 'by', 'we', 'be', 'is', 'are', 'can']
stopwords_extend = ['university', 'univ', 'china', 'department', 'dept', 'laboratory', 'lab',
                     'school', 'al', 'et', 'institute', 'inst', 'college', 'chinese', 'beijing',
                     'journal', 'science', 'international']
stopwords_check = ['a', 'was', 'were', 'that', '2', 'key', '1', 'technology', '0', 'sciences', 'as',
                     'from', 'r', '3', 'academy', 'this', 'nanjing', 'shanghai', 'state', 's', 'research',
                    'p', 'results', 'peoples', '4', 'which', '5', 'high', 'materials', 'study', 'control',
                    'method', 'group', 'c', 'between', 'or', 'it', 'than', 'analysis', 'system',  'sci',
                    'two', '6', 'has', 'h', 'after', 'different', 'n', 'national', 'japan', 'have', 'cell',
                    'time', 'zhejiang', 'used', 'data', 'these']


def compute_idf_from_line_doc(in_files, out_file):
    word2doc_num_contain_it = {}
    doc_num = 0
    for file in in_files:
        with open(file, encoding='utf-8') as f:
            line = f.readline()
            while line:
                doc_num += 1
                words = line.strip().split()
                appeared = set()
                for word in words:
                    if word in appeared:
                        continue
                    if word in word2doc_num_contain_it:
                        word2doc_num_contain_it[word] += 1
                    else:
                        word2doc_num_contain_it[word] = 1
                    appeared.add(word)
                line = f.readline()
    word2idf = {word: np.log(doc_num/(df+1)) for word, df in word2doc_num_contain_it.items()}
    dump_data(word2idf, out_file)
    print('dump idf done')


def compute_idf_from_pub(in_file, out_file, name, if_process=False, if_abstract=True):
    tp = None
    if if_process:
        tp = TextProcesser()
    pubs = load_json(in_file)
    word2doc_num_contain_it = {}
    taken = name.split("_")
    name = taken[0] + taken[1]
    name_reverse = taken[1] + taken[0]
    if len(taken) > 2:
        name = taken[0] + taken[1] + taken[2]
        name_reverse = taken[2] + taken[0] + taken[1]
    authorname_dict = {}

    doc_num = 0
    for pub in pubs.values():
        doc_num += 1
        # save all words' embedding
        org = ""
        for author in pub["authors"]:
            authorname = re.sub(puncs, '', author["name"]).lower()
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
            if authorname == name or authorname == name_reverse:
                if "org" in author:
                    org = author["org"]

        keyword = ""
        if "keywords" in pub:
            for word in pub["keywords"]:
                keyword = keyword + word + " "

        if pub["venue"]:
            pstr = pub["title"] + " " + keyword + " " + pub["venue"] + " " + org
        else:
            pstr = pub["title"] + " " + keyword + " " + org

        if if_abstract and pub['abstract']:
            pstr = pstr + " " + pub["abstract"]

        if "year" in pub:
            pstr = pstr + " " + str(pub["year"])
        pstr = pstr.strip()
        if not if_process:
            pstr = pstr.lower()
            pstr = re.sub(puncs, ' ', pstr)
            pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
            pstr = pstr.split(' ')
            pstr = [word for word in pstr if len(word) > 2]
            pstr = [word for word in pstr if word not in stopwords]
            pstr = [word for word in pstr if word not in stopwords_extend]
            pstr = [word for word in pstr if word not in stopwords_check]
        else:
            pstr = tp.process(pstr)
            pstr = pstr.split()

        words = pstr
        appeared = set()
        for word in words:
            if word in appeared:
                continue
            if word in word2doc_num_contain_it:
                word2doc_num_contain_it[word] += 1
            else:
                word2doc_num_contain_it[word] = 1
            appeared.add(word)
    word2idf = {word: np.log(doc_num / (df + 1)) for word, df in word2doc_num_contain_it.items()}
    dump_data(word2idf, out_file)
    # print('dump idf done')


if __name__ == '__main__':
    # in_files = ['./extract_texts/all_text_processed.txt']
    # compute_idf(in_files, './TFIDF/all_processed_idf.pkl')
    # in_files = ['./extract_texts/valid_text.txt']
    # compute_idf(in_files, './TFIDF/valid_idf.pkl')
    dataset = 'valid'
    name_files = glob(f'./gen_names/{dataset}/*')
    # print(name_files)
    check_mkdir('./TFIDF')
    out_base = f'./TFIDF/{dataset}'
    check_mkdir(out_base)

    for if_abtract in [False, True]:
        for if_process in [False, True]:
            cur_base = f'{out_base}/ab_{int(if_abtract)}_pc_{int(if_process)}'
            for file in tqdm(name_files):
                if not os.path.exists(cur_base):
                    os.mkdir(cur_base)
                name = file.split('/')[-1].split('.')[0]
                # print(name)
                out_file = f'{cur_base}/{name}_idf.pkl'
                compute_idf_from_pub(file, out_file, name, if_abstract=if_abtract, if_process=if_process)
