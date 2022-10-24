from gensim.models import word2vec
try:
    from .utils import *
except Exception:
    from utils import *

sentences = word2vec.Text8Corpus(r'extract_texts/train_valid_test.txt')
model = word2vec.Word2Vec(sentences, size=100, negative=5, min_count=5, window=5)
check_mkdir('./word2vec')
model.save('./word2vec/tvt.model')


# if __name__ == '__main__':
#     atts = ['title', 'abstract', 'keywords', 'org', 'venue']
#     sizes = [100, 100, 100, 64, 64]
#     for att, size in tqdm(zip(atts, sizes)):
#         sentences = word2vec.Text8Corpus(f'./text_by_att/{att}.txt')
#         model = word2vec.Word2Vec(sentences, size=size, negative=5, min_count=5, window=5)
#         model.save(f'./word2vec_byatt/{att}_word2vec.model')
#         print(f'Finish {att} word2vec training.')
