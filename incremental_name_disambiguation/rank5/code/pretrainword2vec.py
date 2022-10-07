from gensim.models import Word2Vec
import json as js
import re

# Step 1 : split all the abstracts, merging with title

paperfile=open("../data/na_v3/cna-valid/whole_author_profiles_pub.json","r",encoding="utf-8")

papers=js.load(paperfile)

sentences=[]
i=0
for item in papers:
    # papersent=[]
    # print(papers[item]["title"])
    # print(papers[item]["abstract"])
    i+=1
    al=re.split("\.|,| |:|;|\)|\(|[0-9]|%|\[|\]",papers[item]["title"])
    while '' in al:
        al.remove('')
    sentences.append(al)
    if papers[item]['abstract']!='':
        al=re.split("\.|,| |:|;|\)|\(|[0-9]|%|\[|\]",papers[item]["abstract"])
        while '' in al:
            al.remove('')
        sentences.append(al)
    # if i==100:
        # break
# print(sentences)

paperfile=open("../data/na_v3/train/train_pub.json","r",encoding="utf-8")

papers2=js.load(paperfile)

for item in papers2:
    # papersent=[]
    # print(papers[item]["title"])
    # print(papers[item]["abstract"])
    i+=1
    al=re.split("\.|,| |:|;|\)|\(|[0-9]|%|\[|\]",papers2[item]["title"])
    while '' in al:
        al.remove('')
    sentences.append(al)
    if papers2[item]['abstract']!='':
        al=re.split("\.|,| |:|;|\)|\(|[0-9]|%|\[|\]",papers2[item]["abstract"])
        while '' in al:
            al.remove('')
        sentences.append(al)
    # if i==100:
        # break
# print(sentences)

model=Word2Vec()
model.build_vocab(sentences)
model.train(sentences,total_examples=len(sentences),epochs = 10)

model.save("../train/w2vmodel.model")
model.wv.save_word2vec_format("../train/w2vmodel.model.bin",binary=True,encoding="utf-8")
