# 复现说明
## 运行库
Python==3.7.7
gensim==3.8.0
numpy==1.18.1
scikit-learn==0.23.1
tqdm==4.47.0
pinyin==0.4.0 
## 运行步骤
1. 数据集准备：在一个路径下存放下载好的数据并解压：
   /train, /sna-valid, /sna-test; 并修改save_text.py line 62
   和 main.py line 30为对应路径，如 base = '/home/data'
2. 执行 python save_text.py 抽取文本字段
3. 执行 python train_w2v.py 训练并保存word2vec模型
4. 修改main.py中的mode(train/valid/test)，执行 python main.py 
   训练聚类模型并保存日志(training set才有)，和推理结果
5. 运行完毕，在 ./res/output/{train|valid|test} 中得到提交文件
   (有标签的训练集，可在./res/{log|log_summary} 得到细粒度的日志)

## 文件结构
```
├── extract_texts      #（生成）存放提取出的数据集所有文本，用于训练word2vec   
├── gen_names          #（生成）存放每个消歧名字对应的论文集合，方便处理
├── gen_relations      #（生成）存放每个消歧名字对应的关系，用于训练基于元路径的随机游走
├── gen_rw             #（生成）存放游走得到的路径缓存文件
├── other_gen          #（生成）存放每个消歧名字对应的文本特征，文本缺失的离群点
├── res      
      ├── log            #（生成）细粒度日志（仅训练集有）
      ├── log_summary    #（生成）总结日志（仅训练集有）
      └── output         #（生成）提交文件
├── word2vec           #（生成）训练得到的word2vec模型
├── data 
      ├── sna-test       #测试集
      ├── sna-valid      #验证集
      └── train          #训练集
├── main.py            #入口文件，可配置超参
├── rw_w2v_cluster.py  #聚类模型类文件
├── save_text.py       #从数据集中抽取文本字段并保存，用于训练word2vec
├── train_w2v.py       #训练并保存word2vec模型
└── utils.py           #导入库，函数等，包括关键的基于元路径的随机游走和match_name函数
```
### 提交版本用到的代码
* main.py: 入口文件，可配置超参
* rw_w2v_cluster.py: 聚类模型类文件
* save_text.py: 从数据集中抽取文本字段并保存，用于训练word2vec
* train_w2v.py: 训练并保存word2vec模型
* utils.py 导入库，函数等，包括关键的基于元路径的随机游走和match_name函数
### 未用到(仅试错时用过)
* save_text_by_att.py: 按不同字段('title', 'abstract', 'keywords', 'org', 'venue')
分别存储文本
* save_tfidf.py 保存各个词的TF-IDF权重
* text_process_test.py 更细粒度的文本预处理(去掉更多词)

