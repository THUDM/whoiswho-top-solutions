代码执行顺序：
step1_data_prepare.py 数据准备，将数据存储为相应的格式，便于后面的操作。
step2_split_data.py 用于将完整的训练数据切分成“现有数据库”和“新增论文”两部分，模拟赛道的真实场景
step3_extract_base_feature.py 用于构造训练集和测试集的基础特征，包括待分配论文的标题、作者姓名、作者单位、会议期刊、发表年份、关键词、摘要等，以及相应的同名作者的用户档案。特别注意的是，考虑到作者姓名的书写格式比较混乱，本文首先以已知作者数据中的名字格式为标准，对每篇论文的合作者姓名，进行标准化处理，如小写转换，去空格加下划线等。此外，考虑到测试集中存在大量找不到候选用户文档的情况，本文设计了一个用于计算姓名差异的函数，当待分配论文作者与候选用户的姓名差异得分存在小于2的情况时，便将将候选用户文档分配给待分配论文作者，否则将姓名差异最小的前两个候选用户文档分配给当前待分配的论文。
step4_extract_coauthors_feature.py 用于构造用户合作者的相关特征，包括所有共同作者的数量、共同作者数量占所有作者的比例、共同作者数占当前论文作者数的比例、一篇文章共同作者的数量、 一篇文章共同作者占所有共同作者的比例等等。
step5_extract_org_feature.py 用于构造与作者机构名称相关的特征，包括作者机构名称与作者以前的机构名称的最大Jaro–Winkler distance、作者机构名称与作者以前的机构名称的平均Jaro–Winkler distance、作者机构名称与作者以前的机构名称的最大字符集合得分、作者机构名称与作者以前的机构名称的平均字符集合得分等等。
step6_extract_word2vec_sim_feature.py 用于构造title、abstract、keywords和venue相关的相似度特征。以计算abstract的相似度为例，首先对数据进行清洗，包括小写转换、去除特殊字符等，然后利用word2vec训练一个128维的词向量，计算每篇paper关于摘要的词向量，最后利用距离公式计算两个paper的余弦相似度。
step7_extract_weight_word2vec_sim_feature.py 用于构造关于时间加权的相似度特征。一般来说，学者的研究方向不是一层不变的，可能会随时间有所改变，时间越近的两篇paper的相关性可能会更高，基于此，本文提出了基于时间加权的论文相似度计算方法，以论文发表时间差的倒数作为权重。
step8_extract_text_countvec_sim_feature.py 利用CountVectorizer计算文本的相似性。

step9_extract_weight_countvec_sim_feature.py 同样用于构造关于时间加权的相似度特征。// TOHERE
step10_extract_text_tfidfvec_sim_feature.py 利用TfidfVectorizer 计算文本的相似性。
step11_extract_weight_tfidfvec_sim_feature.py 同样用于构造关于时间加权的相似度特征。
step12_extract_keywords_abstract_feature.py 统计待分配论文关键词在候选用户文档论文集中出现的次数。
step13_extract_year_feature.py 构造与论文发表时间相关的特征，如发表年份与数据集中年分最小值之间的差、发表年份与数据集中年分最大值之间的差、数据集中年分的平均值、发表年份出现的次数等等
step14_extract_set_feature.py 将论文的title、abstract和venue视为集合，分别计算待分配论文与候选用户文档论文集的交集个数，并求相应的聚合特征。此外，本文还计算了两个文本的Jaro–Winkler distance。
run_lightgbm.py 利用lightgbm模型对测试集进行预测。
run_xgboost.py 利用xgboost模型对测试集进行预测
merge.py 模型融合