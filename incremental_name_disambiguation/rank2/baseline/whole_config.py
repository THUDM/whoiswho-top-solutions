configs = {
    "train_ins": 4800,
    "test_ins": 300,

    "train_neg_sample":9,
    "test_neg_sample": 19,



    "train_max_papers_each_author": 100,
    "train_min_papers_each_author": 5,
    
    "train_max_semi_len": 24,
    "train_max_whole_semi_len": 256,
    "train_max_per_len":128,
    
    
    "train_max_semantic_len": 64,
    "train_max_whole_semantic_len": 512,
    "train_max_whole_len":512,
    "raw_feature_len": 41,
    "feature_len": 36 + 41,
    "str_len": 36,
    "dl_len": 44,
    "train_knrm_learning_rate": 2e-3,
    "local_accum_step": 32,

    
    
    "hidden_size": 768,
    "n_epoch": 15,
    "show_step" : 1,
    "padding_num": 1,
}
