#!bin/bash
time=$(date "+%Y%m%d-%H%M%S")
echo $time

nohup python -u get_bert_simi_feat.py --device "cuda:0" --dataset_type "offline_train_validate" --task_type "get_bert_simi_feature" 1>> construct_bert_simi_oftt_all-$time.log 2>&1 &
nohup python -u get_bert_simi_feat.py --device "cuda:0" --dataset_type "online_test_v1" --task_type "get_bert_simi_feature" 1>> construct_bert_simi_otv1_all-$time.log 2>&1 &
nohup python -u get_bert_simi_feat.py --device "cuda:0" --dataset_type "online_test_v2" --task_type "get_bert_simi_feature" 1>> construct_bert_simi_otv2_all-$time.log 2>&1 &
