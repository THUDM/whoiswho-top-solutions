#!bin/bash
time=$(date "+%Y%m%d-%H%M%S")
echo $time

python -u get_graph_simi_feat.py >> get_graph_simi_feat-$time.log
