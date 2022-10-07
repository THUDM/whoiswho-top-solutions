#!bin/bash
time=$(date "+%Y%m%d-%H%M%S")
echo $time

python -u get_hand_feat.py >> get_hand_feat-$time.log
