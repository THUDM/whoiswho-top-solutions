#!bin/bash
time=$(date "+%Y%m%d-%H%M%S")
echo $time

nohup python -u get_hand_feat.py 1>> get_hand_feat-$time.log 2>&1 &
