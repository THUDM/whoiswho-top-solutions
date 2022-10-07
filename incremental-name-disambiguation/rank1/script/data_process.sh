#!bin/bash
time=$(date "+%Y%m%d-%H%M%S")
echo $time

nohup python -u data_process.py 1>> data_process-$time.log 2>&1 &
