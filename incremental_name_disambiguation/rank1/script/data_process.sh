#!bin/bash
time=$(date "+%Y%m%d-%H%M%S")
echo $time

python -u data_process.py >> data_process-$time.log
