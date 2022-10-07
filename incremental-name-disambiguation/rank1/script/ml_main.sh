#!bin/bash
time=$(date "+%Y%m%d-%H%M%S")
echo $time

python -u ml_main.py 1>> ml_main-$time.log
