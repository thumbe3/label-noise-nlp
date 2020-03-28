#!/bin/bash
set -x
# read -p "noise percentage: " perc
# read -p "baseline (Y/N): " baseline
# if [ $baseline -eq 'Y' ]
# then 
# baseline='--baseline'
# else
# baseline=''
# fi
# echo $baseline

python add_noise.py ag_news 4 0.5
python train_classifier.py --lstm --dataset data/ag_news --max_epoch 30
# python train_classifier.py --lstm --dataset data/ag_news --max_epoch 30 --baseline
