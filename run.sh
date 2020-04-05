#!/bin/bash
set -x
read -p "noise percentage: " perc
read -p "baseline (Y/N): " baseline
if [ $baseline = 'Y' ] ||  [ $baseline = 'yes' ] ||  [ $baseline = 'Yes' ]
 then 
 baseline='--baseline'
 else
 baseline=''
 fi

if [ "$perc" != "" ];
then
python add_noise.py ag_news 4 $perc
fi
python train_classifier.py --lstm --dataset data/ag_news --max_epoch 50 $baseline
