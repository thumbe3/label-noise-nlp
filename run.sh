#!/bin/bash
set -x
read -p "noise percentage(decimal): " perc
read -p "baseline (Y/N): " baseline
if [ $baseline = 'Y' ] ||  [ $baseline = 'yes' ] ||  [ $baseline = 'Yes' ]
 then 
 baseline='--baseline'
 else
 baseline=''
 fi

if [ "$perc" != "" ];
then
echo "a"
echo python add_noise.py --noise $perc
python add_noise.py --noise $perc
fi
python train_classifier.py --lstm --dataset data/ag_news --max_epoch 50 $baseline
