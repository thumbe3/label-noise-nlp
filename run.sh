#!/bin/bash
read -p "noise percentage: " perc
read -p "baseline (Y/N): " baseline
if [ $baseline -eq 'Y' ]
then 
baseline='--baseline'
else
baseline=''
fi
echo $baseline

python noise.py Trec/ 6 $perc
python trial_classifier.py --lstm --dataset Data/Trec --max_epoch $baseline
