#!/bin/bash
set -x
noises=(0.0 0.1 0.2 0.3 0.4 0.5)

for noise in "${noises[@]}"; do


python add_noise.py --noise $noise --dataset trec --num_classes 6 --mode 3
python train_classifier.py --noise $noise --cnn --dataset data/trec --max_epoch 100 --baseline --result cnn/results_length_noise_baseline


done

# 0.1 0.5 0 10-- 92.4, 80.6


