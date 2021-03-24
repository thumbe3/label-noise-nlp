#!/bin/bash
set -x
noises=(0.0 0.1 0.2 0.3 0.4 0.5)
warmups=(6 10)
betas=(2 4 8)

for noise in "${noises[@]}"; do
for warmup in "${warmups[@]}"; do
for beta in "${betas[@]}"; do

python add_noise.py --noise $noise --dataset trec --num_classes 6 --mode 3
python train_classifier.py --noise $noise --cnn --dataset data/trec --max_epoch 100 --beta $beta --warmup $warmup --round_prob 0 --result cnn/results_length_noise

done
done
done
