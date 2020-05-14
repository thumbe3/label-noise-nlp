#!/bin/bash
set -x
noises=(0.1 0.2 0.3 0.4 0.5)
modes=(2 3 4)

for mode in "${modes[@]}"; do

python add_noise.py --noise 0.0 --dataset ag_news --num_classes 4 --mode $mode
python train_classifier.py --noise $mode --cnn --dataset data/ag_news --max_epoch 30 --warmup 7 --round_prob 1 --result cnn/results_instance_specific


#done
#done
#done
done

# 0.1 0.5 0 10-- 92.4, 80.6


