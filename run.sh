#!/bin/bash
export PYTHONPATH=$PWD

repeat=100
for ((i=1;i<=$repeat;i++)); do
    seed=$RANDOM
    echo "Preparing experiment with seed=$seed"
    python3 src/main.py --seed $seed --save-metrics
done
