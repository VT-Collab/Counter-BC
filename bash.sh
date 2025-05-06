#!/bin/bash

# number of demonstrations
# each demo (interaction) contains 20 datapoints
# so 5 demos corresponds to 100 state-action pairs
demos=(5 10 20 40 80)
# number of end-to-end runs per number of demonstrations
count=10

# parameters used in Figure 4, Uniform
for demos_size in "${demos[@]}"; do
  for ((i=1; i<=count; i++)); do
    python get_data.py --n_interactions $demos_size --noise_type uniform --sigma 0.5 --delta 0.5
    python train_bc.py
    python train_ileed.py
    python train_sasaki.py
    python train_counter-bc.py
    python test.py --savename ${demos_size}.csv
  done
done