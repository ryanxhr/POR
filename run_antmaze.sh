#!/bin/bash

# Script to reproduce antmaze results

GPU_LIST=(0 1)

env_list=(
	"antmaze-umaze-v2"
	"antmaze-umaze-diverse-v2"
	"antmaze-medium-play-v2"
	"antmaze-medium-diverse-v2"
	"antmaze-large-play-v2"
	"antmaze-large-diverse-v2"
	)

for seed in 42; do
for env in ${env_list[*]}; do

GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main.py \
  --env_name $env \
  --type 'por_r' \
  --tau 0.9 \
  --alpha 10.0 \
  --eval_period 10000 \
  --n_eval_episodes 50 \
  --seed $seed &

sleep 2
let "task=$task+1"

done
done