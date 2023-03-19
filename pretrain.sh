#!/bin/bash

# Script to reproduce results

GPU_LIST=(0 1)

env_list=(
	"hopper-medium-v2"
	"hopper-medium-expert-v2"
	"hopper-medium-replay-v2"
	)

for env in ${env_list[*]}; do

GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main.py \
  --env_name $env \
  --pretrain \
  --seed 0 &

sleep 2
let "task=$task+1"

done