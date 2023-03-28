#!/bin/bash

# Script to reproduce mujoco results

GPU_LIST=(0 1)

env_list=(
	"hopper-medium-v2"
	"halfcheetah-medium-v2"
	"walker2d-medium-v2"
	"hopper-medium-replay-v2"
	"halfcheetah-medium-replay-v2"
	"walker2d-medium-replay-v2"
	"hopper-medium-expert-v2"
	"halfcheetah-medium-expert-v2"
	"walker2d-medium-expert-v2"
	)

for seed in 42 79; do

for env in ${env_list[*]}; do

GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main_iql.py \
  --env_name $env \
  --type 'iql' \
  --tau 0.7 \
  --alpha 3.0 \
  --eval_period 5000 \
  --n_eval_episodes 10 \
  --policy_lr 0.001 \
  --seed $seed &

sleep 2
let "task=$task+1"

done

done
