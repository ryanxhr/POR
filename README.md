# A Policy-Guided Imitation Approach for Offline Reinforcement Learning

This is the code for the paper A Policy-Guided Imitation Approach for Offline Reinforcement Learning accepted as **oral** at NeurIPS'2022. The paper and slide can be found at [paper](https://arxiv.org/abs/2210.08323) and [slide](https://docs.google.com/presentation/d/1swZTLDSvZLGCrXs46tzSHLWZC6VfO9qYChegjjadCpc/edit#slide=id.g170ea50d4c3_9_42). 

Policy-guided Offline RL (POR) is a new offline RL paradigm, it enables **state-stitching** from the dataset rather than **action-stitching** as conducted in prior offline RL methods. POR enjoys training stability by using *in-sample* learning while still allowing logical *out-of-sample* generalization.
We hope that POR could shed light on how to enable state-stitching in offline RL, which connects well to goal-conditioned RL and hierarchical RL. 

*Note that the copyright of our original code belongs to the previous company I worked for and the current github version is our re-implementation. Although we try to reimplement it as similar to the original version as possible, there exists some minor mismatch from the paper results (except for the "hopper-medium-v2" datasest, where our re-implementation version achieves a 78.6 score, not able to reproduce 98.2 in the white paper).*

### Usage
Mujoco reuslts can be reproduced by first running `./pretrain.sh` and then running `./run_mujoco.sh`, Antmaze results can be reproduced by running `./run_antmaze.sh`. See our running results [here](https://wandb.ai/ryanxhr/POR_paper?workspace=user-ryanxhr).

### Bibtex
```
@inproceedings{xu2022policyguided,
  title  = {A Policy-Guided Imitation Approach for Offline Reinforcement Learning},
  author = {Haoran Xu and Li Jiang and Jianxiong Li and Xianyuan Zhan},
  year   = {2022},
  booktitle = {Advances in Neural Information Processing Systems},
}
```

