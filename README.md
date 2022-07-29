# A Policy-Guided Imitation Approach for Offline Reinforcement Learning

This is the code for reproducing the results of the paper A Policy-Guided Imitation Approach for Offline Reinforcement Learning accepted at NeurIPS'2022. The paper can be found [here](https://arxiv.org/abs/2106.06860).

### Usage
Paper results were collected with [MuJoCo 1.50](http://www.mujoco.org/) (and [mujoco-py 1.50.1.1](https://github.com/openai/mujoco-py)) in [OpenAI gym 0.17.0](https://github.com/openai/gym) with the [D4RL datasets](https://github.com/rail-berkeley/d4rl). Networks are trained using [PyTorch 1.4.0](https://github.com/pytorch/pytorch) and Python 3.6.

The paper results can be reproduced by running:
```
./run_por.sh
```


### Bibtex
```
@inproceedings{xu2022discriminator,
  title = {Discriminator-Weighted Offline Imitation Learning from Suboptimal Demonstrations},
  author = {Haoran Xu and Xianyuan Zhan and Honglei Yin and Huiling Qin},
  booktitle = {International Conference on Machine Learning},
  year = {2022},
}
```

