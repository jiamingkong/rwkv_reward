# RWKV based reward model

Work in progress, not ready for use.

## Introduction

This git implements a basic reward model based on RWKV models. The purposes of training such a reward model are:

- Show that RWKV models can be used to learn reward functions
- With the reward model trained, we can use it to train a policy using RL algorithms that encourages the base RWKV model to generate more diverse trajectories and high quality answers.

## To run this experiment

### 0. Environments

You will need to install RWKV official pypi package using `pip install rwkv`. You will also need `datasets` and `pytorch` packages.

```
pip install rwkv
pip install datasets
pip install torch
```

### 2. Download the weights

In this experiment I chose a smaller RWKV model with 430M parameters for enabling a quick training and testing on a single GPU. You can download the weight from https://huggingface.co/BlinkDL/rwkv-4-pile-430m

I used this weight: https://huggingface.co/BlinkDL/rwkv-4-pile-430m/resolve/main/RWKV-4-Pile-430M-20220808-8066.pth

### 3. Run the experiment

```
python train.py
```

In this experiment, we will use the reward dataset from https://huggingface.co/datasets/yitingxie/rlhf-reward-datasets. We sampled 100 data points from the train and 20 from the validation to show that the reward model can be trained to predict the reward values.

## The detail of the reward model

In short, the reward model in this setting will 1) read the prompt and output from another LLM model; 2) Rate the output from 1 to -1 ( in logits) for "accept" and "reject" respectively. In the dataset `yitingxie/rlhf-reward-datasets`, for each prompt, two answers were generated by an anonymous LLM, then human raters rated one of the answers as "accept" and the other as "reject". The reward model is trained to predict the human ratings.

Once such a reward model is trained, we can use it to train a policy using RL algorithms that encourages the base RWKV model to get higher scores (i.e. more "accept" ratings).


## Work in progress

1. The reward model is not trained well yet. Only the last few layers are trained.
2. I will implement qLora on top of this reward model to make it fully trainable.
3. The dataset we used for illustration here is not big enough and diverse enough. We will need to collect more data to train a better reward model.