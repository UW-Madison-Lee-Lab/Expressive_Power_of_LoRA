<h1 align="center"> <p>The Expressive Power of Low-Rank Adaptation</p></h1>
<h4 align="center">
    <p><a href="https://yzeng58.github.io/zyc_cv/" target="_blank">Yuchen Zeng</a>, <a href="https://kangwooklee.com/aboutme/" target="_blank">Kangwook Lee</a></p>
    <p>UW-Madison</p>
    </h4>

Paper Link: https://arxiv.org/abs/2310.17513

*Low-Rank Adaptation* (LoRA), a parameter-efficient fine-tuning method that leverages low-rank adaptation of weight matrices, has emerged as a prevalent technique for fine-tuning pre-trained models such as large language models and diffusion models. Despite its huge success in practice, the theoretical underpinnings of LoRA have largely remained unexplored. This paper takes the first step to bridge this gap by theoretically analyzing the expressive power of LoRA. We prove that, for fully connected neural networks, LoRA can adapt any model $f$ to accurately represent any smaller target model $\overline{f}$ if LoRA-rank $\geq(\text{width of }f) \times \frac{\text{depth of }\overline{f}}{\text{depth of }f}$, under a mild assumption. We also quantify the approximation error when the LoRA-rank is lower than the threshold. For Transformer networks, we show any model can be adapted to a target model of the same size with LoRA adapters of rank $(\frac{\text{embedding size}}{2})$. All our theoretical insights are validated by numerical experiments.

## Introduction

Please check the corresponding experiment section in our manuscript for backgrounds.

* `run_exp.py`: this file contains the functions for running the experiments;
* `configs/`: this folder contains the configurations we used in our experiments;
* `results/`: this folder contains the code for downloading our results from wandb and the results themselves;
* `draw_plots/`: this folder contains the the code for visualizing our results.

## Run Customized Experiments

Run `python run_exp.py` with specified setting:

* `--exp`: which model to use ['fnn', 'tfn'];
* `--width` : model width or embedding size $D$;
* `--taget_depth`: depth of the target model $\bar{L}$;
* `--frozen_depth`: depth of the frozen model $L$;
* `--rank`: LoRA-rank $R$;
* `--use_bias`: whether the models consider biases or not (linear models have no biases);
* `--activation`: activation function from ['relu', 'linear'];
* `--std`: standard deviation for generating the weight matrices;
* `--method`: three methods

  * 'ours': computing LoRA adapters using our construction in proof;
  * 'sgd': computing LoRA adapters using gradient update;
  * 'flt': updating the final layers using gradient update;
* `--batch_size`: (i) batch size used for gradient update, and (ii) number of inputs for selecting appropriate large bias in 'ours' (our construction) in the FNN cases;
* `--n_epochs`: the number of epochs used for running gradient updates;
* `--lr`: the learning rate employed in gradient update;
* `--n_test`: the number of test samples for model evaluation;
* `--weight_decay`: the weight decay employed in gradient update;
* `--init_mode`: the mode for initializing the weight matrices,

  * 'default': the weight initialization scheme described in our paper;
  * 'uniform_singular_values': control the singular values of the error matrix $E$ such that it has identical singular values;
* `--pretrained`: whether the frozen model has been pretrained or not;
* `--pretrained_epochs`: the number epochs for pretraining the frozen model;
* `--pretrained_lr`: the learning rate employed when pretraining the frozen model;
* `--pretrained_level`: stop pretraining when the loss $\leq$ (initial loss)/`pretrained_level`;
* `--tune_bias`: whether to tune bias or not in FNN experiments on LoRA;
* `--last_layers`: the number final layers to update in the final layer tuning ('flt') experiments;
* `--seed`: the random seed;
* `--rank_step`: whether to increase the LoRA-rank per layer by `rank_step`, default as 0;
* `--n_head`: the number of attention head $H$;
* `--seq_length`: sequence length $N$;
* `--wandb`: whether upload the results to wandb.

Now, we provide a few examples and the expected outputs.

### Example 1: Approximating one-layer FNN with random multi-layer FNN via LoRA

Command:

```bash
python run_exp.py --width 8 --target_depth 1 --frozen_depth 2 --rank 3 --use_bias 1 --activation relu --std 0.25 --method ours --batch_size 256 --n_epochs 5000 --lr 0.0001 --n_test 5000 --weight_decay 0.01 --init_mode uniform_singular_values --exp fnn --wandb 0
```

Output:

```bash
Experiment Setting:
| width: 8
| target_depth: 1
| frozen_depth: 2
| rank: 3
| use_bias: 1
| activation: relu
| std: 0.25
| method: ours
| batch_size: 256
| n_epochs: 5000
| lr: 0.0001
| n_test: 5000
| weight_decay: 0.01
| init_mode: uniform_singular_values
| pretrained: 0
| pretrained_epochs: 1000
| pretrained_lr: 0.001
| pretrained_level: 3
| tune_bias: 1
| last_layers: 1
| seed: 123
| rank_step: 0
| n_head: 2
| seq_length: 10
| exp: fnn
| wandb: 0
Singular values: tensor([0.5820, 0.5820, 0.5820, 0.5820, 0.5820, 0.5820, 0.5820, 0.5820])
Test loss: 0.0329
```

### Example 2: Approximating one-layer FNN with pretrained multi-layer FNN via LoRA

Command:

```bash
python run_exp.py --width 8 --target_depth 1 --frozen_depth 2 --rank 2 --use_bias 1 --activation relu --std 0.25 --method ours --batch_size 256 --n_epochs 5000 --lr 0.0001 --n_test 5000 --weight_decay 0.01 --init_mode uniform_singular_values --exp fnn --wandb 0 --pretrained 1 --pretrained_epochs 1000 --pretrained_lr .001 --pretrained_level 3
```

Output:

```bash
Experiment Setting:
| width: 8
| target_depth: 1
| frozen_depth: 2
| rank: 2
| use_bias: 1
| activation: relu
| std: 0.25
| method: ours
| batch_size: 256
| n_epochs: 5000
| lr: 0.0001
| n_test: 5000
| weight_decay: 0.01
| init_mode: uniform_singular_values
| pretrained: 1
| pretrained_epochs: 1000
| pretrained_lr: 0.001
| pretrained_level: 3
| tune_bias: 1
| last_layers: 1
| seed: 123
| rank_step: 0
| n_head: 2
| seq_length: 10
| exp: fnn
| wandb: 0
Pretraining...
Loss of SGD: 0.0715:  14%|██████████████▍                                                                                            | 135/1000 [00:00<00:01, 670.50it/s]Pretraining finished at epoch 199.
Loss of SGD: 0.0715:  20%|█████████████████████▎                                                                                     | 199/1000 [00:00<00:01, 671.65it/s]
Singular values: tensor([0.7180, 0.6025, 0.3661, 0.3111, 0.2087, 0.1261, 0.0905, 0.0088])
Test loss: 0.0041
```

### Example 3: Approximating TFN with random TFN via LoRA

Command:

```bash
python run_exp.py --width 8 --target_depth 1 --rank 2  --n_head 2 --batch_size 256 --seq_length 10 --method sgd --n_epochs 800 --lr .001 --weight_decay .01 --exp tfn --wandb 0 --std 0.25 --n_test 5000
```

Output:

```bash
Experiment Setting:
| width: 8
| target_depth: 1
| frozen_depth: 2
| rank: 2
| use_bias: 1
| activation: relu
| std: 0.25
| method: sgd
| batch_size: 256
| n_epochs: 800
| lr: 0.001
| n_test: 5000
| weight_decay: 0.01
| init_mode: default
| pretrained: 0
| pretrained_epochs: 1000
| pretrained_lr: 0.001
| pretrained_level: 3
| tune_bias: 1
| last_layers: 1
| seed: 123
| rank_step: 0
| n_head: 2
| seq_length: 10
| exp: tfn
| wandb: 0
Loss of SGD: 2.1334: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 800/800 [00:16<00:00, 48.79it/s]
Validation loss: 2.1712
Test loss: 2.2730
```

### Example 4: Approximating TFN with pretrained TFN via LoRA

Command:

```bash
python run_exp.py --width 8 --target_depth 1 --rank 2  --n_head 2 --batch_size 256 --seq_length 10 --method sgd --n_epochs 800 --lr .001 --weight_decay .01 --exp tfn --wandb 0 --std 0.25 --n_test 5000 --pretrained 1 --pretrained_epochs 1000 --pretrained_lr .002 --pretrained_level 3
```

Output:

```bash
Experiment Setting:
| width: 8
| target_depth: 1
| frozen_depth: 2
| rank: 2
| use_bias: 1
| activation: relu
| std: 0.25
| method: sgd
| batch_size: 256
| n_epochs: 800
| lr: 0.001
| n_test: 5000
| weight_decay: 0.01
| init_mode: default
| pretrained: 1
| pretrained_epochs: 1000
| pretrained_lr: 0.002
| pretrained_level: 3
| tune_bias: 1
| last_layers: 1
| seed: 123
| rank_step: 0
| n_head: 2
| seq_length: 10
| exp: tfn
| wandb: 0
Pretraining...
Loss of SGD: 2.2482:  10%|███████████▎                                                                                                | 105/1000 [00:02<00:17, 51.11it/s]Pretraining finished at epoch 110.
Loss of SGD: 2.2482:  11%|███████████▉                                                                                                | 110/1000 [00:02<00:16, 53.82it/s]
Loss of SGD: 1.8310: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 800/800 [00:14<00:00, 54.11it/s]
Validation loss: 1.8559
Test loss: 1.9348
```

### Example 5: Tuning Final Layers

Command:

```bash
python run_exp.py --width 8 --target_depth 1 --frozen_depth 2 --rank 2 --use_bias 1 --activation relu --std 0.25 --method flt --batch_size 256 --n_epochs 5000 --lr 0.0001 --n_test 5000 --weight_decay 0.01 --init_mode uniform_singular_values --exp fnn --wandb 0 --pretrained 1 --pretrained_epochs 1000 --pretrained_lr .001 --pretrained_level 3 --last_layers 1
```

Output:

```bash
Experiment Setting:
| width: 8
| target_depth: 1
| frozen_depth: 2
| rank: 2
| use_bias: 1
| activation: relu
| std: 0.25
| method: flt
| batch_size: 256
| n_epochs: 5000
| lr: 0.0001
| n_test: 5000
| weight_decay: 0.01
| init_mode: uniform_singular_values
| pretrained: 1
| pretrained_epochs: 1000
| pretrained_lr: 0.001
| pretrained_level: 3
| tune_bias: 1
| last_layers: 1
| seed: 123
| rank_step: 0
| n_head: 2
| seq_length: 10
| exp: fnn
| wandb: 0
Pretraining...
Loss of SGD: 0.0715:  15%|███████████████▌                                                                                           | 146/1000 [00:00<00:01, 725.42it/s]Pretraining finished at epoch 199.
Loss of SGD: 0.0715:  20%|█████████████████████▎                                                                                     | 199/1000 [00:00<00:01, 729.73it/s]
Loss of SGD: 0.0523: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 5000/5000 [00:04<00:00, 1023.34it/s]
Validation loss: 0.0554
Test loss: 0.0545
```

## Run a Group of Experiments

In `configs/`, we have `config.py` for specifying the group of experiments to run, and the group of configurations are saved into `fnn_configs.csv` and `tfn_configs.csv` once you run `python config.py`.

If you want to further add additional experiment configurations, you just need to add the corresponding code into `config.py`, and run `python config.py`. The additional configurations will be saved into `additional_fnn_configs.csv` and `additional_tfn_configs.csv`.

If you changed the code of the experiment functions, and want to rerun some experiments, you can specify the configurations to be rerun in `config.py`. The corresponding configurations will be stored into `update_fnn_configs.csv` and `update_tfn_configs.csv` by running `python config.py`.

If there are some failed jobs, you can run `python failed_config.py` to collect the corresponding configurations, and they are saved in `rerun_fnn_configs.csv` and `rerun_tfn.csv`.

## Results Visualization

By running the notebooks in `draw_plots`, you can obtain the figures in our paper.
