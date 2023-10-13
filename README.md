# Example

## Example 1: FNN w. random
```
python run_exp.py --width 8 --target_depth 1 --frozen_depth 2 --rank 3 --use_bias 1 --activation relu --std 0.25 --method ours --batch_size 256 --n_epochs 5000 --lr 0.0001 --n_test 5000 --weight_decay 0.01 --init_mode uniform_singular_values --exp fnn --wandb 0
```
Output
```
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
| n_head: 2
| seq_length: 10
| exp: fnn
| wandb: 0
Singular values: tensor([0.5686, 0.5686, 0.5686, 0.5686, 0.5686, 0.5686, 0.5686, 0.5686])
Test loss: 0.0445
```

## Example 2: FNN w. pretrained
```
python run_exp.py --width 8 --target_depth 1 --frozen_depth 2 --rank 2 --use_bias 1 --activation relu --std 0.25 --method ours --batch_size 256 --n_epochs 5000 --lr 0.0001 --n_test 5000 --weight_decay 0.01 --init_mode uniform_singular_values --exp fnn --wandb 0 --pretrained 1 --pretrained_epochs 1000 --pretrained_lr .001 --pretrained_level 3
```
Output
```
Pretraining...
Loss of SGD: 0.0497:  15%|█████████████████▎                                                                                                 | 151/1000 [00:00<00:01, 755.48it/s]Pretraining finished at epoch 192.
Loss of SGD: 0.0497:  19%|██████████████████████                                                                                             | 192/1000 [00:00<00:01, 781.32it/s]
Singular values: tensor([0.5390, 0.4101, 0.3990, 0.3265, 0.2287, 0.1750, 0.1188, 0.0557])
Test loss: 0.0071
```

## Example 3: TFN w/o pretrained
```
python run_exp.py --width 8 --target_depth 1 --rank 2  --n_head 2 --batch_size 256 --seq_length 10 --method sgd --n_epochs 800 --lr .001 --weight_decay .01 --exp tfn --wandb 0 --std 0.25 --n_test 5000
```
Output
```
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
| n_head: 2
| seq_length: 10
| exp: tfn
| wandb: 0
Loss of SGD: 2.1334: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 800/800 [00:15<00:00, 52.82it/s]
Validation loss: 2.1712
Test loss: 2.2730
```

## Example 4: TFN w. pretrained 
```
python run_exp.py --width 8 --target_depth 1 --rank 2  --n_head 2 --batch_size 256 --seq_length 10 --method sgd --n_epochs 800 --lr .001 --weight_decay .01 --exp tfn --wandb 0 --std 0.25 --n_test 5000 --pretrained 1 --pretrained_epochs 1000 --pretrained_lr .002 --pretrained_level 3
```
Output
```
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
| pretrained_lr: 0.005
| pretrained_level: 3
| n_head: 2
| seq_length: 10
| exp: tfn
| wandb: 0
Pretraining...
Loss of SGD: 2.3058:   3%|███▌                                                                                                                 | 30/1000 [00:00<00:17, 55.81it/s]Pretraining finished at epoch 35.
Loss of SGD: 2.3058:   4%|████                                                                                                                 | 35/1000 [00:00<00:17, 54.03it/s]
Loss of SGD: 1.8673: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 800/800 [00:14<00:00, 56.25it/s]
Validation loss: 1.8949
Test loss: 1.9799
```

## Example 5: FFN w. pretrained method: lp
```
python run_exp.py --width 8 --target_depth 1 --frozen_depth 2 --rank 2 --use_bias 1 --activation relu --std 0.25 --method lp --batch_size 256 --n_epochs 5000 --lr 0.0001 --n_test 5000 --weight_decay 0.01 --init_mode uniform_singular_values --exp fnn --wandb 0 --pretrained 1 --pretrained_epochs 1000 --pretrained_lr .001 --pretrained_level 3 
```

Output
```
Experiment Setting:
| width: 8
| target_depth: 1
| frozen_depth: 2
| rank: 2
| use_bias: 1
| activation: relu
| std: 0.25
| method: lp
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
| n_head: 2
| seq_length: 10
| exp: fnn
| wandb: 0
Pretraining...
Loss of SGD: 0.0497:  15%|███████████████████████                                                                                                                                   | 150/1000 [00:00<00:01, 744.07it/s]Pretraining finished at epoch 192.
Loss of SGD: 0.0497:  19%|█████████████████████████████▌                                                                                                                            | 192/1000 [00:00<00:01, 768.58it/s]
Loss of SGD: 0.0477: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5000/5000 [00:04<00:00, 1004.20it/s]
Validation loss: 0.0484
Test loss: 0.0469
```