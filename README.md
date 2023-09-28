# Example

## Example 1
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

## Example 2
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