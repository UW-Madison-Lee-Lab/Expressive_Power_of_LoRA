# Example

fnn
```
python run_exp.py --width 8 --target_depth 1 --frozen_depth 2 --rank 3 --use_bias 1 --activation relu --std 0.25 --method ours --batch_size 256 --n_epochs 5000 --lr 0.0001 --n_test 5000 --weight_decay 0.01 --init_mode uniform_singular_values --exp fnn --wandb 0
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
| exp: fnn
| wandb: 0
Singular values: tensor([0.5686, 0.5686, 0.5686, 0.5686, 0.5686, 0.5686, 0.5686, 0.5686])
Test loss: 0.0445
```