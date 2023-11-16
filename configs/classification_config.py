import pandas as pd
import os

wandb = 1
n_test = 5000
std = .25
batch_size = 256
n_epochs = 5000
width = 16

pretrained_epochs = 1000
pretrained_lr = 1e-3
pretrained_level = 3
init_mode = 'default'

pretrained = 0
tune_bias = 1
use_bias = 1
activation = 'relu'

exp = 'fnn'
task = 'classification'

n_rep = 5

configs = []

for seed in range(n_rep):
    for target_depth in [1,2]:
        frozen_depth = 2 * target_depth
        for rank in range(1, width+1):
            
            # sgd
            method = 'sgd'
            for lr in [1e-2, 1e-3, 1e-4]:
                for weight_decay in [0, 1e-4, 1e-3, 1e-2]:
                    config = (
                        width, 
                        target_depth, 
                        frozen_depth, 
                        rank, 
                        use_bias,
                        activation, 
                        std, 
                        method, 
                        batch_size, 
                        n_epochs,
                        lr,
                        n_test,
                        weight_decay,
                        init_mode, 
                        exp,
                        wandb,
                        pretrained,
                        pretrained_epochs,
                        pretrained_lr,
                        pretrained_level,
                        tune_bias,
                        0, # last_layers
                        seed,
                        0, # rank_step
                        task,
                    )
                    configs.append(config)
                    
            # ours
            method = 'ours'
            config = (
                width, 
                target_depth, 
                frozen_depth, 
                rank, 
                use_bias,
                activation, 
                std, 
                method, 
                batch_size, 
                n_epochs,
                lr,
                n_test,
                weight_decay,
                init_mode, 
                exp,
                wandb,
                pretrained,
                pretrained_epochs,
                pretrained_lr,
                pretrained_level,
                1, # tune_bias
                0, # last_layers
                seed,
                0, # rank_step
                task,
            )
            configs.append(config)

config_df = pd.DataFrame(configs)
config_df.to_csv('classification_config.csv', index=False, header=False)