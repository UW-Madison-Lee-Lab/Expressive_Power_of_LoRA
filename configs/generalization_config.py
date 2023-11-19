import pandas as pd
import os

wandb = 1
n_test = 5000
std = .25
batch_size = 256
n_epochs = 20000
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
n_rep = 5
task = 'regression'
best_epoch = 0

configs = []

for seed in range(n_rep):
    for target_depth in [1,2]:
        frozen_depth = 2 * target_depth
        n_train = 400 * target_depth
        for rank in range(1, width+1):
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
                        n_train,
                        best_epoch,
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
                'inf', # n_train
                0, # best_epoch
            )
            configs.append(config)
    
new_generalization_configs = pd.DataFrame(configs)

if os.path.exists('generalization_config.csv'):
    origin_generalization_configs = pd.read_csv('generalization_config.csv', header=None)
    
    # Find additional configs
    additional_generalization_configs = new_generalization_configs.merge(origin_generalization_configs, how='left', indicator=True).loc[lambda x : x['_merge']=='left_only'].drop('_merge', axis=1).reset_index(drop=True)
    additional_generalization_configs.to_csv('additional_generalization_configs.csv', index=False, header=False)
    
    while 1:
        # Ask the user if they want to overwrite the original configs
        user_input = input("Do you want to overwrite the original generalization configs? (y/n): ")
        if user_input.lower() in ['y', 'yes']:
            new_generalization_configs.to_csv('generalization_config.csv', index=False, header=False)
            print('Data overwritten!')
            break
        elif user_input.lower() in ['n', 'no']:
            print('Sure!')
            break
        else:
            print("Invalid input!")
else:
    new_generalization_configs.to_csv('generalization_config.csv', index=False, header=False)
