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

n_rep = 5

configs = []

for task in ['classification', 'binary_classification', 'regression']:
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
                
                
    for seed in range(n_rep):
        for target_depth in [1,2]:
            frozen_depth = 2 * target_depth
            
            config = (
                width, 
                target_depth, 
                frozen_depth, 
                0, # rank, 
                use_bias,
                activation, 
                std, 
                'ours', # method, 
                batch_size, 
                0, # n_epochs
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
                0, # tune_bias,
                0, # last_layers
                seed,
                0, # rank_step
                task,
            )
            configs.append(config)
            
task = 'regression'
target_depth = 1

for seed in range(n_rep):
    for frozen_depth in range(2, 16, 2):
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


new_classification_configs = pd.DataFrame(configs)

if os.path.exists('classification_config.csv'):
    origin_classification_configs = pd.read_csv('classification_config.csv', header=None)
    
    # Find additional configs
    additional_classification_configs = new_classification_configs.merge(origin_classification_configs, how='left', indicator=True).loc[lambda x : x['_merge']=='left_only'].drop('_merge', axis=1).reset_index(drop=True)
    additional_classification_configs.to_csv('additional_classification_configs.csv', index=False, header=False)
    
    while 1:
        # Ask the user if they want to overwrite the original configs
        user_input = input("Do you want to overwrite the original classification configs? (y/n): ")
        if user_input.lower() in ['y', 'yes']:
            new_classification_configs.to_csv('classification_config.csv', index=False, header=False)
            print('Data overwritten!')
            break
        elif user_input.lower() in ['n', 'no']:
            print('Sure!')
            break
        else:
            print("Invalid input!")
else:
    new_classification_configs.to_csv('classification_config.csv', index=False, header=False)