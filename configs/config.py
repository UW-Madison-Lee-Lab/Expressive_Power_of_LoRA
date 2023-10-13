import numpy as np
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


fnn_configs = []
# when the original code was updated, we want to rerun the experiments for the following update_configs
update_fnn_configs = []

# fnn
exp = 'fnn'
for init_mode in ['default', 'uniform_singular_values']:
    # matrix approximation 
    target_depth = 1
    use_bias = 0
    activation = 'linear'
    
    for pretrained in [0, 1]:
        for frozen_depth in [2, 4, 8]:
            for rank in range(1, width + 1):
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
                            1, # tune_bias
                        )
                        fnn_configs.append(config)
                        
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
                )
                fnn_configs.append(config)
                
    #fnn approximation
    use_bias = 1
    activation = 'relu'
    
    for pretrained in [0, 1]:
        for target_depth in [1, 2, 4]:
            frozen_depth_list = np.array([2, 4, 8]) * target_depth
            for frozen_depth in frozen_depth_list:
                tdl = frozen_depth // target_depth
                for rank in range(1, width + 1):
                    for tune_bias in [0, 1]:
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
                                )
                                fnn_configs.append(config)
                
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
                    )
                    fnn_configs.append(config)

    target_depth = 1          
    for pretrained in [0, 1]:
            frozen_depth_list = np.array([2, 4, 8]) * target_depth
            for frozen_depth in frozen_depth_list:
                tdl = frozen_depth // target_depth
                for last_layers in range(1, frozen_depth):
                    # final layers tuning
                    method = 'flt'
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
                                1, # tune_bias 
                                last_layers,
                            )
                            fnn_configs.append(config)
                            
# tfn
exp = 'tfn'
n_head = 2
seq_len = 10
wandb = 1
n_test = 5000
std = .25
batch_size = 256
n_epochs = 5000
embed_dim = 16

pretrained_epochs = 1000
pretrained_lr = 1e-3
pretrained_level = 3

tfn_configs = []
# when the original code was updated, we want to rerun the experiments for the following update_configs
update_tfn_configs = []

for pretrained in [0, 1]:
    for depth in [1, 2, 4]:
        for rank in range(1, width + 1):
            # sgd
            method = 'sgd'
            for lr in [1e-2, 1e-3, 1e-4]:
                for weight_decay in [0, 1e-4, 1e-3, 1e-2]:
                    config = (
                        embed_dim,
                        n_head, 
                        depth,
                        rank,
                        batch_size,
                        seq_len,
                        method,
                        n_epochs,
                        lr,
                        weight_decay,
                        wandb,
                        std,
                        n_test,
                        exp,
                        pretrained,
                        pretrained_epochs,
                        pretrained_lr,
                        pretrained_level,
                    )
                    tfn_configs.append(config)
                    update_tfn_configs.append(config)
                    
            # ours
            method = 'ours'
            config = (
                embed_dim,
                n_head, 
                depth,
                rank,
                batch_size,
                seq_len,
                method,
                n_epochs,
                lr,
                weight_decay,
                wandb,
                std,
                n_test,
                exp,
                pretrained,
                pretrained_epochs,
                pretrained_lr,
                pretrained_level,
            )
            tfn_configs.append(config)

new_fnn_configs = pd.DataFrame(fnn_configs)
new_tfn_configs = pd.DataFrame(tfn_configs)

update_fnn_configs = pd.DataFrame(update_fnn_configs)
update_tfn_configs = pd.DataFrame(update_tfn_configs)
update_fnn_configs.to_csv('update_fnn_configs.csv', index=False, header=False)
update_tfn_configs.to_csv('update_tfn_configs.csv', index=False, header=False)

if os.path.exists('fnn_configs.csv'):
    origin_fnn_configs = pd.read_csv('fnn_configs.csv', header=None)
    
    # Find additional configs
    additional_fnn_configs = new_fnn_configs.merge(origin_fnn_configs, how='left', indicator=True).loc[lambda x : x['_merge']=='left_only'].drop('_merge', axis=1).reset_index(drop=True)
    additional_fnn_configs.to_csv('additional_fnn_configs.csv', index=False, header=False)
    
    while 1:
        # Ask the user if they want to overwrite the original configs
        user_input = input("Do you want to overwrite the original fnn configs? (y/n): ")
        if user_input.lower() in ['y', 'yes']:
            new_fnn_configs.to_csv('fnn_configs.csv', index=False, header=False)
            print('Data overwritten!')
            break
        elif user_input.lower() in ['n', 'no']:
            print('Sure!')
            break
        else:
            print("Invalid input!")
else:
    new_fnn_configs.to_csv('fnn_configs.csv', index=False, header=False)
    
if os.path.exists('tfn_configs.csv'):
    origin_tfn_configs = pd.read_csv('tfn_configs.csv', header=None)
    
    # Find additional configs
    additional_tfn_configs = new_tfn_configs.merge(origin_tfn_configs, how='left', indicator=True).loc[lambda x : x['_merge']=='left_only'].drop('_merge', axis=1).reset_index(drop=True)
    additional_tfn_configs.to_csv('additional_tfn_configs.csv', index=False, header=False)
    
    while 1:
        # Ask the user if they want to overwrite the original configs
        user_input = input("Do you want to overwrite the original tfn confis? (y/n): ")
        if user_input.lower() in ['y', 'yes']:
            new_tfn_configs.to_csv('tfn_configs.csv', index=False, header=False)
            print('Data overwritten!')
            break
        elif user_input.lower() in ['n', 'no']:
            print('Sure!')
            break
        else:
            print("Invalid input!")
else:    
    new_tfn_configs.to_csv('tfn_configs.csv', index=False, header=False)

