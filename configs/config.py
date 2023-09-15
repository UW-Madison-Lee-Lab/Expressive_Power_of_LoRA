import numpy as np
import pandas as pd
import os

wandb = 1
n_test = 5000
std = .25
batch_size = 256
n_epochs = 5000

fnn_configs = []

# fnn
exp = 'fnn'
for init_mode in ['default', 'uniform_singular_values']:
    # matrix approximation 
    target_depth = 1
    use_bias = 0
    activation = 'linear'
    
    for width in [4, 8, 16]:
        for frozen_depth in [2, 4, 8]:
            for rank in range(1, width//frozen_depth + int(width % frozen_depth > 0) + 1 ):
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
                )
                fnn_configs.append(config)
                
    #fnn approximation
    use_bias = 1
    activation = 'relu'
    
    for target_depth in [1, 2, 4]:
        for width in [4, 8, 16]:
            frozen_depth_list = np.array([2, 4, 8]) * target_depth
            for frozen_depth in frozen_depth_list:
                tdl = frozen_depth // target_depth
                for rank in range(1, width//tdl + int(width % tdl > 0) + 1):
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
tfn_configs = []

for embed_dim in [4, 8]:
    for depth in [1, 2, 4]:
        for rank in range(1, width//2 + int(width % 2 != 0) + 1):
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
                    )
                    tfn_configs.append(config)
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
            )
            tfn_configs.append(config)

new_fnn_configs = pd.DataFrame(fnn_configs)
new_tfn_configs = pd.DataFrame(tfn_configs)

if os.path.exists('fnn_configs.csv'):
    origin_fnn_configs = pd.read_csv('fnn_configs.csv', header=None)
    
    # Find additional configs
    additional_fnn_configs = pd.concat([origin_fnn_configs, new_fnn_configs]).drop_duplicates(keep=False)
    additional_fnn_configs.to_csv('additional_fnn_configs.csv', index=False, header=False)
    
    while 1:
        # Ask the user if they want to overwrite the original configs
        user_input = input("Do you want to overwrite the original confis? (y/n): ")
        if user_input.lower() in ['y', 'yes']:
            break
        elif user_input.lower() in ['n', 'no']:
            print('Exit!')
            exit()
        else:
            print("Invalid input!")
    
if os.path.exists('tfn_configs.csv'):
    origin_tfn_configs = pd.read_csv('tfn_configs.csv', header=None)
    
    # Find additional configs
    additional_tfn_configs = pd.concat([origin_tfn_configs, new_tfn_configs]).drop_duplicates(keep=False)
    additional_tfn_configs.to_csv('additional_tfn_configs.csv', index=False, header=False)
    
    while 1:
        # Ask the user if they want to overwrite the original configs
        user_input = input("Do you want to overwrite the original confis? (y/n): ")
        if user_input.lower() in ['y', 'yes']:
            break
        elif user_input.lower() in ['n', 'no']:
            print('Exit!')
            exit()
        else:
            print("Invalid input!")

            
new_fnn_configs.to_csv('fnn_configs.csv', index=False, header=False)
new_tfn_configs.to_csv('tfn_configs.csv', index=False, header=False)

