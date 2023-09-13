import numpy as np
import pandas as pd

wandb = 1
n_test = 5000
std = .25
batch_size = 256
n_epochs = 5000

configs = []

# fnn
exp = 'fnn'
for init_mode in ['default', 'uniform_singular_values']:
    # matrix approximation 
    target_depth = 1
    use_bias = 0
    activation = 'linear'
    
    for width in [4, 8, 16]:
        for frozen_depth in [2, 4, 8]:
            for rank in range(1, width//frozen_depth + int(width % frozen_depth > 0) ):
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
                )
                configs.append(config)
                
    #fnn approximation
    use_bias = 1
    activation = 'relu'
    
    for target_depth in [1, 2, 4]:
        for width in [4, 8, 16]:
            frozen_depth_list = np.array([2, 4, 8]) * target_depth
            for frozen_depth in frozen_depth_list:
                tdl = frozen_depth // target_depth
                for rank in range(1, width//tdl + int(width % tdl > 0) ):
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
                    )
                    configs.append(config)
                    
pd.DataFrame(configs).to_csv('fnn_configs.csv', index=False, header=False)