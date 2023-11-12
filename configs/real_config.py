import os
import pandas as pd

configs = []

for rank in [2,4,6,8,10,12]:
    for pretrained in [0,1]:
        
        model = 'roberta-base'
        
        task = 'cola'
        num_gpus = 4
        configs.append([
            num_gpus,
            task,
            model,
            pretrained,
            rank,
        ])
        
        task = 'mnli'
        num_gpus = 2
        configs.append([
            num_gpus,
            task,
            model,
            pretrained,
            rank,
        ])
        
        task = 'mrpc'
        num_gpus = 2
        configs.append([
            num_gpus,
            task,
            model,
            pretrained,
            rank,
        ])
        
        task = 'qnli'
        num_gpus = 4
        configs.append([
            num_gpus,
            task,
            model,
            pretrained,
            rank,
        ])
        
        task = 'qqp'
        num_gpus = 2
        configs.append([
            num_gpus,
            task,
            model,
            pretrained,
            rank,
        ])
        
        task = 'rte'
        num_gpus = 4
        configs.append([
            num_gpus,
            task,
            model,
            pretrained,
            rank,
        ])
        
        task = 'sst2'
        num_gpus = 2
        configs.append([
            num_gpus,
            task,
            model,
            pretrained,
            rank,
        ])
        
        task = 'stsb'
        num_gpus = 2
        configs.append([
            num_gpus,
            task,
            model,
            pretrained,
            rank,
        ])
        
        model = 'roberta-large'
        
        task = 'cola'
        num_gpus = 2
        configs.append([
            num_gpus,
            task,
            model,
            pretrained,
            rank,
        ])
        
        task = 'mnli'
        num_gpus = 2
        configs.append([
            num_gpus,
            task,
            model,
            pretrained,
            rank,
        ])
        
        task = 'mrpc'
        num_gpus = 2
        configs.append([
            num_gpus,
            task,
            model,
            pretrained,
            rank,
        ])
        
        task = 'qnli'
        num_gpus = 2
        configs.append([
            num_gpus,
            task,
            model,
            pretrained,
            rank,
        ])
        
        task = 'qqp'
        num_gpus = 2
        configs.append([
            num_gpus,
            task,
            model,
            pretrained,
            rank,
        ])
        
        task = 'rte'
        num_gpus = 4
        configs.append([
            num_gpus,
            task,
            model,
            pretrained,
            rank,
        ])
        
        task = 'sst2'
        num_gpus = 4
        configs.append([
            num_gpus,
            task,
            model,
            pretrained,
            rank,
        ])
        
        task = 'stsb'
        num_gpus = 2
        configs.append([
            num_gpus,
            task,
            model,
            pretrained,
            rank,
        ])
        
config_df = pd.DataFrame(configs)
config_df.to_csv('real_config.csv', index=False, header=False)
        
        