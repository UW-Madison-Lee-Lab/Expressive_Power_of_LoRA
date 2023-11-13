import os
import pandas as pd

configs = []

for rank in [2,4,6,8,10,12]:
    for pretrained in [0,1]:
        
        model = 'roberta-base'
        
        task = 'cola'
        num_gpus = 1
        cuda_capability_min = 8.0
        cuda_capability_max = 9.0
        job_length = 'short'
        configs.append([
            num_gpus,
            task,
            model,
            pretrained,
            rank,
            cuda_capability_min,
            cuda_capability_max,
            job_length,
        ])
        
        task = 'mnli'
        num_gpus = 1
        cuda_capability_min = 8.0
        cuda_capability_max = 8.0
        job_length = 'medium'
        configs.append([
            num_gpus,
            task,
            model,
            pretrained,
            rank,
            cuda_capability_min,
            cuda_capability_max,
            job_length,
        ])
        
        task = 'mrpc'
        num_gpus = 1
        cuda_capability_min = 8.0
        cuda_capability_max = 9.0
        job_length = 'short'
        configs.append([
            num_gpus,
            task,
            model,
            pretrained,
            rank,
            cuda_capability_min,
            cuda_capability_max,
            job_length,
        ])
        
        task = 'qnli'
        num_gpus = 1
        cuda_capability_min = 8.0
        cuda_capability_max = 9.0
        job_length = 'short'
        configs.append([
            num_gpus,
            task,
            model,
            pretrained,
            rank,
            cuda_capability_min,
            cuda_capability_max,
            job_length,
        ])
        
        task = 'qqp'
        num_gpus = 1
        cuda_capability_min = 8.0
        cuda_capability_max = 8.0
        job_length = 'medium'
        configs.append([
            num_gpus,
            task,
            model,
            pretrained,
            rank,
            cuda_capability_min,
            cuda_capability_max,
            job_length,
        ])
        
        task = 'rte'
        num_gpus = 1
        cuda_capability_min = 8.0
        cuda_capability_max = 9.0
        job_length = 'short'
        configs.append([
            num_gpus,
            task,
            model,
            pretrained,
            rank,
            cuda_capability_min,
            cuda_capability_max,
            job_length,
        ])
        
        task = 'sst2'
        num_gpus = 1
        cuda_capability_min = 8.0
        cuda_capability_max = 8.0
        job_length = 'medium'
        configs.append([
            num_gpus,
            task,
            model,
            pretrained,
            rank,
            cuda_capability_min,
            cuda_capability_max,
            job_length,
        ])
        
        task = 'stsb'
        num_gpus = 1
        cuda_capability_min = 8.0
        cuda_capability_max = 9.0
        job_length = 'short'
        configs.append([
            num_gpus,
            task,
            model,
            pretrained,
            rank,
            cuda_capability_min,
            cuda_capability_max,
            job_length,
        ])
        
        model = 'roberta-large'
        
        task = 'cola'
        num_gpus = 1
        cuda_capability_min = 8.0
        cuda_capability_max = 9.0
        job_length = 'short'
        configs.append([
            num_gpus,
            task,
            model,
            pretrained,
            rank,
            cuda_capability_min,
            cuda_capability_max,
            job_length,
        ])
        
        task = 'mnli'
        num_gpus = 1
        cuda_capability_min = 8.0
        cuda_capability_max = 8.0
        job_length = 'medium'
        configs.append([
            num_gpus,
            task,
            model,
            pretrained,
            rank,
            cuda_capability_min,
            cuda_capability_max,
            job_length,
        ])
        
        task = 'mrpc'
        num_gpus = 1
        cuda_capability_min = 8.0
        cuda_capability_max = 9.0
        job_length = 'short'
        configs.append([
            num_gpus,
            task,
            model,
            pretrained,
            rank,
            cuda_capability_min,
            cuda_capability_max,
            job_length,
        ])
        
        task = 'qnli'
        num_gpus = 1
        cuda_capability_min = 8.0
        cuda_capability_max = 9.0
        job_length = 'medium'
        configs.append([
            num_gpus,
            task,
            model,
            pretrained,
            rank,
            cuda_capability_min,
            cuda_capability_max,
            job_length,
        ])
        
        task = 'qqp'
        num_gpus = 1
        cuda_capability_min = 8.0
        cuda_capability_max = 8.0
        job_length = 'medium'
        configs.append([
            num_gpus,
            task,
            model,
            pretrained,
            rank,
            cuda_capability_min,
            cuda_capability_max,
            job_length,
        ])
        
        task = 'rte'
        num_gpus = 1
        cuda_capability_min = 8.0
        cuda_capability_max = 9.0
        job_length = 'short'
        configs.append([
            num_gpus,
            task,
            model,
            pretrained,
            rank,
            cuda_capability_min,
            cuda_capability_max,
            job_length,
        ])
        
        task = 'sst2'
        num_gpus = 1
        cuda_capability_min = 8.0
        cuda_capability_max = 9.0
        job_length = 'short'
        configs.append([
            num_gpus,
            task,
            model,
            pretrained,
            rank,
            cuda_capability_min,
            cuda_capability_max,
            job_length,
        ])
        
        task = 'stsb'
        num_gpus = 1
        cuda_capability_min = 8.0
        cuda_capability_max = 9.0
        job_length = 'short'
        configs.append([
            num_gpus,
            task,
            model,
            pretrained,
            rank,
            cuda_capability_min,
            cuda_capability_max,
            job_length,
        ])
        
config_df = pd.DataFrame(configs)
config_df.to_csv('real_config.csv', index=False, header=False)
        
        