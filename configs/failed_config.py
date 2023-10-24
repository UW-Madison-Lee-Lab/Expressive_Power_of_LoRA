import pandas as pd

config_fnn_df = pd.read_csv('../configs/fnn_configs.csv', header = None)
config_fnn_df.columns = [
    'width', 
    'target_depth',
    'frozen_depth',
    'rank',
    'use_bias',
    'activation',
    'std',
    'method',
    'batch_size',
    'n_epochs',
    'lr',
    'n_test',
    'weight_decay',
    'init_mode',
    'exp',
    'wandb',
    'pretrained',
    'pretrained_epochs',
    'pretrained_lr',
    'pretrained_level',
    'tune_bias',
    'last_layers',
    'seed'
]
# drop the wandb column
config_fnn_df = config_fnn_df.drop(columns = ['wandb'])

run_fnn_df = pd.read_pickle('../results/results.pkl')
run_fnn_df = run_fnn_df[run_fnn_df.exp == 'fnn'][[
    'width', 
    'target_depth',
    'frozen_depth',
    'rank',
    'use_bias',
    'activation',
    'std',
    'method',
    'batch_size',
    'n_epochs',
    'lr',
    'n_test',
    'weight_decay',
    'init_mode',
    'exp',
    'pretrained',
    'pretrained_epochs',
    'pretrained_lr',
    'pretrained_level',
    'tune_bias',
    'last_layers',
    'seed'
]]

# Find additional configs
rerun_fnn_df = config_fnn_df.merge(run_fnn_df, how='left', indicator=True).loc[lambda x : x['_merge']=='left_only'].drop('_merge', axis=1).reset_index(drop=True)
# insert column wandb consisting of ones after the column weight_decay
rerun_fnn_df.insert(15, 'wandb', 1)
rerun_fnn_df.to_csv('../configs/rerun_fnn_configs.csv', index=False, header=False)
