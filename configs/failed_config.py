import pandas as pd

# newly added parameters in the experiments, name: default value 
new_fnn_column = {'rank_step': 0}

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
    'seed',
    *new_fnn_column.keys(),
]
# drop the wandb column
config_fnn_df = config_fnn_df.drop(columns = ['wandb', *new_fnn_column.keys()])

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
    'seed',
]]

# Find additional configs
rerun_fnn_df = config_fnn_df.merge(run_fnn_df, how='left', indicator=True).loc[lambda x : x['_merge']=='left_only'].drop('_merge', axis=1).reset_index(drop=True)
# insert column wandb consisting of ones after the column weight_decay
rerun_fnn_df.insert(15, 'wandb', 1)
for key, value in new_fnn_column.items():
    rerun_fnn_df[key] = value
rerun_fnn_df.to_csv('../configs/rerun_fnn_configs.csv', index=False, header=False)

config_tfn_df = pd.read_csv('../configs/tfn_configs.csv', header = None)
config_tfn_df.columns = [
    'embed_dim',
    'n_head', 
    'depth',
    'rank',
    'batch_size',
    'seq_len',
    'method',
    'n_epochs',
    'lr',
    'weight_decay',
    'wandb',
    'std',
    'n_test',
    'exp',
    'pretrained',
    'pretrained_epochs',
    'pretrained_lr',
    'pretrained_level',
    'seed',
]
# drop the wandb column
config_tfn_df = config_tfn_df.drop(columns = ['wandb'])

run_tfn_df = pd.read_pickle('../results/results.pkl')
run_tfn_df = run_tfn_df[run_tfn_df.exp == 'tfn'][[
    'width',
    'n_head', 
    'target_depth',
    'rank',
    'batch_size',
    'seq_length',
    'method',
    'n_epochs',
    'lr',
    'weight_decay',
    'std',
    'n_test',
    'exp',
    'pretrained',
    'pretrained_epochs',
    'pretrained_lr',
    'pretrained_level',
    'seed',
]]

run_tfn_df.columns = [
    'embed_dim',
    'n_head', 
    'depth',
    'rank',
    'batch_size',
    'seq_len',
    'method',
    'n_epochs',
    'lr',
    'weight_decay',
    'std',
    'n_test',
    'exp',
    'pretrained',
    'pretrained_epochs',
    'pretrained_lr',
    'pretrained_level',
    'seed',
]

# Find additional configs
rerun_tfn_df = config_tfn_df.merge(run_tfn_df, how='left', indicator=True).loc[lambda x : x['_merge']=='left_only'].drop('_merge', axis=1).reset_index(drop=True)
# insert column wandb consisting of ones after the column weight_decay
rerun_tfn_df.insert(11, 'wandb', 1)
rerun_tfn_df.to_csv('../configs/rerun_tfn_configs.csv', index=False, header=False)
