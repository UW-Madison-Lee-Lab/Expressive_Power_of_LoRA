import pandas as pd
import numpy as np

def get_rows(
    df,
    new_columns,
    args_dict,
):
    idx = {}
    for key, value in args_dict.items():
        if key in new_columns and value == new_columns[key]:
            idx[key] = (df[key] == value) | (df[key].isna())
        else:
            idx[key] = df[key] == value
        
    run_idx = np.ones_like(df['width'], dtype=bool)
    for key, value in idx.items():
        run_idx = run_idx & value
    
    # print(f"Selected {run_idx.sum()} runs!")
    
    run_df = df[run_idx].reset_index(drop=True)   
    return run_df

# newly added parameters in the experiments, name: default value 
new_fnn_column = {}

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
    'rank_step',
    *new_fnn_column.keys(),
]
# drop the wandb column
config_fnn_df = get_rows(config_fnn_df, new_fnn_column, {**new_fnn_column, 'exp': 'fnn'})
config_fnn_df = config_fnn_df.drop(columns = ['wandb', *new_fnn_column.keys()])

run_fnn_df = pd.read_pickle('../results/results.pkl')
run_fnn_df = get_rows(run_fnn_df, new_fnn_column, {**new_fnn_column, 'exp': 'fnn'})
run_fnn_df = run_fnn_df[[
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
    'rank_step',
]]

run_fnn_df.to_csv('../results/results_fnn.csv', index=False)

# Find additional configs
# rerun_fnn_df = config_fnn_df.merge(run_fnn_df, how='left', indicator=True).loc[lambda x : x['_merge']=='left_only'].drop('_merge', axis=1).reset_index(drop=True)
rerun_fnn_df = pd.concat([config_fnn_df, run_fnn_df, run_fnn_df]).drop_duplicates(keep=False)

rerun_fnn_df['wandb'] = 1

rerun_fnn_df = rerun_fnn_df[[
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
    'rank_step',
]]


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
rerun_tfn_df['wandb'] = 1
rerun_tfn_df = rerun_tfn_df[[
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
]]

rerun_tfn_df.to_csv('../configs/rerun_tfn_configs.csv', index=False, header=False)
