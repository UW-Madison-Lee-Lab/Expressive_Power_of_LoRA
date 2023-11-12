export num_gpus=2
export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"  
path=$(pwd)
root_dir="${path%/*/*/*}"
output_dir=$root_dir/results/real_exp/NLU/mrpc

python -m torch.distributed.launch --nproc_per_node=$num_gpus \
examples/text-classification/run_glue.py \
--model_name_or_path roberta-base \
--lora_path ./roberta_base_lora_mnli.bin \
--task_name mrpc \
--do_train \
--do_eval \
--max_seq_length 512 \
--per_device_train_batch_size 16 \
--learning_rate 4e-4 \
--num_train_epochs 1 \
--output_dir $output_dir/model \
--overwrite_output_dir \
--logging_steps 10 \
--logging_dir $output_dir/log \
--evaluation_strategy epoch \
--save_strategy epoch \
--warmup_ratio 0.06 \
--apply_lora \
--lora_r 8 \
--lora_alpha 16 \
--seed 0 \
--weight_decay 0.1

# python -m torch.distributed.launch --nproc_per_node=$num_gpus \
# examples/text-classification/run_glue.py \
# --model_name_or_path roberta-base \
# --lora_path ./roberta_base_mnli_lora.bin \
# --task_name mrpc \
# --do_train \
# --do_eval \
# --max_seq_length 512 \
# --per_device_train_batch_size 16 \
# --learning_rate 4e-4 \
# --num_train_epochs 30 \
# --output_dir $output_dir/model \
# --overwrite_output_dir \
# --logging_steps 10 \
# --logging_dir $output_dir/log \
# --evaluation_strategy epoch \
# --save_strategy epoch \
# --warmup_ratio 0.06 \
# --apply_lora \
# --lora_r 8 \
# --lora_alpha 16 \
# --seed 0 \
# --weight_decay 0.1
