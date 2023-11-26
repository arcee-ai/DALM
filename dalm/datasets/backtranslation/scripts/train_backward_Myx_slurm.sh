#!/usr/bin/bash

#SBATCH --job-name=backward
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log

#SBATCH --partition=Partition
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH -x SH-IDCA1404-10-140-54-116

#SBATCH --nodes=1
#SBATCH --gres=gpu:4


source ~/anaconda3/bin/activate torch

num_nodes=1
num_gpu_per_node=4

bsz=32
output_dir="outputs/backward_model_on_seed_data_scheduled_ds1"

mkdir -p $output_dir
bsz_per_dev=$(echo "${bsz} / ${num_nodes} / ${num_gpu_per_node}" | bc)

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIS ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo "Node: $head_node"
echo "Node IP: $head_node_ip"

srun torchrun \
    --nnodes ${num_nodes} \
    --nproc_per_node ${num_gpu_per_node} \
    --node_rank $SLURM_NODEID \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint $head_node:29518 \
    -m src.core.train_flash_attn \
        --reverse \
        --deepspeed conf/ds_zero2default.json \
        --model_name_or_path /home/zhutong/Llama-2-7b-hf \
        --data_path data/seed/seed.jsonl \
        --per_device_train_batch_size ${bsz_per_dev} \
        --per_device_eval_batch_size ${bsz_per_dev} \
        --num_train_epochs 15 \
        --adam_beta1 0.9 \
        --adam_beta2 0.95 \
        --learning_rate "1e-5" \
        --final_lr "9e-6" \
        --weight_decay 0.1 \
        --max_grad_norm 1.0 \
        --evaluation_strategy "no" \
        --logging_strategy steps \
        --logging_steps 1 \
        --save_strategy epoch \
        --save_total_limit 1 \
        --output_dir ${output_dir} \
        --overwrite_output_dir \
        --ddp_timeout 30000 \
        --logging_first_step True \
        --bf16 True \
        --tf32 True \
        --ddp_find_unused_parameters False \
        --gradient_checkpointing \
        --report_to none \
        --log_level info \
        --lazy_preprocess True

        # --fsdp "full_shard auto_wrap" \
        # --fsdp_config conf/fsdp_config.json \
