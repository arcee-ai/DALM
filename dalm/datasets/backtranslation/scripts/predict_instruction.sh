# TODO: to add inference executive file with multiple GPUs
#!/usr/bin/bash

export CUDA_VISIBLE_DEVICES=7

num_nodes=1
num_gpu_per_node=1

bsz=4
model_path="/dev/shm/tzhu/outputs/forward_model_on_seed_data_scheduled"

bsz_per_dev=$(echo "${bsz} / ${num_nodes} / ${num_gpu_per_node}" | bc)

torchrun \
    --nnodes ${num_nodes} \
    --nproc_per_node ${num_gpu_per_node} \
    -m src.core.predict \
        --mixed_precision="bf16" \
        --model_path=${model_path} \
        --data_filepath="data/seed/seed.jsonl" \
        --save_filepath="outputs/seed_pred.jsonl" \
        --prompt_column_name="instruction" \
        --batch_size=${bsz_per_dev}
