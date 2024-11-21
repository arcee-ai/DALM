#!/usr/bin/bash

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

model_path="/dev/shm/Humpback/outputs/backward_model_on_seed_data_scheduled"
data_filepath="data/unlabelled/falcon-refinedweb-sampled.jsonl"
save_filepath="/dev/shm/Humpback/outputs/m1/unlabelled_gen_instruction.jsonl"
prompt_column_name="content"

# python -m src.core.predict_vllm \
#     --reverse \ 
#     --model_path=${model_path} \
#     --data_filepath=${data_filepath} \
#     --save_filepath=${save_filepath} \
#     --prompt_column_name=${prompt_column_name} \
#     --tensor_parallel_size=8

python src/core/predict_vllm.py \
    --reverse \
    --model_path=${model_path} \
    --data_filepath=${data_filepath} \
    --save_filepath=${save_filepath} \
    --prompt_column_name=${prompt_column_name} \
    --tensor_parallel_size=8
