#!/usr/bin/bash

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

model_path="/dev/shm/Humpback/models/m1_strict_score_matching_2400steps"
unlabelled_data_filepath="/dev/shm/Humpback/outputs/m1/unlabelled_gen_instruction.jsonl"
middle_save_filepath="/dev/shm/Humpback/outputs/m1/mid1_unlabelled_data_for_curation.jsonl"
predicted_save_filepath="/dev/shm/Humpback/outputs/m1/mid2_unlabelled_data_curation_predicted.jsonl"
curation_results_save_filepath="/dev/shm/Humpback/outputs/m1/mid3_unlabelled_data_curation_results.jsonl"
curated_save_filepath="/dev/shm/Humpback/outputs/m1/unlabelled_curated_data.jsonl"

echo "(1/3) => Build dataset for curation ..."
python -m src.core.build_curation_dataset \
    --data_filepath=${unlabelled_data_filepath} \
    --save_filepath=${middle_save_filepath} \
    --curation_prompt_filepath="data/prompts/self_curation.txt" \
    --generated_instruction_column_name="response" \
    --response_column_name="prompt"

echo "(2/3) => Predict curation results ..."
python -m src.core.predict_vllm \
    --model_path=${model_path} \
    --data_filepath=${middle_save_filepath} \
    --save_filepath=${predicted_save_filepath} \
    --prompt_column_name="prompt" \
    --tensor_parallel_size=8

echo "(3/3) => Curate results ..."
python -m src.core.filter_curation_results \
    --data_filepath=${predicted_save_filepath} \
    --middle_save_filepath=${curation_results_save_filepath} \
    --save_filepath=${curated_save_filepath} \
    --curation_response_column_name="response" \
    --score=5
