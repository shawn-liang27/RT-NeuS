#!/bin/bash
# PARAM $1: log directory name


MODEL=internvl2
MODEL_LABEL=InternVL2_5-8B
MODEL_ARGS=OpenGVLab/InternVL2_5-8B
#-------------7-10B--------------
# internvl2 InternVL2_5-8B OpenGVLab/InternVL2_5-8B
# qwen2_5_vl Qwen2.5-VL-7B-Instruct Qwen/Qwen2.5-VL-7B-Instruct
# llava_onevision llava-onevision-qwen2-7b-ov lmms-lab/llava-onevision-qwen2-7b-ov
# llava_vid LLaVA-Video-7B-Qwen2 lmms-lab/LLaVA-Video-7B-Qwen2

# For qwen2_5_vl run uv pip install transformers --upgrade
# For InternVL2_5, run: uv pip install transformers==4.45.0
# For llava-onevision-qwen2-7b-ov, run: uv pip install transformers==4.45.0

# internvideo2_5 InternVideo2_Chat_8B_InternLM2_5 "OpenGVLab/InternVideo2_Chat_8B_InternLM2_5"
# videollama3 VideoLLaMA3-7B DAMO-NLP-SG/VideoLLaMA3-7B
# phi4_multimodal Phi-4-multimodal-instruct "microsoft/Phi-4-multimodal-instruct"
# gemma3 Gemma-3-4B-It "google/gemma-3-4b-it"

source .env/bin/activate
# TYPE: naive rt-neus
TYPE=$1
TRIAL=$2 # experiment trial number
SET=$3 # neusqa or original
LOG_BASE_PATH="/usr/homes/sgl57/NeuS/NeuS-QA/experiment_results/lmms_eval/mlvu/${TYPE}/experiment_${TRIAL}"

JOB_ID=$(date +%Y%m%d_%H%M)
LOG_DIR="${LOG_BASE_PATH}/${MODEL_LABEL}/${SET}"
mkdir -p $LOG_DIR

DATA_DIR="/mnt/data0/sgl57/data/mlvu/MLVU"
DATA_FILE="${DATA_DIR}/${TYPE}/experiment_${TRIAL}.json"

FILE=/usr/homes/sgl57/NeuS/NeuS-QA/lmms-eval/lmms_eval/tasks/mlvu/mlvu_dev.yaml
ORIGINAL_FILE=/mnt/data0/sgl57/data/mlvu/MLVU/original_dataset_lmms.json

if [ "$SET" = "neusqa" ]; then
    TARGET_PATH="$DATA_FILE"
else
    TARGET_PATH="$ORIGINAL_FILE"
fi

# Execute the replacement using the chosen path
sed -i "s|\(test: \).*|\1\"$TARGET_PATH\"|" "$FILE"

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
# Export all relevant GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export DECORD_EOF_RETRY_MAX=20480

# Qwen2.5 VL
python3 -m accelerate.commands.launch \
    --num_processes=6 \
    --mixed_precision="bf16" \
    -m lmms_eval \
    --model $MODEL \
    --model_args "pretrained=${MODEL_ARGS},attn_implementation=flash_attention_2" \
    --batch_size 1 \
    --verbosity=DEBUG \
    --tasks mlvu_dev \
    --log_samples \
    --output_path $LOG_DIR > "${LOG_DIR}/eval.out" 2>&1

# export TRANSFORMERS_NO_META=1
# # InternVL2
# python3 -m lmms_eval \
#     --model "$MODEL" \
#     --model_args "${MODEL_ARGS},attn_implementation=flash_attention_2" \
#     --batch_size 1 \
#     --verbosity=DEBUG \
#     --tasks longvideobench_val_v \
#     --log_samples \
#     --output_path "$LOG_DIR" > "${LOG_DIR}/eval.out" 2>&1