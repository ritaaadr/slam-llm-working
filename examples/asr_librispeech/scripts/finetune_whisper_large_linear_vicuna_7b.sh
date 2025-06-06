#!/bin/bash
# export PYTHONPATH=/workspace/whisper:$PYTHONPATH
export PYTHONPATH=/workspace/fairseq:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1

# debug setting for multiple gpus
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=INFO

run_dir=/workspace/SLAM-LLM
cd $run_dir
code_dir=examples/asr_librispeech

speech_encoder_path=~/.cache/whisper/large-v3.pt
llm_path=/workspace/SLAM-LLM/src/slam_llm/models/vicuna-7b
#llm_path=/workspace/SLAM-LLM/src/slam_llm/models/Llama-3.2-1B/original
#train_data_path=/workspace/SLAM-LLM/LibriSpeech/librispeech_dev.jsonl
train_data_path=/workspace/SLAM-LLM/LibriSpeech/debug-wav-fixed.jsonl
val_data_path=/workspace/SLAM-LLM/LibriSpeech/debug-wav-fixed.jsonl

output_dir=/workspace/SLAM-LLM/tmp/vicuna-7b-librispeech-linear-steplrwarmupkeep1e-4-whisper-largev3-$(date +"%Y%m%d")

hydra_args="
hydra.run.dir=$output_dir \
++model_config.llm_name=vicuna-7b \
++model_config.llm_path=$llm_path \
++model_config.llm_dim=4096 \
++model_config.encoder_name=whisper \
++model_config.encoder_projector_ds_rate=5 \
++model_config.encoder_path=$speech_encoder_path \
++model_config.encoder_path_hf=null \
++model_config.encoder_dim=1280 \
++model_config.encoder_projector=linear \
++dataset_config.dataset=speech_dataset \
++dataset_config.train_data_path=$train_data_path \
++dataset_config.val_data_path=$val_data_path \
++dataset_config.input_type=mel \
++dataset_config.mel_size=128 \
++train_config.model_name=asr \
++train_config.num_epochs=3 \
++train_config.freeze_encoder=true \
++train_config.freeze_llm=true \
++train_config.batching_strategy=custom \
++train_config.warmup_steps=1000 \
++train_config.total_steps=100000 \
++train_config.lr=1e-4 \
++train_config.validation_interval=5 \
++train_config.batch_size_training=1 \
++train_config.val_batch_size=1 \
++train_config.gradient_accumulation_steps=4 \
++train_config.num_workers_dataloader=2 \
++train_config.output_dir=$output_dir \
++metric=acc \
"

# -m debugpy --listen 5678 --wait-for-client
if [[ $CUDA_VISIBLE_DEVICES != *","* ]]; then
    python -m debugpy --listen 5678 --wait-for-client $code_dir/finetune_asr.py \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        $hydra_args
else
    torchrun \
        --nnodes 1 \
        --nproc_per_node 2 \
        --master_port=29503 \
        $code_dir/finetune_asr.py \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        ++train_config.enable_fsdp=true \
        ++train_config.enable_ddp=false \
        ++train_config.use_fp16=false \
        $hydra_args
fi
