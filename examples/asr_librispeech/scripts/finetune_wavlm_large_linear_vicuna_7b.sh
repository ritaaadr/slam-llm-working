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

speech_encoder_path=/workspace/SLAM-LLM/checkpoints/WavLM-Large.pt
llm_path=/workspace/SLAM-LLM/src/slam_llm/models/vicuna-7b
#train_data_path=/workspace/SLAM-LLM/LibriSpeech/jsonl/debug-wav-fixed.jsonl
#val_data_path=/workspace/SLAM-LLM/LibriSpeech/jsonl/debug-wav-fixed.jsonl
train_data_path=/workspace/SLAM-LLM/dataset/jsonl/training.jsonl
val_data_path=/workspace/SLAM-LLM/dataset/jsonl/validation.jsonl

output_dir=/workspace/SLAM-LLM/tmp/vicuna-7b-v1.5-librispeech-linear-steplrwarmupkeep1e-4-wavlm-large-$(date +"%Y%m%d")

hydra_args="
hydra.run.dir=$output_dir \
++model_config.llm_name=vicuna-7b-v1.5 \
++model_config.llm_path=$llm_path \
++model_config.llm_dim=4096 \
++model_config.encoder_name=wavlm \
++model_config.normalize=true \
++dataset_config.normalize=true \
++model_config.encoder_projector_ds_rate=5 \
++model_config.encoder_path=$speech_encoder_path \
++model_config.encoder_dim=1024 \
++model_config.encoder_projector=linear \
++model_config.load_in_8bit=true \
++dataset_config.dataset=speech_dataset \
++dataset_config.train_data_path=$train_data_path \
++dataset_config.val_data_path=$val_data_path \
++dataset_config.input_type=raw \
++train_config.model_name=asr \
++train_config.num_epochs=3 \
++train_config.freeze_encoder=true \
++train_config.freeze_llm=true \
++train_config.batching_strategy=custom \
++train_config.context_length=2048 \
++train_config.warmup_steps=15 \
++train_config.total_steps=600 \
++train_config.lr=1e-2 \
++train_config.use_gradient_checkpointing=true \
++train_config.validation_interval=400 \
++train_config.batch_size_training=3 \
++train_config.val_batch_size=1 \
++train_config.num_workers_dataloader=2 \
++train_config.low_cpu_fsdp=false \
++train_config.quantization=true \
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
    CUDA_VISIBLE_DEVICES=0 \
    python $code_dir/finetune_asr.py \
    --config-path "conf" \
    --config-name "prompt.yaml" \
    ++train_config.enable_fsdp=false \
    ++train_config.enable_ddp=false \
    ++train_config.use_fp16=true \
    ++train_config.one_gpu=true \
    $hydra_args

fi
