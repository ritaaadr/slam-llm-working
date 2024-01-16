#!/bin/bash
# export PYTHONPATH=/root/whisper:$PYTHONPATH
export PYTHONPATH=/root/fairseq:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=4,5,6,7
# export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1

# debug setting for multiple gpus
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=INFO

cd /root/SLAM-LLM

speech_encoder_path=/nfs/zhifu.gzf/ckpt/Whisper/large-v2.pt
# speech_encoder_path=/nfs/maziyang.mzy/models/Whisper/large-v2-qwen.pt

llm_path=/nfs/maziyang.mzy/models/TinyLlama-1.1B-Chat-v0.4

output_dir=/nfs/maziyang.mzy/exps/TinyLlama-1.1B-Chat-v0.4-finetune-asr-ds5-proj2048-lr1e-4-finetune-whisper-large-v2-prompt-padding30-20240115

# -m debugpy --listen 5678 --wait-for-client
if [[ $CUDA_VISIBLE_DEVICES != *","* ]]; then
python -m debugpy --listen 5678 --wait-for-client src/llama_recipes/pipeline/finetune.py \
--model_name asr \
--freeze_encoder \
--freeze_llm \
--llm_name vicuna-13b-v1.5 \
--llm_path $llm_path \
--llm_dim 5120 \
--encoder_name whisper \
--encoder_ds_rate 2 \
--encoder_path $speech_encoder_path \
--encoder_dim 1280 \
--encoder_projector linear \
--encoder_projector_ds_rate 5 \
--dataset speech_dataset \
--speech_dataset.train_data_path /nfs/maziyang.mzy/data/librispeech/librispeech_train_960h.jsonl \
--speech_dataset.val_data_path /nfs/maziyang.mzy/data/librispeech/librispeech_dev_other_filtered.jsonl \
--batching_strategy custom \
--num_epochs 100 \
--batch_size_training 4 \
--val_batch_size 4 \
--num_workers_dataloader 4 \
--lr 1e-4 \
--output_dir $output_dir \
--metric acc \
# --log_file $output_dir/test.log \
# --use_wandb \
# --wandb_dir $output_dir \
# --wandb_entity_name zym22 \
# --wandb_project_name slam-llm \
# --wandb_exp_name test \
# --log_interval 5 \
# --ckpt_path "/nfs/maziyang.mzy/exps/llama-2-hf-finetune-asr-ds5-proj2048-lr1e-5-whisper-lora-prompt/asr/5/model.pt" \
# --peft_ckpt "/nfs/maziyang.mzy/exps/llama-2-hf-finetune-asr-ds5-proj2048-lr1e-5-whisper-lora-prompt/asr/5" \
# --use_peft --peft_method lora \

else
torchrun \
--nnodes 1 \
--nproc_per_node 4 \
--master_port=29501 \
src/llama_recipes/pipeline/finetune.py \
--model_name asr \
--freeze_llm \
--use_fp16 \
--enable_fsdp \
--llm_name tinyllama-1.1b-chat-v0.4 \
--llm_path $llm_path \
--llm_dim 2048 \
--encoder_name whisper \
--encoder_ds_rate 2 \
--encoder_path $speech_encoder_path \
--encoder_dim 1280 \
--encoder_projector linear \
--encoder_projector_ds_rate 5 \
--dataset speech_dataset \
--speech_dataset.train_data_path /nfs/maziyang.mzy/data/librispeech/librispeech_train_960h.jsonl \
--speech_dataset.val_data_path /nfs/maziyang.mzy/data/librispeech/librispeech_dev_other_filtered.jsonl \
--batching_strategy custom \
--num_epochs 100 \
--batch_size_training 4 \
--val_batch_size 4 \
--num_workers_dataloader 4 \
--lr 1e-4 \
--output_dir $output_dir \
--metric acc \
--log_file /$output_dir/train.log \
--use_wandb \
--wandb_dir $output_dir \
--wandb_entity_name zym22 \
--wandb_project_name slam-llm \
--wandb_exp_name test \
--log_interval 5 \
# --peft_ckpt "/nfs/maziyang.mzy/exps/llama-2-hf-finetune-asr-ds5-proj2048-lr1e-5-whisper-prompt-padding30-20231228/asr/4" \
# --ckpt_path "/nfs/maziyang.mzy/exps/llama-2-hf-finetune-asr-ds5-proj2048-lr1e-5-whisper-prompt-padding30-20231228/asr/4/model.pt" \
# --use_peft --peft_method lora \
# --freeze_encoder \
fi