#!/bin/bash
# export PYTHONPATH=/root/whisper:$PYTHONPATH
export PYTHONPATH=/root/fairseq:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1

# debug setting for multiple gpus
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=INFO

cd /root/SLAM-LLM

# speech_encoder_path= TODO!


llm_path=/nfs/maziyang.mzy/models/vicuna-7b-v1.5
# llm_path=/nfs/maziyang.mzy/models/vicuna-13b-v1.5

output_dir=/nfs/yangguanrou.ygr/vicuna-13b-v1.5-finetune-avsr-20230115

# -m debugpy --listen 5678 --wait-for-client
if [[ $CUDA_VISIBLE_DEVICES != *","* ]]; then
python src/llama_recipes/pipeline/finetune.py \
--model_name avsr \
--freeze_encoder \
--freeze_llm \
--llm_name vicuna-13b-v1.5 \
--llm_path $llm_path \
--llm_dim 4096 \
--encoder_name moco_wav2vec2 \
--encoder_ds_rate 2 \
--encoder_dim 512 \
--encoder_projector linear \
--encoder_projector_ds_rate 5 \
--dataset avsr_dataset \
--avsr_dataset.file src/llama_recipes/datasets/avsr_dataset.py:get_audio_dataset \
--batching_strategy custom \
--num_epochs 20 \
--batch_size_training 6 \
--val_batch_size 2 \
--num_workers_dataloader 2 \
--lr 1e-4 \
--output_dir $output_dir \
--metric acc \
--log_file "/root/SLAM-LLM/log/second_try.log" \
--use_wandb \
--wandb_dir $output_dir \
--wandb_entity_name yanghaha \
--wandb_project_name slam-llm \
--wandb_exp_name avsr \
--log_interval 5 \

else
torchrun \
--nnodes 1 \
--nproc_per_node 4 \
src/llama_recipes/pipeline/finetune.py \
--model_name avsr \
--freeze_encoder \
--freeze_llm \
--use_fp16 \
--enable_fsdp \
--llm_name vicuna-13b-v1.5 \
--llm_path $llm_path \
--llm_dim 4096 \
--encoder_name moco_wav2vec2 \
--encoder_ds_rate 2 \
--encoder_dim 512 \
--encoder_projector linear \
--encoder_projector_ds_rate 5 \
--dataset avsr_dataset \
--avsr_dataset.file src/llama_recipes/datasets/avsr_dataset.py:get_audio_dataset \
--batching_strategy custom \
--num_epochs 20 \
--batch_size_training 2 \
--val_batch_size 2 \
--num_workers_dataloader 2 \
--lr 1e-4 \
--output_dir $output_dir \
--metric acc \
--log_file "/root/SLAM-LLM/log/second_try.log" \
--use_wandb \
--wandb_dir $output_dir \
--wandb_entity_name yanghaha \
--wandb_project_name slam-llm \
--wandb_exp_name avsr \
--log_interval 5 \
# --peft_ckpt "/nfs/maziyang.mzy/exps/llama-2-hf-finetune-asr-ds5-proj2048-lr1e-5-whisper-prompt-padding30-20231228/asr/4" \
# --ckpt_path "/nfs/maziyang.mzy/exps/llama-2-hf-finetune-asr-ds5-proj2048-lr1e-5-whisper-prompt-padding30-20231228/asr/4/model.pt" \
# --use_peft --peft_method lora \
# --master_port=29501 \
fi

# {"key": "1001-134707-0000_ASR", "prompt": "<ASR>", "source": "/cpfs01/shared/Group-speech/beinian.lzr/data/open_data/librispeech_audio/audio/se_librispeech_1001-134707-0000.wav", "target": "1 little recks the laborer. How near his work is holding him to God, The loving laborer through space and time, after all, not to create, only or found only.", "target_len": 157, "source_len": 1581, "text-type": "Transcribe", "audio_language": "en", "text_language": "en", "task-type": "<ASR>"}
# {"key": "1688-142285-0005", "prompt": "<ASR>", "source": "/nfs/beinian.lzr/workspace/datasets/data/16k/opendata/librispeech/test_other/wav/1688-142285-0005.wav", "target": "YOU WHO WERE ALWAYS ACCUSING PEOPLE OF BEING SHOPPY AT HELSTONE", "target_len": 11, "source_len": 220, "text-type": "Transcribe", "audio_language": "en", "text_language": "en", "task-type": "<ASR>"}



# 没用 encoder_ds_rate

# 1.15

# 7b batch size 开到2 ok的

#  6 2 0 可以