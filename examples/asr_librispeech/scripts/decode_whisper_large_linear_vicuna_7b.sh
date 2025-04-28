#!/bin/bash
#export PYTHONPATH=/workspace/whisper:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1

export RANK=0
export WORLD_SIZE=2
export LOCAL_RANK=0


run_dir=/workspace/SLAM-LLM
cd $run_dir
code_dir=examples/asr_librispeech

speech_encoder_path=/root/.cache/whisper/base.pt
llm_path=/workspace/SLAM-LLM/src/slam_llm/models/vicuna-7b

#output_dir=/workspace/SLAM-LLM/tmp/vicuna-7b-librispeech-linear-steplrwarmupkeep1e-4-whisper-largev3-20250130
#ckpt_path=$output_dir/asr_epoch_1_step_1000
ckpt_path=/root/.cache/whisper
split=debug-wav
val_data_path=/workspace/SLAM-LLM/LibriSpeech/${split}.jsonl
decode_log=$ckpt_path/decode_${split}_beam4

# -m debugpy --listen 5678 --wait-for-client
#torchrun --nproc_per_node=2 $code_dir/inference_asr_batch.py \
python $code_dir/inference_asr_batch.py \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        hydra.run.dir=$ckpt_path \
        ++model_config.llm_name="vicuna-7b" \
        ++model_config.llm_path=$llm_path \
        ++model_config.llm_dim=2048 \
        ++model_config.encoder_name=whisper \
        ++model_config.encoder_projector_ds_rate=5 \
        ++model_config.encoder_path=$speech_encoder_path \
        ++model_config.encoder_path_hf=null \
        ++model_config.encoder_dim=1280 \
        ++model_config.encoder_projector=linear \
        ++dataset_config.dataset=speech_dataset \
        ++dataset_config.val_data_path=$val_data_path \
        ++dataset_config.input_type=mel \
        ++dataset_config.mel_size=128 \
        ++dataset_config.inference_mode=true \
        ++train_config.model_name=asr \
        ++train_config.freeze_encoder=true \
        ++train_config.freeze_llm=true \
        ++train_config.batching_strategy=custom \
        ++train_config.num_epochs=1 \
        ++train_config.val_batch_size=1 \
        ++train_config.gradient_accumulation_steps=1 \
        ++train_config.mixed_precision=false \
        ++train_config.enable_fsdp=true \
        ++train_config.low_cpu_fsdp=true \
        ++train_config.enable_deepspeed=true \
        ++train_config.num_workers_dataloader=2 \
        ++train_config.output_dir=$output_dir \
        ++decode_log=$decode_log \
        ++ckpt_path=$ckpt_path/base.pt \
        ++dataset_config.fix_length_audio=30 \
        # ++peft_ckpt=$ckpt_path \
        # ++train_config.use_peft=true \
        # ++train_config.peft_config.r=32 \
        # ++dataset_config.normalize=true \
        # ++model_config.encoder_projector=q-former \



