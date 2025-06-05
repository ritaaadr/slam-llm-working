# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import time
import yaml
from contextlib import nullcontext
from pathlib import Path
from pkg_resources import packaging
from torch.distributed.algorithms.join import Join

import torch
import torch.cuda.nccl as nccl
import torch.distributed as dist
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from tqdm import tqdm
from transformers import LlamaTokenizer


from slam_llm.utils.checkpoint_handler import (
    save_model_checkpoint, 
    save_model_and_optimizer_sharded, 
    save_optimizer_checkpoint, 
    save_model_checkpoint_peft,
    save_model_checkpoint_peft_full_shard
)
from slam_llm.policies import fpSixteen,bfSixteen_mixed, get_llama_wrapper
from slam_llm.utils.memory_utils import MemoryTrace
from slam_llm.utils.metric import compute_accuracy
from functools import partial  # (if not already imported)

import wandb
import logging
logger = logging.getLogger(__name__)

import time 
from jiwer import wer


def set_tokenizer_params(tokenizer: LlamaTokenizer):
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token #TODO aggiunto questo

# Converting Bytes to Megabytes
def byte2mb(x):
    return int(x / 2**20)

def train(model, train_dataloader,eval_dataloader, tokenizer, optimizer, lr_scheduler, gradient_accumulation_steps, train_config, log_config, fsdp_config=None, local_rank=None, rank=None):
    """
    Trains the model on the given dataloader

    Args:
        model: The model to be trained
        train_dataloader: The dataloader containing the training data
        optimizer: The optimizer used for training
        lr_scheduler: The learning rate scheduler
        gradient_accumulation_steps: The number of steps to accumulate gradients before performing a backward/update operation
        num_epochs: The number of epochs to train for
        local_rank: The rank of the current node in a distributed setting
        train_config: The training configuration
        log_config: The logging configuration
        eval_dataloader: The dataloader containing the eval data
        tokenizer: tokenizer used in the eval for decoding the predicitons

    Returns: results dictionary containing average training and validation perplexity and loss
    """
    # Create a gradient scaler for fp16
    # if train_config.use_fp16 and train_config.enable_fsdp:
    #     scaler = ShardedGradScaler()
    # elif train_config.use_fp16 and not train_config.enable_fsdp:
    #     scaler = torch.cuda.amp.GradScaler()
    if train_config.use_fp16:
        scaler = torch.cuda.amp.GradScaler()
        if train_config.enable_fsdp:
            scaler = ShardedGradScaler()
    if train_config.enable_fsdp or train_config.enable_ddp:
        world_size = int(os.environ["WORLD_SIZE"])
    # Updated autocast logic to allow dtype override for quantization
    if train_config.quantization:
        autocast = partial(torch.cuda.amp.autocast, dtype=torch.float16)
    elif train_config.use_fp16:
        autocast = partial(torch.cuda.amp.autocast, dtype=torch.float16)
    elif train_config.use_bf16:
        autocast = partial(torch.cuda.amp.autocast, dtype=torch.bfloat16)
    else:
        autocast = nullcontext

    
    train_prep = []
    train_loss = []
    train_acc = []
    val_prep = []
    val_loss =[]
    val_acc = []
    val_wer = []
    epoch_times = []
    checkpoint_times = []
    results = {}
    best_val_loss = float("inf")
    best_val_acc = 0.0
    #=================Early stopping=====================================
    patience_counter = 0
    min_delta = getattr(train_config, "early_stopping_min_delta", 0.0)
    patience = getattr(train_config, "early_stopping_patience", 3)
    #=====================================================================

    # === Training encoder projector ======================================================
    #print("\n[INFO] Trainable parameters:")
    #for name, param in model.named_parameters():
    #    print(f"{'[TRAINABLE]' if param.requires_grad else '[FROZEN]':<12} {name}")
    #num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #print(f"[INFO] Total trainable parameters: {num_trainable:,}\n")
    # =====================================================================================

    for epoch in range(train_config.num_epochs):
        epoch_start_time = time.perf_counter()
        #HO TOLTO QUESTO
        #with MemoryTrace() as memtrace, Join([model, optimizer]):
        #HO MESSO:
        if train_config.enable_fsdp or train_config.enable_ddp:
            join_context = Join([model, optimizer])
        else:
            join_context = nullcontext()
        with MemoryTrace() as memtrace, join_context: # track the memory usage
        #FINE MIA MODIFICA
            model.train()
            total_loss = 0.0
            total_acc = 0.0
            if train_config.batching_strategy != "dynamic":
                total_length = len(train_dataloader)//gradient_accumulation_steps
                pbar = tqdm(colour="blue", desc=f"Training Epoch: {epoch+1}", total=total_length, dynamic_ncols=True)
            else:
                pbar = tqdm(colour="blue", desc=f"Training Epoch: {epoch+1}", dynamic_ncols=True)
            for step, batch in enumerate(train_dataloader):
                for key in batch.keys():
                    if train_config.enable_fsdp or train_config.enable_ddp:
                        batch[key] = batch[key].to(local_rank) if isinstance(batch[key], torch.Tensor) else batch[key]
                        if isinstance(batch[key], dict):
                            for k2 in batch[key].keys():
                                batch[key][k2] = batch[key][k2].to(local_rank) if isinstance(batch[key][k2], torch.Tensor) else batch[key][k2]
                    else:
                        batch[key] = batch[key].to('cuda:0') if isinstance(batch[key], torch.Tensor) else batch[key]
                        if isinstance(batch[key], dict):
                            for k2 in batch[key].keys():
                                batch[key][k2] = batch[key][k2].to('cuda:0') if isinstance(batch[key][k2], torch.Tensor) else batch[key][k2]
                #=============================================================================================
                if "audio" in batch:
                    if rank is None or rank == 0:
                        print(f"[DEBUG] Checking audio shapes at step {step}")
                    if isinstance(batch["audio"], (list, tuple)):
                        for i, waveform in enumerate(batch["audio"]):
                            if waveform.numel() == 0:
                                print(f"[ERROR] Batch sample {i} has zero-length waveform!")
                    elif isinstance(batch["audio"], torch.Tensor):
                        for i in range(batch["audio"].shape[0]):
                            if batch["audio"][i].numel() == 0:
                                print(f"[ERROR] Batch sample {i} has zero-length waveform!")
                        else:
                            if rank is None or rank == 0:
                                print(f"[DEBUG] All {batch['audio'].shape[0]} samples have non-zero length.")
                #=============================================================================================
                with autocast():
                    outputs, *rest = model(**batch)
                    # =============================================================
                    # DEBUG: check if labels and decoding make sense
                    if epoch == 0 and step == 0:
                        labels = batch.get("labels", None)
                        input_ids = batch.get("input_ids", None)
                        if labels is not None and input_ids is not None:
                            for i in range(len(labels)):
                                decoded_labels = tokenizer.decode([t for t in labels[i] if t >= 0], skip_special_tokens=True)
                                decoded_inputs = tokenizer.decode([t for t in input_ids[i] if t >= 0], skip_special_tokens=True)
                                print(f"[DEBUG] Sample {i} - Final labels (target):", decoded_labels)
                                print(f"[DEBUG] Sample {i} - Final inputs (source):", decoded_inputs)


                    # =============================================================

                acc = rest[0] if rest else -1
                loss = outputs.loss

                loss = loss / gradient_accumulation_steps
                acc = acc / gradient_accumulation_steps

                if log_config.use_wandb and step % log_config.log_interval == 0:
                    if train_config.enable_fsdp or train_config.enable_ddp:
                        if rank==0:
                            wandb.log({"train_inner/train_inner_loss":loss, "train_inner/train_inner_accuracy":acc}, step=(epoch * total_length + step) if train_config.batching_strategy != "dynamic" else step + 1)
                    else:
                        wandb.log({"train_inner/train_inner_loss":loss, "train_inner/train_inner_accuracy":acc}, step=(epoch * total_length + step) if train_config.batching_strategy != "dynamic" else step + 1)
                total_loss += loss.detach().float()
                total_acc += acc
                if train_config.use_fp16:
                    # if fp16 is enabled, use gradient scaler to handle gradient update
                    scaler.scale(loss).backward()

                    if (step + 1) % gradient_accumulation_steps == 0 or (train_config.batching_strategy != "dynamic" and step == len(train_dataloader) - 1):
                        scaler.step(optimizer)
                        scaler.update()
                        if lr_scheduler is not None:
                            lr_scheduler.step()
                            current_lr = lr_scheduler.get_last_lr()[0]
                        else:
                            current_lr = optimizer.param_groups[0]["lr"]
                        if current_lr == 0:
                            break
                        if log_config.use_wandb and step % log_config.log_interval == 0:
                            if train_config.enable_fsdp or train_config.enable_ddp:
                                if rank==0:
                                    wandb.log({"train_inner/lr":current_lr}, step=(epoch * total_length + step) if train_config.batching_strategy != "dynamic" else step + 1)
                            else:
                                wandb.log({"train_inner/lr":current_lr}, step=(epoch * total_length + step) if train_config.batching_strategy != "dynamic" else step + 1)
                        optimizer.zero_grad()
                        pbar.update(1)
                else:
                    # regular backpropagation when fp16 is not used
                    loss.backward()
                    if (step + 1) % gradient_accumulation_steps == 0 or (  train_config.batching_strategy != "dynamic" and step == len(train_dataloader) - 1):
                        optimizer.step()
                        if lr_scheduler is not None:
                            lr_scheduler.step()
                            current_lr = lr_scheduler.get_last_lr()[0]
                        else:
                            current_lr = optimizer.param_groups[0]["lr"]
                        if current_lr == 0:
                            break
                        if log_config.use_wandb and step % log_config.log_interval == 0:
                            if train_config.enable_fsdp or train_config.enable_ddp:
                                if rank==0:
                                    wandb.log({"train_inner/lr":current_lr}, step=(epoch * total_length + step) if train_config.batching_strategy != "dynamic" else step + 1)
                            else:
                                wandb.log({"train_inner/lr":current_lr}, step=(epoch * total_length + step) if train_config.batching_strategy != "dynamic" else step + 1)
                        optimizer.zero_grad()
                        pbar.update(1)

                pbar.set_description(f"Training Epoch: {epoch+1}/{train_config.num_epochs}, step {step}/{len(train_dataloader) if train_config.batching_strategy != 'dynamic' else ''} completed (loss: {loss.detach().float()}, acc: {acc})")
                
                if (epoch * total_length + step + 1 if train_config.batching_strategy != "dynamic" else step + 1) % train_config.validation_interval == 0 and train_config.run_validation:
                    eval_ppl, eval_epoch_loss, eval_epoch_acc, eval_wer = evaluation(model, train_config, eval_dataloader, local_rank, tokenizer, rank=rank)
                    #eval_epoch_acc = rest[0] if rest else -1
                    checkpoint_start_time = time.perf_counter()
                    val_wer.append(eval_wer)
                    if train_config.save_model and (eval_epoch_loss < best_val_loss):
                        checkpoint_name = f"{train_config.model_name}_epoch_{str(epoch+1)}_step_{step+1}"
                        if train_config.enable_fsdp or train_config.enable_ddp:
                            dist.barrier()
                        if train_config.use_peft:
                            if train_config.enable_fsdp or train_config.enable_ddp:
                                if rank==0:
                                    logger.info(f"we are about to save the PEFT modules")
                            else:
                                logger.info(f"we are about to save the PEFT modules")
                            if train_config.enable_fsdp:
                                if fsdp_config.sharding_strategy == ShardingStrategy.FULL_SHARD:
                                    save_model_checkpoint_peft_full_shard(
                                            model, optimizer, rank, train_config, epoch=epoch
                                        )
                                elif fsdp_config.sharding_strategy == ShardingStrategy.NO_SHARD:
                                    if rank==0:
                                        save_model_checkpoint_peft(
                                            model, optimizer, rank, train_config, checkpoint_name=checkpoint_name
                                        )
                                    dist.barrier()
                            elif train_config.enable_ddp:
                                if rank==0:
                                    save_model_checkpoint_peft(
                                            model, optimizer, rank, train_config, checkpoint_name=checkpoint_name
                                        )
                                dist.barrier()
                            else:
                                # model.save_pretrained(train_config.output_dir)
                                save_model_checkpoint_peft(
                                        model, optimizer, rank, train_config, checkpoint_name=checkpoint_name
                                    )
                            if train_config.enable_fsdp or train_config.enable_ddp:
                                if rank==0:
                                    logger.info(f"PEFT modules are saved in {train_config.output_dir} directory")
                            else:
                                logger.info(f"PEFT modules are saved in {train_config.output_dir} directory")
                        
                        elif not train_config.use_peft and train_config.freeze_llm:
                            logger.info(f"llm is frozen, we are about to save other parts.")
                            if train_config.enable_fsdp:
                                if fsdp_config.sharding_strategy == ShardingStrategy.FULL_SHARD:
                                    save_model_checkpoint_peft_full_shard(
                                            model, optimizer, rank, train_config, epoch=epoch
                                        )
                                elif fsdp_config.sharding_strategy == ShardingStrategy.NO_SHARD:
                                    if rank==0:
                                        save_model_checkpoint_peft(
                                            model, optimizer, rank, train_config, checkpoint_name=checkpoint_name
                                        )
                                    dist.barrier()
                            elif train_config.enable_ddp:
                                if rank==0:
                                    save_model_checkpoint_peft(
                                            model, optimizer, rank, train_config, checkpoint_name=checkpoint_name
                                        )
                                dist.barrier()
                            else:
                                save_model_checkpoint_peft(
                                        model, optimizer, rank, train_config, checkpoint_name=checkpoint_name
                                    )

                        else:
                            if train_config.enable_fsdp:
                                if getattr(StateDictType, fsdp_config.checkpoint_type) == StateDictType.FULL_STATE_DICT:
                                    save_model_checkpoint(
                                        model, optimizer, rank, train_config, epoch=epoch
                                    )
                                elif getattr(StateDictType, fsdp_config.checkpoint_type) == StateDictType.SHARDED_STATE_DICT:
                                    logger.info(" Saving the FSDP model checkpoints using SHARDED_STATE_DICT")
                                    logger.info("=====================================================")

                                    save_model_and_optimizer_sharded(model, rank, train_config)
                                    if train_config.save_optimizer:
                                        save_model_and_optimizer_sharded(model, rank, train_config, optim=optimizer)
                                        logger.info(" Saving the FSDP model checkpoints and optimizer using SHARDED_STATE_DICT")
                                        logger.info("=====================================================")

                                if train_config.save_optimizer:
                                    save_optimizer_checkpoint(
                                        model, optimizer, rank, train_config, epoch=epoch
                                    )
                                    logger.info(" Saving the FSDP model checkpoints and optimizer using FULL_STATE_DICT")
                                    logger.info("=====================================================")

                            elif train_config.enable_ddp:
                                if rank==0:
                                    save_model_checkpoint_peft(
                                            model, optimizer, rank, train_config, checkpoint_name=checkpoint_name
                                        )
                                dist.barrier()
                                    
                            else:
                                save_model_checkpoint_peft(
                                        model, optimizer, rank, train_config, checkpoint_name=checkpoint_name
                                    )
                                
                        if train_config.enable_fsdp or train_config.enable_ddp:
                            dist.barrier()
                    checkpoint_end_time = time.perf_counter() - checkpoint_start_time
                    checkpoint_times.append(checkpoint_end_time)
                    if eval_epoch_loss < best_val_loss:
                        best_val_loss = eval_epoch_loss
                        if train_config.enable_fsdp or train_config.enable_ddp:
                            if rank==0:
                                logger.info(f"best eval loss on epoch {epoch+1} is {best_val_loss}")
                        else:
                            logger.info(f"best eval loss on epoch {epoch+1} is {best_val_loss}")
                    val_loss.append(eval_epoch_loss)
                    val_prep.append(eval_ppl)
                    if rest:
                        if eval_epoch_acc > best_val_acc:
                            best_val_acc = eval_epoch_acc
                            if train_config.enable_fsdp or train_config.enable_ddp:
                                if rank==0:
                                    logger.info(f"best eval acc on epoch {epoch+1} is {best_val_acc}")
                            else:
                                logger.info(f"best eval acc on epoch {epoch+1} is {best_val_acc}")
                        val_acc.append(rest[0]) 
                    else: 
                        val_acc.append(-1)
                    
                    if log_config.use_wandb:
                        if train_config.enable_fsdp or train_config.enable_ddp:
                            if rank==0:
                                wandb.log({"valid/val_epoch_loss":eval_epoch_loss, "valid/val_perplexity":eval_ppl, "valid/best_val_loss":best_val_loss, "valid/val_accuracy":val_acc[-1], "valid/val_best_accuracy":best_val_acc})
                        else:
                            wandb.log({"valid/val_epoch_loss":eval_epoch_loss, "valid/val_perplexity":eval_ppl, "valid/best_val_loss":best_val_loss, "valid/val_accuracy":val_acc[-1], "valid/val_best_accuracy":best_val_acc})

                if train_config.run_test_during_validation:
                    if train_config.enable_fsdp or train_config.enable_ddp:
                        if rank==0:
                            logger.info("=====================================")
                            logger.info(f"Test the file {train_config.run_test_during_validation_file} during validation:")
                            with autocast():
                                logger.info(model.inference(train_config.run_test_during_validation_file, train_config.run_test_during_validation_prompt))
                            logger.info("=====================================")
                        dist.barrier()
                    else:
                        logger.info("=====================================")
                        logger.info(f"Test the file {train_config.run_test_during_validation_file} during validation:")
                        with autocast():
                            logger.info(model.inference(train_config.run_test_during_validation_file, train_config.run_test_during_validation_prompt))
                        logger.info("=====================================")
            pbar.close()

        epoch_end_time = time.perf_counter()-epoch_start_time
        epoch_times.append(epoch_end_time)
        # Reducing total_loss across all devices if there's more than one CUDA device
        if torch.cuda.device_count() > 1 and (train_config.enable_fsdp or train_config.enable_ddp):
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_acc, op=dist.ReduceOp.SUM)
        train_epoch_loss = total_loss / (len(train_dataloader)  if train_config.batching_strategy != "dynamic" else (step + 1) *train_config.num_epochs)
        train_epoch_acc = total_acc / (len(train_dataloader) if train_config.batching_strategy != "dynamic" else (step + 1) *train_config.num_epochs)
        if train_config.enable_fsdp or train_config.enable_ddp:
            train_epoch_loss = train_epoch_loss/world_size
            train_epoch_acc = train_epoch_acc/world_size
        train_perplexity = torch.exp(train_epoch_loss)

        train_prep.append(train_perplexity)
        train_loss.append(train_epoch_loss)
        train_acc.append(train_epoch_acc)

        if log_config.use_wandb:
            if train_config.enable_fsdp or train_config.enable_ddp:
                if rank==0:
                    wandb.log({"train/train_perplexity":train_perplexity, "train/train_epoch_loss":train_epoch_loss, "train/train_epoch_acc":train_epoch_acc})
            else:
                wandb.log({"train/train_perplexity":train_perplexity, "train/train_epoch_loss":train_epoch_loss, "train/train_epoch_acc":train_epoch_acc})

        if train_config.enable_fsdp or train_config.enable_ddp:
            if rank==0:
                logger.info(f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")
        else:
            logger.info(f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")
        if train_config.early_stopping:
            if len(val_loss) > 0:
                current_val_loss = val_loss[-1]
                if current_val_loss + min_delta < best_val_loss:
                    patience_counter = 0
                    best_val_loss = current_val_loss
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"Early stopping triggered. No improvement in validation loss for {patience} validations.")
                        break


        if train_config.enable_fsdp:
            if rank==0:
                logger.info(f"Max CUDA memory allocated was {memtrace.peak} GB")
                logger.info(f"Max CUDA memory reserved was {memtrace.max_reserved} GB")
                logger.info(f"Peak active CUDA memory was {memtrace.peak_active_gb} GB")
                logger.info(f"Cuda Malloc retires : {memtrace.cuda_malloc_retires}")
                logger.info(f"CPU Total Peak Memory consumed during the train (max): {memtrace.cpu_peaked + memtrace.cpu_begin} GB")
        else:
            logger.info(f"Max CUDA memory allocated was {memtrace.peak} GB")
            logger.info(f"Max CUDA memory reserved was {memtrace.max_reserved} GB")
            logger.info(f"Peak active CUDA memory was {memtrace.peak_active_gb} GB")
            logger.info(f"Cuda Malloc retires : {memtrace.cuda_malloc_retires}")
            logger.info(f"CPU Total Peak Memory consumed during the train (max): {memtrace.cpu_peaked + memtrace.cpu_begin} GB")

        # Update the learning rate as needed
        # lr_scheduler.step()

    epoch_duration = time.perf_counter() - epoch_start_time
    logger.info(f"[TIME] Epoch {epoch+1} took {epoch_duration:.2f} seconds.")
    avg_epoch_time = sum(epoch_times)/ len(epoch_times)
    avg_checkpoint_time = sum(checkpoint_times)/ len(checkpoint_times) if len(checkpoint_times) > 0 else 0
    avg_train_prep = sum(train_prep)/len(train_prep)
    avg_train_loss = sum(train_loss)/len(train_loss)
    avg_train_acc = sum(train_acc)/len(train_acc)
    if train_config.run_validation:
        avg_eval_prep = sum(val_prep)/len(val_prep)
        avg_eval_loss = sum(val_loss)/len(val_loss)
        avg_eval_acc = sum(val_acc)/len(val_acc)

    results['avg_train_prep'] = avg_train_prep
    results['avg_train_loss'] = avg_train_loss
    results['avg_train_acc'] = avg_train_acc
    if train_config.run_validation:
        results['avg_eval_prep'] = avg_eval_prep
        results['avg_eval_loss'] = avg_eval_loss
        results['avg_eval_acc'] = avg_eval_acc
    results["avg_epoch_time"] = avg_epoch_time
    results["avg_checkpoint_time"] = avg_checkpoint_time
    
    val_wer = [x for x in val_wer if x is not None]
    if len(val_wer) > 0:
        avg_eval_wer = sum(val_wer) / len(val_wer)
    else:
        avg_eval_wer = None  # or 0.0 or skip logging, as appropriate

    results['avg_eval_wer'] = avg_eval_wer

    

    #saving the training params including fsdp setting for reference.
    # if (train_config.enable_fsdp or train_config.enable_ddp)and not train_config.use_peft:
    #     save_train_params(train_config, fsdp_config, rank)

    return results

def evaluation(model,train_config, eval_dataloader, local_rank, tokenizer, rank=None):
    """
    Evaluates the model on the given dataloader

    Args:
        model: The model to evaluate
        eval_dataloader: The dataloader containing the evaluation data
        local_rank: The rank of the current node in a distributed setting
        tokenizer: The tokenizer used to decode predictions

    Returns: eval_ppl, eval_epoch_loss
    """
    
    all_decoded = []
    if train_config.enable_fsdp or train_config.enable_ddp:
        world_size = int(os.environ["WORLD_SIZE"])
    model.eval()
    eval_preds = []
    eval_loss = 0.0  # Initialize evaluation loss
    eval_acc = 0.0
    #autocast = torch.cuda.amp.autocast if train_config.use_fp16 else nullcontext # (Fix:MZY): fix expected scalar type mismatch in norm 
    ground_truths = []


    if train_config.quantization:
        autocast = partial(torch.cuda.amp.autocast, dtype=torch.float16)
    elif train_config.use_fp16:
        autocast = partial(torch.cuda.amp.autocast, dtype=torch.float16)
    elif getattr(train_config, "use_bf16", False):  # in case bf16 is optionally configured
        autocast = partial(torch.cuda.amp.autocast, dtype=torch.bfloat16)
    else:
        autocast = nullcontext

    
    with MemoryTrace() as memtrace:
        if train_config.batching_strategy != "dynamic":
            total_length = len(eval_dataloader)
            pbar = tqdm(colour="green", desc=f"Evaluating Epoch", total=total_length, dynamic_ncols=True)
        else:
            pbar = tqdm(colour="green", desc=f"Evaluating Epoch",  dynamic_ncols=True)
        for step, batch in enumerate(eval_dataloader):
            for key in batch.keys():
                if train_config.enable_fsdp or train_config.enable_ddp:
                    batch[key] = batch[key].to(local_rank) if isinstance(batch[key], torch.Tensor) else batch[key]
                else:
                    batch[key] = batch[key].to('cuda:0') if isinstance(batch[key], torch.Tensor) else batch[key]
            # Ensure no gradients are computed for this scope to save memory
            with torch.no_grad():
                # Forward pass and compute loss
                with autocast(): # (Fix:MZY): fix expected scalar type mismatch in norm 
                    outputs, *rest = model(**batch)
                acc = rest[0] if rest else -1
                loss = outputs.loss

                eval_loss += loss.detach().float()
                eval_acc += acc
                
            #Decode predictions and add to evaluation predictions list
            try:
                preds = torch.argmax(outputs.logits, -1)
                eval_preds.extend(
                    tokenizer.batch_decode(preds.detach().cpu().numpy(), skip_special_tokens=True)
                )
            except Exception:
                pass  # vallex does not need to show it's result (we can't view any thing from abstract acoustic token)
            #=================================WER===================================================
            # Compute ground-truth references for WER
            if "labels" in batch:
                labels = batch["labels"]
                for i in range(len(preds)):
                    # Skip padding and special tokens (<0)
                    decoded_ref = tokenizer.decode([t for t in labels[i] if t >= 0], skip_special_tokens=True)
                    decoded_hyp = tokenizer.decode([t for t in preds[i] if t >= 0], skip_special_tokens=True)
                    ground_truths.append(decoded_ref)
                    all_decoded.append(decoded_hyp)
            #=========================================================================================
            
            #====================================================
            # Generation-based evaluation
            #if step == 0:
            #    print(f"[DEBUG] Running generation on eval set...")

            #try:
            #    generated_ids = model.generate(**batch)
            #    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            #    all_decoded.extend(decoded)
            #except Exception as e:
            #    print(f"[WARNING] Failed during generation: {e}")

            #=====================================================================
            #print
            # Decode predictions (greedy)
            try:
                preds = torch.argmax(outputs.logits, -1)
                decoded = tokenizer.batch_decode(preds.detach().cpu().numpy(), skip_special_tokens=True)
                all_decoded.extend(decoded)
            except Exception as e:
                print(f"[WARNING] Failed decoding logits: {e}")
            #====================================================================
            pbar.update(1)
            pbar.set_description(f"step: {step+1}/{total_length if train_config.batching_strategy != 'dynamic' else '' }, eval_loss: {eval_loss/(step+1):.4f}, eval_acc: {eval_acc/(step+1):.4f}")

        #========================================WER====================================
        if len(ground_truths) == len(all_decoded):
            wer_score = wer(ground_truths, all_decoded)
            logger.info(f"[WER] Word Error Rate: {wer_score:.4f}")
        else:
            wer_score = None
            logger.warning("[WER] Mismatch between number of predictions and references.")
        #=================================================

        # After the eval loop finishes
        print("\n======== FINAL GENERATED TRANSCRIPTIONS ========")
        for i, text in enumerate(all_decoded[:10]):  # or remove [:10] to show all
            print(f"[Transcript {i}] {text}")
        print("=================================================\n")
    # If there's more than one CUDA device, reduce evaluation loss across all devices
    if torch.cuda.device_count() > 1 and train_config.enable_fsdp or train_config.enable_ddp:
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(eval_acc, op=dist.ReduceOp.SUM)

    # Compute average loss and perplexity
    eval_epoch_loss = eval_loss / (len(eval_dataloader) if train_config.batching_strategy != "dynamic" else step + 1)
    eval_epoch_acc = eval_acc / (len(eval_dataloader) if train_config.batching_strategy != "dynamic" else step + 1)
    if train_config.enable_fsdp or train_config.enable_ddp:
        eval_epoch_loss = eval_epoch_loss/world_size
        eval_epoch_acc = eval_epoch_acc/world_size
    eval_ppl = torch.exp(eval_epoch_loss)

    # Print evaluation metrics
    if train_config.enable_fsdp or train_config.enable_ddp:
        if local_rank==0:
            logger.info(f" {eval_ppl=} {eval_epoch_loss=} {eval_epoch_acc=}")
    else:
        logger.info(f" {eval_ppl=} {eval_epoch_loss=} {eval_epoch_acc=}")

    return eval_ppl, eval_epoch_loss, eval_epoch_acc, wer_score

def freeze_transformer_layers(model, num_layer):
   for i, layer in enumerate(model.model.layers):
            if i < num_layer:
                for param in layer.parameters():
                    param.requires_grad = False


def check_frozen_layers_peft_model(model):
     for i, layer in enumerate(model.base_model.model.model.layers):
            for name, param in layer.named_parameters():
                logger.info(f"Layer {i}, parameter {name}: requires_grad = {param.requires_grad}")


def setup():
    """Initialize the process group for distributed training"""
    dist.init_process_group("nccl")


def setup_environ_flags(rank):
    """Set environment flags for debugging purposes"""
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    # This flag will help with CUDA memory fragmentations that can lead into OOM in some cases.
    # Note this is only availble in PyTorch Nighlies (as of July 30 2023)
    # os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
    if rank == 0:
        logger.info(f"--> Running with torch dist debug set to detail")


def cleanup():
    """Clean up the process group after training"""
    dist.destroy_process_group()


def clear_gpu_cache(rank=None):
    """Clear the GPU cache for all ranks"""
    if rank == 0:
        logger.info(f"Clearing GPU cache for all ranks")
    torch.cuda.empty_cache()


def get_parameter_dtypes(model):
    """Get the data types of model parameters"""
    parameter_dtypes = {}
    for name, parameter in model.named_parameters():
        parameter_dtypes[name] = parameter.dtype
    return parameter_dtypes

def print_model_size(model, config, rank: int = 0) -> None:
    """
    log model name, the number of trainable parameters and initialization time.

    Args:
        model: The PyTorch model.
        model_name (str): Name of the model.
        init_time_start (float): Initialization start time.
        init_time_end (float): Initialization end time.
        rank (int, optional): Current process's rank. Defaults to 0.
    """
    if rank == 0:
        logger.info(f"--> Model {config.model_name}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"--> {config.model_name} has {total_params / 1e6} Million params\n")

def print_module_size(module, module_name, rank: int = 0) -> None:
    """
    Print module name, the number of trainable parameters and initialization time.

    Args:
        module: The PyTorch module.
        module_name (str): Name of the model.
        rank (int, optional): Current process's rank. Defaults to 0.
    """
    if rank == 0:
        logger.info(f"--> Module {module_name}")
        total_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        logger.info(f"--> {module_name} has {total_params / 1e6} Million params\n")


def get_policies(cfg, rank):
    """Get the policies for mixed precision and fsdp wrapping"""

    verify_bfloat_support = (
    torch.version.cuda
    and torch.cuda.is_bf16_supported()
    and packaging.version.parse(torch.version.cuda).release >= (11, 0)
    and dist.is_nccl_available()
    and nccl.version() >= (2, 10)
    )


    mixed_precision_policy = None
    wrapping_policy = None

    # Mixed precision
    if cfg.mixed_precision:
        bf16_ready = verify_bfloat_support

        if bf16_ready and not cfg.use_fp16:
            mixed_precision_policy = bfSixteen_mixed
            if rank == 0:
                logger.info(f"bFloat16 enabled for mixed precision - using bfSixteen policy")
        elif cfg.use_fp16:
            mixed_precision_policy = fpSixteen
            if rank == 0:
                logger.info(f"FP16 enabled")
        else:
            logger.info(f"bFloat16 support not present. Using FP32, and not mixed precision")
    wrapping_policy = get_llama_wrapper()
    return mixed_precision_policy, wrapping_policy

def save_train_params(train_config, fsdp_config, rank):
    """
    This function saves the train_config and FSDP config into a train_params.yaml.
    This will be used by converter script in the inference folder to fetch the HF model name or path.
    It also would be hepful as a log for future references.
    """
    # Convert the train_config and fsdp_config objects to dictionaries,
    # converting all values to strings to ensure they can be serialized into a YAML file
    train_config_dict = {k: str(v) for k, v in vars(train_config).items() if not k.startswith('__')}
    fsdp_config_dict = {k: str(v) for k, v in vars(fsdp_config).items() if not k.startswith('__')}
    # Merge the two dictionaries into one
    train_params_dict = {**train_config_dict, **fsdp_config_dict}
    # Construct the folder name (follwoing FSDP checkpointing style) using properties of the train_config object
    folder_name = (
    train_config.dist_checkpoint_root_folder
    + "/"
    + train_config.dist_checkpoint_folder
    + "-"
    + train_config.model_name
    )

    save_dir = Path.cwd() / folder_name
    # If the directory does not exist, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Convert the dictionary to a YAML string
    config_yaml = yaml.dump(train_params_dict, indent=4)
    file_name = os.path.join(save_dir,'train_params.yaml')

    # Check if there's a directory with the same name as the file
    if os.path.isdir(file_name):
        logger.info(f"Error: {file_name} is a directory, not a file.")
    else:
        # Write the YAML string to the file
        with open(file_name, 'w') as f:
            f.write(config_yaml)
        if rank==0:
            logger.info(f"training params are saved in {file_name}")
