import os.path as osp
import random
import json, yaml
import copy
from transformers import AutoFeatureExtractor, WhisperModel,WhisperFeatureExtractor
import numpy as np
from scipy import signal
import soundfile as sf
import torch.distributed as dist
import os
import torch
import torchaudio
from torch.utils.data import Dataset
import whisper
from slam_llm.utils.compute_utils import calculate_output_length_1d
from datasets import load_dataset,load_from_disk
from datasets import Audio


class SpeechDatasetJsonl(torch.utils.data.Dataset):
    
    def __init__(self,
                 dataset_config,
                 tokenizer=None,
                 split='train',
                 ):
        super().__init__()
        self.mel_size = dataset_config.get("mel_size", 80) # 80 for whisper large v1 and v2, 128 for large v3

        rank = dist.get_rank()  


        data_name = "yxdu/covost2_en_x"
        local_dataset_path= data_name.split("/")[-1]+"_"+split+"_cache"

        if os.path.exists(local_dataset_path):
            ds = load_from_disk(local_dataset_path)
            print(ds)
        else:
            if rank==0:
                ds = load_dataset(data_name, split=split)
                ds = ds.cast_column("audio", Audio(sampling_rate=16000))
                print(ds)


            
                def prepare_dataset(example):            
                    audio_raw = whisper.pad_or_trim(example["audio"]["array"])

                    audio_raw = torch.tensor(audio_raw, dtype=torch.float32)  
                    audio_mel = whisper.log_mel_spectrogram(audio_raw, n_mels=self.mel_size).permute(1, 0)
                    
                    example["audio_mel"] = audio_mel
                    
                    
                    return example
                
                ds = ds.map(prepare_dataset, remove_columns="audio")

                ds.save_to_disk(local_dataset_path)
            
            dist.barrier()
            if rank != 0:
                if os.path.exists(local_dataset_path):
                    ds = load_from_disk(local_dataset_path)
                else:
                    raise FileNotFoundError("No Dataset。")

        


        self.ds = ds
        self.tokenizer = tokenizer
        self.dataset_config = dataset_config
        
        self.IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        self.prompt = dataset_config.get("prompt", None)
        self.bf16 = dataset_config.get("bf16", True)
        self.source = dataset_config.get("source", None)

        self.answer_template = "{}"
        self.fix_length_audio = dataset_config.get("fix_length_audio", -1)
        self.inference_mode = dataset_config.get("inference_mode", False)
        self.normalize = dataset_config.get("normalize", False)
        self.input_type = dataset_config.get("input_type", None)
        assert self.input_type in ["raw", "mel"], "input_type must be one of [raw, mel]" 

        self.data_list = []
        self.count = 0


        self.printed = False  

    
    def __len__(self):
        print(len(self.ds))
        return len(self.ds)
    
    def __getitem__(self, index):

        data_dict = self.ds[index]


        prompt =  "<|"+self.source+"|>"
        target = data_dict["en"]+prompt+data_dict[self.source]
        

        
        if not self.printed:  
            print(prompt)
            print(target)
            self.printed = True  

        if self.bf16:
            audio_mel = torch.tensor(data_dict["audio_mel"], dtype=torch.bfloat16)
        
        
        if self.fix_length_audio > 0:
            audio_length = self.fix_length_audio
        
        audio_pseudo = torch.full((audio_length,), -1) # placeholder

        

        if self.inference_mode:
            
            prompt_ids = self.tokenizer.encode(prompt)

            prompt_ids = torch.tensor(prompt_ids, dtype=torch.int64)
            example_ids = torch.cat((audio_pseudo, prompt_ids))  # [audio,prompt]
            example_mask = example_ids.ge(-1)  # [True,True]

            return {
                "input_ids": example_ids,
                "attention_mask": example_mask,
                "audio_length": audio_length,
                "audio_mel":audio_mel,
                "key":index,
                "target": target,
                "prompt":prompt
            }
        


        prompt_ids = self.tokenizer.encode(prompt)
        prompt_length = len(prompt_ids)


        answer = self.answer_template.format(target)
        example = prompt + answer  # FIX(MZY): avoid putting a bos token before answer.

        example_ids = self.tokenizer.encode(example)  # [prompt,answer]
        example_ids.append(self.tokenizer.eos_token_id)  # [prompt,answer,eos]
        example_ids = torch.tensor(
            example_ids, dtype=torch.int64)

        example_ids = torch.cat((audio_pseudo, example_ids))  # [audio,prompt,answer,eos]

        labels_ids = copy.deepcopy(example_ids)  # [audio,prompt,answer,eos]
        labels_ids[:audio_length + prompt_length] = -1  # [-1,-1,answer,eos];
        example_mask = example_ids.ge(-1)  # FIX(GZF): [True,True,True,True]

        label_mask = labels_ids.ge(0)  # [False,False,True,True]
        example_ids[~example_mask] = 0  # [audio,prompt,answer,eos]
        labels_ids[~label_mask] = self.IGNORE_INDEX  # [-100,-100,answer,eos]



        return {
            "input_ids": example_ids,
            "labels": labels_ids,
            "attention_mask": example_mask,
            "audio_mel": audio_mel if self.input_type == "mel" else None,
            "audio_length": audio_length,
        }

    def pad(self, sequence, max_length, padding_idx=0):
        if isinstance(sequence, (int, list, tuple)):
            if len(sequence) < max_length:
                sequence = sequence + [padding_idx] * (max_length - len(sequence))
            else:
                sequence = sequence[:max_length]
        elif isinstance(sequence, torch.Tensor):
            if len(sequence) < max_length:
                sequence = torch.cat(
                    (sequence, torch.full(([max_length - len(sequence)] + list(sequence.size())[1:]), padding_idx)))
            else:
                sequence = sequence[:max_length]
        elif isinstance(sequence, np.ndarray):
            if len(sequence) < max_length:
                sequence = np.concatenate(
                    (sequence, np.full((max_length - len(sequence),) + sequence.shape[1:], padding_idx)))
            else:
                sequence = sequence[:max_length]
        else:
            raise Exception("Type mismatch during padding!")
        return sequence
    
    def collator(self, samples):
        assert samples is not None
        input_ids_max_length = max([s['input_ids'].shape[0] for s in samples])
        input_ids = torch.stack([self.pad(s['input_ids'], input_ids_max_length, self.tokenizer.pad_token_id)
                                 for s in samples])
        attention_mask = torch.stack([self.pad(s['attention_mask'], input_ids_max_length, False)
                                      for s in samples])
        

        if self.input_type == "mel":
            audio_mel_max_length = max([s['audio_mel'].shape[0] for s in samples])
            audio_mel = torch.stack([self.pad(s['audio_mel'], audio_mel_max_length, 0)
                                  for s in samples])
            audio_mel_post_mask = torch.zeros(len(samples), (audio_mel_max_length + 1) // 2) # ad-hoc for whisper for 2x downsample from mel to feats
            for line, sample in enumerate(samples):
                audio_mel_post_mask[line, :(sample['audio_mel'].shape[0] + 1) // 2] = 1


        if self.inference_mode:
            keys = [s['key'] for s in samples]
            targets = [s['target'] for s in samples]
            prompts = [s['prompt'] for s in samples]


            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "audio_mel": audio_mel if self.input_type == "mel" else None,
                "audio_mel_post_mask": audio_mel_post_mask if self.input_type == "mel" else None,
                "keys": keys,
                "targets": targets,
                "prompts":prompts
            }

        labels = torch.stack([self.pad(s['labels'], input_ids_max_length, self.IGNORE_INDEX)
                              for s in samples])
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "audio_mel": audio_mel if self.input_type == "mel" else None,
            "audio_mel_post_mask": audio_mel_post_mask if self.input_type == "mel" else None,
        }



def get_speech_dataset(dataset_config, tokenizer, split):
    dataset = SpeechDatasetJsonl(dataset_config, tokenizer, split)

    return dataset