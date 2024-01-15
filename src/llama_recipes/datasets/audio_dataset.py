import os.path as osp
import random
import json, yaml
import copy

import numpy as np
from scipy import signal
import soundfile as sf

import torch
import torchaudio
from torch.utils.data import Dataset
from llama_recipes.utils.compute_utils import calculate_output_length_1d
from llama_recipes.models.BEATs.BEATs import BEATs


class AudioDatasetJsonl(torch.utils.data.Dataset):
    
    def __init__(self,
                 dataset_config,
                 tokenizer=None,
                 split='train',
                 ):
        super().__init__()
        self.dataset_config = dataset_config
        self.tokenizer = tokenizer
        # data_parallel_size = dist.get_world_size()
        data_parallel_size = 1
        
        # self.data_list = contents
        self.IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        self.prompt_template = "USER: {}\n ASSISTANT:"
        self.answer_template = "{}"
        self.fix_length_audio = dataset_config.fix_length_audio

        self.data_list = []
        if split == "train":
            with open(dataset_config.train_data_path, encoding='utf-8') as fin:
                for line in fin:
                    data_dict = json.loads(line.strip())
                    self.data_list.append(data_dict)
        else:
            with open(dataset_config.val_data_path, encoding='utf-8') as fin:
                for line in fin:
                    data_dict = json.loads(line.strip())
                    self.data_list.append(data_dict)

        # # debug
        # with open(dataset_config.train_data_path, encoding='utf-8') as fin:
        #         for line in fin:
        #             data_dict = json.loads(line.strip())
        #             self.data_list.append(data_dict)
        # if split == "train":
        #     self.data_list = self.data_list[:80]
        # else:
        #     self.data_list = self.data_list[80:100]

    def get_source_len(self, data_dict):
        return data_dict["source_len"]

    def get_target_len(self, data_dict):
    
        return data_dict["target_len"] if "target_len" in data_dict else 0
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        data_dict = self.data_list[index]
        audio_path = data_dict.get("source")
        target = data_dict.get("target", None)
        task = data_dict.get("prompt", "AAC")
        
        audio_raw, sample_rate = torchaudio.load(audio_path)
        assert sample_rate == 16e3, "Sample rate should be 16kHz, but got {}in file {}".format(sr, source_file)
        audio_mel = BEATs.preprocess(audio_raw[0], fbank_mean=self.dataset_config.fbank_mean, fbank_std=self.dataset_config.fbank_std)

        prompt = "Describe the audio you hear. Output the audio caption directly without redundant content. Ensure that the output is not duplicated. "

        prompt = self.prompt_template.format(prompt)
        answer = self.answer_template.format(target)

        prompt_ids = self.tokenizer.encode(prompt)

        prompt_length = len(prompt_ids)
        audio_length = (audio_mel.shape[0] + 1) // 2  # ad-hoc for beats for 2x downsample from mel to feats
        audio_length = audio_length // 5 # ad-hoc for 5x fc downsample
        # audio_length = calculate_output_length_1d(audio_length, 5, 5, 0) # ad-hoc for 5x cov1d downsample
        if self.fix_length_audio > 0:
            audio_length = self.fix_length_audio
        audio_pseudo = torch.full((audio_length,), -1) # placeholder

        example = prompt + answer  # FIX(MZY): avoid putting a bos token before answer.
        example_ids = self.tokenizer.encode(example)  # [prompt,answer]
        example_ids.append(self.tokenizer.eos_token_id)  # [prompt,answer,eos]
        example_ids = torch.tensor(
            example_ids, dtype=torch.int64
        )
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
            'audio_mel': audio_mel,
            'audio_length': audio_length,
    
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
        else:
            raise Exception("Type mismatch during padding!")
        return sequence

    def collator(self, samples):
        assert samples is not None
        input_ids_max_length = max([s['input_ids'].shape[0] for s in samples])
        input_ids = torch.stack([self.pad(s['input_ids'], input_ids_max_length, self.tokenizer.pad_token_id)
                                 for s in samples])
        labels = torch.stack([self.pad(s['labels'], input_ids_max_length, self.IGNORE_INDEX)
                              for s in samples])
        attention_mask = torch.stack([self.pad(s['attention_mask'], input_ids_max_length, False)
                                      for s in samples])
    
        audio_mel_max_length = max([s['audio_mel'].shape[0] for s in samples])
        audio_mel = torch.stack([self.pad(s['audio_mel'], audio_mel_max_length, 0)
                                  for s in samples])
        audio_mel_mask = torch.zeros(len(samples), audio_mel_max_length)
        for line, sample in enumerate(samples):
            audio_mel_mask[line, :sample['audio_mel'].shape[0]] = 1
        modality_mask = torch.zeros_like(attention_mask)
        for line, sample in enumerate(samples):
            modality_mask[line, :sample['audio_length']] = 1
    
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'audio_mel': audio_mel,
            'audio_mel_mask': audio_mel_mask,
            'modality_mask': modality_mask
        }



def get_audio_dataset(dataset_config, tokenizer, split):
    dataset = AudioDatasetJsonl(dataset_config, tokenizer, split)

    return dataset
