import os
import torch
import torchaudio
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from slam_llm.models.encoder import WavLMEncoder

import json
from torch.utils.data import Dataset, DataLoader

AUDIO_DIR = "/workspace/SLAM-LLM/LibriSpeech/debug-wav"
MODEL_PATH = "/workspace/SLAM-LLM/src/slam_llm/models/vicuna-7b"
JSONL_FILE = "/workspace/SLAM-LLM/LibriSpeech/debug-wav-fixed.jsonl"

class AudioTranscriptionDataset(Dataset):
    def __init__(self, jsonl_file, tokenizer, audio_dir):
        self.audio_data = []
        with open(jsonl_file, "r") as f:
            for line in f:
                entry = json.loads(line)
                self.audio_data.append(entry)
        self.tokenizer = tokenizer
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.audio_data)

    def __getitem__(self, idx):
        entry = self.audio_data[idx]
        audio_path = os.path.join(self.audio_dir, entry['source'])
        transcription = entry['target']
        
        waveform, sr = torchaudio.load(audio_path)
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)
        
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)  # convert stereo to mono
        
        # DO NOT UNSQUEEZE HERE
        # waveform = waveform.unsqueeze(0)  # REMOVE THIS LINE
        
        # Tokenize the transcription (target text)
        target = self.tokenizer(transcription, padding="max_length", truncation=True, return_tensors="pt")

        return waveform, target.input_ids.squeeze(0)



def load_vicuna_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config, device_map="auto")
    return tokenizer, model

def load_encoder():
    encoder_cfg = type("Dummy", (), {
        "encoder_path": "/workspace/SLAM-LLM/checkpoints/WavLM-Large.pt",
        "normalize": True,
        "encoder_type": "finetune"
    })()
    return WavLMEncoder.load(encoder_cfg).cuda().eval()

def process_audio(file_path):
    waveform, sr = torchaudio.load(file_path)
    waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # Convert to mono

    # Ensure the tensor is 3D: (batch_size, num_channels, seq_length)
    
    print(f"waveform shape: {waveform.shape}")  # Debugging shape
    
    return waveform.to(torch.float32).to("cuda")



def fine_tune_vicuna():
    # Load the model and encoder
    tokenizer, vicuna = load_vicuna_model(MODEL_PATH)
    encoder = load_encoder()

    # Create the dataset and dataloader
    dataset = AudioTranscriptionDataset(JSONL_FILE, tokenizer, AUDIO_DIR)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Projection layer to match the size of encoder outputs to the prompt size
    projector = torch.nn.Linear(1024, 4096).to("cuda")

    vicuna.gradient_checkpointing_enable()

    # Fine-tuning loop
    vicuna.train()  # Set the model in training mode
    optimizer = torch.optim.AdamW(vicuna.parameters(), lr=5e-6)
    for waveform, labels in dataloader:
        waveform = waveform.squeeze(1).to("cuda")  # Remove the unnecessary channel dimension if needed

        # Encode the waveform and apply projection
        padding_mask = torch.zeros(waveform.shape[0], waveform.shape[1], dtype=torch.bool).to(waveform.device)
        encoder_outs = encoder.extract_features(waveform, padding_mask)
        encoder_outs = projector(encoder_outs)
        encoder_outs = encoder_outs.to(torch.float16)

        # Tokenize the prompt and concatenate with encoder outputs
        prompt = "USER: Transcribe the following audio.\nASSISTANT:"
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        prompt_embeds = vicuna.model.embed_tokens(prompt_ids)
        inputs_embeds = torch.cat([prompt_embeds, encoder_outs], dim=1)

        attention_mask = torch.ones(inputs_embeds.shape[:-1], dtype=torch.long).to(inputs_embeds.device)

        # Forward pass through Vicuna and compute loss

        # Compute the full input length (prompt + audio embeddings)
        input_len = inputs_embeds.shape[1]

        # Shift labels to align with generation part
        labels_shifted = torch.full((labels.shape[0], input_len), -100, dtype=labels.dtype, device=labels.device)
        gen_start = prompt_embeds.shape[1]  # index where generation starts
        gen_end = min(input_len, gen_start + labels.shape[1])  # just in case it's truncated

        labels_shifted[:, gen_start:gen_end] = labels[:, :gen_end - gen_start]

        # Now compute loss
        outputs = vicuna(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels_shifted)
        loss = outputs.loss


        # Backward pass and optimization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vicuna.parameters(), max_norm=1.0)
        
        if torch.isnan(loss) or torch.isinf(loss):
            print("[WARNING] Skipping step due to invalid loss.")
            optimizer.zero_grad()
            continue

        optimizer.step()
        optimizer.zero_grad()

        # Clear cache
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        print(f"[INFO] Loss: {loss.item()}")

@torch.no_grad()
def run_all_inference(audio_dir, model_path):
    tokenizer, vicuna = load_vicuna_model(model_path)
    encoder = load_encoder()

    # Projection layer to match the size of encoder outputs to prompt size
    projector = torch.nn.Linear(1024, 4096).to("cuda")

    for fname in os.listdir(audio_dir):
        if not fname.endswith(".wav"):
            continue
        fpath = os.path.join(audio_dir, fname)
        print(f"\n[INFO] Processing file: {fname}")
        
        waveform = process_audio(fpath)
        padding_mask = torch.zeros(waveform.shape[0], waveform.shape[1], dtype=torch.bool).to(waveform.device)
        encoder_outs = encoder.extract_features(waveform, padding_mask)

        # Debugging step: print encoder outputs shape and content
        print(f"[DEBUG] encoder_outs shape: {encoder_outs.shape}")
        print(f"[DEBUG] First few encoder_outs values:\n{encoder_outs[0, :5]}")

        # Check if we need to project the encoder output to match the prompt size
        if encoder_outs.shape[2] != 4096:
            print(f"[WARNING] Projecting encoder output to match the prompt embedding size.")
            encoder_outs = projector(encoder_outs)

        # Ensure encoder_outs are in float16 for compatibility with the model
        encoder_outs = encoder_outs.to(torch.float16)

        print("[INFO] Tokenizing prompt...")
        prompt = "USER: Transcribe the following audio.\nASSISTANT:"
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        prompt_embeds = vicuna.model.embed_tokens(prompt_ids)

        print("[INFO] Concatenating prompt and audio embeddings...")
        inputs_embeds = torch.cat([prompt_embeds, encoder_outs], dim=1)
        attention_mask = torch.ones(inputs_embeds.shape[:-1], dtype=torch.long).to(inputs_embeds.device)

        print("[INFO] Generating output...")
        outputs = vicuna.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=200,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            repetition_penalty=1.0,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
        
        # Debugging step: print raw model output before decoding
        print(f"[DEBUG] Raw model output tokens: {outputs[0]}")
        
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"[RESULT] {fname} ➜ {decoded.strip()}")

if __name__ == "__main__":
    #run_all_inference(AUDIO_DIR, MODEL_PATH)
     fine_tune_vicuna()
