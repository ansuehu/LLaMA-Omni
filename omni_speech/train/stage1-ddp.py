import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from omni_speech.model.builder import create_model
import argparse
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchaudio import load
from omni_speech.conversation import conv_templates
import math
import json
from omni_speech.datasets.preprocess import tokenizer_speech_token
from transformers import TrainingArguments, Trainer, TrainerCallback
from torch.nn.utils.rnn import pad_sequence
import torch.distributed as dist
import gc

def collate_fn(batch):
    """Optimized collate function with memory management"""
    input_ids, labels, speech_tensors, speech_lengths = zip(
        *[list(item.values()) for item in batch]
    )
    
    # Pad sequences
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=128009)
    labels = pad_sequence(labels, batch_first=True, padding_value=128009)

    # Handle speech tensors more efficiently
    max_length = max(t.shape[-1] for t in speech_tensors)
    batch_size = len(speech_tensors)
    
    # Pre-allocate tensor to avoid fragmentation
    padded_speech = torch.zeros(batch_size, max_length, dtype=torch.float32)
    
    for i, tensor in enumerate(speech_tensors):
        length = tensor.shape[-1] if tensor.dim() > 1 else tensor.shape[0]
        if tensor.dim() == 2 and tensor.shape[0] == 1:
            padded_speech[i, :length] = tensor.squeeze(0)
        else:
            padded_speech[i, :length] = tensor
    
    # Convert to bfloat16 only when needed
    speech_lengths = torch.tensor(speech_lengths, dtype=torch.long)
    
    return {
        "input_ids": input_ids,
        "labels": labels, 
        "speech": padded_speech.to(torch.bfloat16),
        "speech_lengths": speech_lengths
    }

class CustomDataset(Dataset):
    def __init__(self, questions, tokenizer, conv_mode, model_config, input_type, mel_size, device):
        self.questions = questions
        self.tokenizer = tokenizer
        self.conv_mode = conv_mode
        self.model_config = model_config
        self.input_type = input_type
        self.mel_size = mel_size
        self.device = device

    def __getitem__(self, index):
        item = self.questions[index]
        speech_file = item["speech"]
        qs = item["conversations"][0]["value"]
        re = item["conversations"][1]["value"]

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], re)
        prompt = conv.get_prompt()

        try:
            # Load and process audio more efficiently
            speech, sample_rate = load(speech_file)
            speech = speech.squeeze(0)
            
            # Truncate very long audio to prevent OOM
            max_audio_length = 480000  # ~30 seconds at 16kHz, adjust as needed
            if speech.shape[0] > max_audio_length:
                speech = speech[:max_audio_length]
            
            input_ids = tokenizer_speech_token(prompt, self.tokenizer, return_tensors='pt')
            input_ids = input_ids.squeeze(0)  # Remove batch dimension for collate_fn
            
            return {
                "input_ids": input_ids,
                "labels": input_ids.clone(),
                "speech": speech,
                "speech_lengths": speech.shape[0]
            }
        except Exception as e:
            print(f"Error processing {speech_file}: {e}")
            # Return a smaller dummy item
            input_ids = tokenizer_speech_token(prompt, self.tokenizer, return_tensors='pt').squeeze(0)
            return {
                "input_ids": input_ids,
                "labels": input_ids.clone(),
                "speech": torch.zeros(8000, dtype=torch.float32),  # Smaller dummy
                "speech_lengths": 8000
            }

    def __len__(self):
        return len(self.questions)

def create_data_loader(questions, tokenizer, conv_mode, model_config, input_type, mel_size, device, batch_size=2, num_workers=1):
    dataset = CustomDataset(questions, tokenizer, conv_mode, model_config, input_type, mel_size, device)
    return dataset

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

class OptimizedMemoryCallback(TrainerCallback):
    def __init__(self):
        self.step_count = 0
        
    def on_step_end(self, args, state, control, **kwargs):
        self.step_count += 1
        
        # Clear cache every few steps
        if self.step_count % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()
            
        if torch.cuda.is_available() and self.step_count % 50 == 0:
            for device_id in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(device_id) / 1024**3
                reserved = torch.cuda.memory_reserved(device_id) / 1024**3
                print(f"[Step {state.global_step}] GPU {device_id} - Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")
        
        return control

def train_model(args):
    # Initialize distributed training if using multiple GPUs
    if 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        device = f'cuda:{local_rank}'
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set memory fraction to prevent OOM
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% of GPU memory
        
    model_path = os.path.expanduser(args.model_path)
    tokenizer, model, context_len = create_model(
        model_path, 
        args.model_base, 
        is_lora=args.is_lora, 
        s2s=args.s2s, 
        device=device
    )

    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    data_loader = create_data_loader(
        questions, 
        tokenizer, 
        args.conv_mode, 
        model.config, 
        args.input_type, 
        args.mel_size, 
        device
    )

    # Optimize model for training
    model = model.to(torch.bfloat16)
    
    # Freeze speech encoder to save memory
    for p in model.get_model().speech_encoder.parameters():
        p.requires_grad = False
    
    # Only train the projector
    for p in model.get_model().speech_projector.parameters():
        p.requires_grad = True

    # Set up tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Optimized training arguments for H100
    training_args = TrainingArguments(
        output_dir=f'saves/{args.output_folder}',
        overwrite_output_dir=True,
        do_train=True,
        do_eval=False,
        
        # Batch size optimization for H100
        per_device_train_batch_size=1,  # Start small, increase if stable
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,  # Effective batch size = 1 * 8 * num_gpus
        
        # Learning rate and optimization
        learning_rate=5e-5,  # Slightly lower for stability
        weight_decay=0.01,
        adam_beta2=0.95,
        warmup_ratio=0.03,
        lr_scheduler_type='cosine',
        
        # Memory optimization
        bf16=True,
        tf32=True,  # Use TF32 on H100 for better performance
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        dataloader_num_workers=2,  # Reduce if OOM persists
        
        # Distributed training settings
        ddp_find_unused_parameters=False,  # More efficient DDP
        ddp_broadcast_buffers=False,
        
        # Logging and saving
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        num_train_epochs=3,
        
        # Reporting
        report_to=['tensorboard', 'wandb'] if args.use_wandb else ['tensorboard'],
        seed=3407,
        
        # Additional optimizations
        remove_unused_columns=False,  # Keep all columns for custom forward
        prediction_loss_only=True,
        
        # Advanced memory settings
        max_grad_norm=1.0,  # Gradient clipping
    )

    # Clear cache before training
    torch.cuda.empty_cache()
    gc.collect()

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=data_loader,
        data_collator=collate_fn,
        callbacks=[OptimizedMemoryCallback()],
    )
    
    # Start training with error handling
    try:
        trainer.train()
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("CUDA OOM Error! Try reducing batch size or sequence length.")
            torch.cuda.empty_cache()
            gc.collect()
            raise
        else:
            raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--question-file", type=str)
    parser.add_argument("--conv-mode", type=str, default="v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--input_type", type=str, default="raw")
    parser.add_argument("--mel_size", type=int, default=128)
    parser.add_argument("--s2s", action="store_true", default=False)
    parser.add_argument("--is_lora", action="store_true", default=False)
    parser.add_argument("--output-folder", type=str, default=None)
    parser.add_argument("--use-wandb", action="store_true", default=True)
    args = parser.parse_args()
    train_model(args)