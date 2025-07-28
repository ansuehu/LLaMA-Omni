import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from omni_speech.model.builder import create_model
import argparse
import os
import torch
from torch.utils.data import Dataset, DataLoader
# import whisper
# from whisper.audio import load_audio
from torchaudio import load
from omni_speech.conversation import conv_templates
import math
import json
from omni_speech.datasets.preprocess import tokenizer_speech_token
from transformers import TrainingArguments, Trainer, HubertModel
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

# Custom dataset class

def collate_fn(batch):
    # for i in range(len(batch)):
    #     batch[i]= batch[i].values()
        
    input_ids, labels, speech_tensors, speech_lengths = zip(
        *[list(item.values()) for item in batch]
    )
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=128009)
    labels = pad_sequence(labels, batch_first=True, padding_value=128009)

    # Transpose speech tensors to [length, 1] for pad_sequence
    speech_tensors = [t.transpose(0, 1) if t.shape[0] == 1 else t for t in speech_tensors]
    speech_tensors = pad_sequence(speech_tensors, batch_first=True, padding_value=0)
    # Transpose back to [batch, 1, length]
    # speech_tensors = speech_tensors.transpose(1, 2)

    # speech_tensors = torch.stack(speech_tensors, dim=0)
    return {"input_ids":input_ids,"labels":labels, "speech":speech_tensors, "speech_lengths":speech_lengths}

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

        speech = load(speech_file)[0] # Torchaudio returns a tuple (waveform, sample_rate)
        speech = speech.squeeze(0)
        # if self.input_type == "raw":
        #     # speech = torch.from_numpy(speech) # If loading with torchaudio this is not needed
        #     if self.model_config.speech_normalize:
        #         speech = torch.nn.functional.layer_norm(speech, speech.shape)
        # elif self.input_type == "mel":
        #     speech = whisper.pad_or_trim(speech)
        #     speech = whisper.log_mel_spectrogram(speech, n_mels=self.mel_size).permute(1, 0)
        input_ids = tokenizer_speech_token(prompt, self.tokenizer, return_tensors='pt')
        ret=dict(
            input_ids=input_ids.to(self.device),
            labels=input_ids.to(self.device),
            speech=speech.to(torch.bfloat16).to(self.device),
            speech_lengths=torch.LongTensor([speech.shape[0]]).to(self.device)
            )
        return ret
    def __len__(self):
        return len(self.questions)
    
# DataLoader
def create_data_loader(questions, tokenizer, conv_mode, model_config, input_type, mel_size, device, batch_size=2, num_workers=1):
    # assert batch_size == 1, "batch_size must be 1"
    
    dataset = CustomDataset(questions, tokenizer, conv_mode, model_config, input_type, mel_size, device)
    #data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return dataset


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def train_model(args):
    # 设置每张卡的device
    
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'     # Log output target; if you don't want to use wandb, you can set it to None.
    # device = 'cpu' 

    model_path = os.path.expanduser(args.model_path)
    tokenizer, model, context_len = create_model(model_path, args.model_base, is_lora=args.is_lora, s2s=args.s2s, device=device)

    # speech_encoder = model.get_model().speech_encoder
    # hubert = HubertModel.from_pretrained("Ansu/mHubert-basque-k1000-L9")

    # def compare_model_weights(model1, model2):
    #     for (name1, param1), (name2, param2) in zip(tqdm(model1.named_parameters()), model2.named_parameters()):
    #         if not torch.equal(param1, param2):
    #             print(f"Difference found in layer: {name1} vs {name2}: {param1} vs {param2}")
    #             return False
    #     print("All weights are the same.")
    #     return True
    
    # compare_model_weights(speech_encoder, hubert)
    # return

    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx) #chunk 1 chunk-idx 0 取list中的多少进行测试
    data_loader = create_data_loader(questions, tokenizer, args.conv_mode, model.config, args.input_type, args.mel_size, device)


    # 初始化Trainer
    training_args = TrainingArguments(
        output_dir='saves',                         # Output_path, including checkpoints, intermediate results, etc
        overwrite_output_dir=True,                  # Wheter to overwrite output_dir
        do_train=True,                              # Wheter to train
        do_eval=True,                               # Whether to evaluate
        eval_steps=1,                               # Evaluation step interval
        per_device_train_batch_size=2,              # Per-device batch size
        gradient_accumulation_steps=6,              # Gradient accumulation step size, saves video memory, but is not necessary fo small models, using 1 converges faster
        learning_rate=1e-4,
        weight_decay=0.01,
        adam_beta2=0.95,
        warmup_ratio=0.01,
        lr_scheduler_type='cosine',                 # Learning rate scheduling strategy, LLM training generally uses cosine
        logging_steps=1,                            # Print step interval
        report_to='tensorboard',                    # Log output target
        num_train_epochs=50,                        # Number of training rounds, 2 ~ 3 is enough
        save_steps=1000,                            # Checkpoint save step interval
        save_total_limit=2,                         # maximum number of checkpoints to keep in output_dir
        seed=3407,                                  # random seed
        bf16=True,                                  # Wheter to enable mixed precision training
        use_cpu=True if device == 'cpu' else False, # Whether to use CPU for training, if you use GPU, set it to False
        dataloader_pin_memory=False,                # Whether to pin memory in dataloader, if you use CPU, set it to False
    )
    tokenizer.pad_token = tokenizer.eos_token
    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=data_loader,
        eval_dataset=data_loader,
        data_collator=collate_fn,
    )
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--question-file", type=str)
    parser.add_argument("--answer-file", type=str)
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
    args = parser.parse_args()
    train_model(args)