# Adopted from https://github.com/haotian-liu/LLaVA. We modify the code to support speech input. Below is the original copyright:
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import warnings
import shutil

from transformers import AutoTokenizer, LlamaTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from omni_speech.model import *
from omni_speech.model.speech_encoder.builder import build_speech_encoder
from omni_speech.model.speech_projector.builder import build_speech_projector
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_pretrained_model(model_path, model_base, is_lora=False, s2s=False, load_8bit=False, load_4bit=False, device="cuda", use_flash_attn=False, **kwargs):
    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'
    
    model_cls = OmniSpeech2SLlamaForCausalLM if s2s else OmniSpeechLlamaForCausalLM

    # Load OmniSpeech model
    if is_lora:
        assert model_base is not None, "model_base is required for LoRA models."
        from omni_speech.model.language_model.omni_speech_llama import OmniSpeechConfig
        lora_cfg_pretrained = OmniSpeechConfig.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
        print('Loading OmniSpeech from base model...')
        model = model_cls.from_pretrained(model_base, low_cpu_mem_usage=False, config=lora_cfg_pretrained, **kwargs)
        print('Loading additional OmniSpeech weights...')
        if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
            non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
        non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
        if any(k.startswith('model.model.') for k in non_lora_trainables):
            non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
        model.load_state_dict(non_lora_trainables, strict=False)

        from peft import PeftModel
        print('Loading LoRA weights...')
        model = PeftModel.from_pretrained(model, model_path)
        print('Merging LoRA weights...')
        model = model.merge_and_unload()
        print('Model is loaded...')
    elif model_base is not None:
        print('Loading OmniSpeech from base model...')
        tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
        cfg_pretrained = AutoConfig.from_pretrained(model_path)
        model = model_cls.from_pretrained(model_base, low_cpu_mem_usage=False, config=cfg_pretrained, **kwargs)
        
        speech_projector_weights = torch.load(os.path.join(model_path, 'speech_projector.bin'), map_location='cpu')
        speech_projector_weights = {k: v.to(torch.float16) for k, v in speech_projector_weights.items()}
        model.load_state_dict(speech_projector_weights, strict=False)
        model = model.to(device=device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = model_cls.from_pretrained(
            model_path,
            low_cpu_mem_usage=False,
            **kwargs
        )

    model.get_model().speech_encoder = build_speech_encoder(model.config)
    model.get_model().speech_encoder.to(dtype=torch.float16)

    model.get_model().speech_projector = build_speech_projector(model.config)
    model.get_model().speech_projector.to(dtype=torch.float16)
    model = model.to(device=device)

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, context_len

def create_model(model_path, model_base, is_lora=False, s2s=False, load_8bit=False, load_4bit=False, device="cuda", use_flash_attn=False, **kwargs):
    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.bfloat16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'
    model_cls = OmniSpeech2SLlamaForCausalLM if s2s else OmniSpeechLlamaForCausalLM
    if is_lora:
        assert model_base is not None, "model_base is required for LoRA models."
        from omni_speech.model.language_model.omni_speech_llama import OmniSpeechConfig
        lora_cfg_pretrained = OmniSpeechConfig.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
        print('Loading OmniSpeech from base model...')
        model = model_cls.from_pretrained(model_base, low_cpu_mem_usage=False, config=lora_cfg_pretrained, **kwargs)
        print('Loading additional OmniSpeech weights...')
        # if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
        #     non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
        #     non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
        # if any(k.startswith('model.model.') for k in non_lora_trainables):
        #     non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
        #model.load_state_dict(non_lora_trainables, strict=False)

        from peft import PeftModel
        print('Loading LoRA weights...')
        model = PeftModel.from_pretrained(model, model_path)
        print('Merging LoRA weights...')
        model = model.merge_and_unload()
        print('Model is loaded...')
        
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = model_cls.from_pretrained(
            model_path,
            low_cpu_mem_usage=False,
            **kwargs,
        )
    
    if s2s:
        model.initialize_speech_generator(model.config)
    
    model = model.to(device=device,dtype=torch.bfloat16)
    model.get_model().speech_encoder = build_speech_encoder(model.config)
    model.get_model().speech_projector = build_speech_projector(model.config)
    # model.get_model().speech_encoder = build_speech_encoder(model.config)
    # model.get_model().speech_encoder.to(device=device, dtype=torch.bfloat16)
    #冻住speech_encoder的参数
    for param in model.get_model().speech_encoder.parameters():
        param.requires_grad = False
        
    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, context_len
