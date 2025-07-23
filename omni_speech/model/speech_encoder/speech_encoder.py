# Adopted from https://github.com/ddlBoJack/SLAM-LLM/blob/main/src/slam_llm/models/encoder.py

import types
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import HubertModel

class WhisperWrappedEncoder:
    
    @classmethod
    def load(cls, model_config, device='cpu'):

        def replace_layer_norm(module):
            from whisper.model import LayerNorm
            for name, child in module.named_children():
                if isinstance(child, LayerNorm):
                    old_params = child.state_dict()
                    new_layer_norm = nn.LayerNorm(child.normalized_shape, eps=child.eps, elementwise_affine=child.elementwise_affine)
                    new_layer_norm.load_state_dict(old_params)
                    setattr(module, name, new_layer_norm)
                else:
                    replace_layer_norm(child)
        import whisper
        encoder = whisper.load_model(name=model_config.speech_encoder, device=device).encoder
        replace_layer_norm(encoder)
        
        return encoder
    
class HubertEncoder:
    
    @classmethod
    def load(cls, model_config):
        print(f'Encoder path: {model_config.encoder_path}')
        model = HubertModel.from_pretrained(model_config.encoder_path)
        # if model_config.encoder_type == "pretrain":
        #     pass
        # elif model_config.encoder_type == "finetune":
        #     model.w2v_encoder.proj = None
        #     model.w2v_encoder.apply_mask = False
        # else:
        #     assert model_config.encoder_type in ["pretrain", "finetune"], "input_type must be one of [pretrain, finetune]" 
        return model
    