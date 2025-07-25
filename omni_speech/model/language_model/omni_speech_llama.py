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

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..omni_speech_arch import OmniSpeechMetaModel, OmniSpeechMetaForCausalLM


class OmniSpeechConfig(LlamaConfig):
    model_type = "omni_speech_llama"
    encoder_path = "Ansu/mHubert-basque-k1000-L9"
    speech_encoder_type = "hubert"  # "whisper" or "hubert"
    encoder_type = "finetune"  # "pretrain" or "finetune"
    speech_projector_lr = 0.001
    speech_projector_type = "linear"
    tune_speech_projector = True
    freeze_speech_projector = False
    speech_encoder_ds_rate = 5
    speech_encoder_hidden_size = 768 # if using whisper, it should be 1280


class OmniSpeechLlamaModel(OmniSpeechMetaModel, LlamaModel):
    config_class = OmniSpeechConfig

    def __init__(self, config: LlamaConfig):
        super(OmniSpeechLlamaModel, self).__init__(config)


class OmniSpeechLlamaForCausalLM(LlamaForCausalLM, OmniSpeechMetaForCausalLM):
    config_class = OmniSpeechConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = OmniSpeechLlamaModel(config)   #llm和speech_ecoder 和projector
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        speech: Optional[torch.FloatTensor] = None,
        speech_lengths: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if inputs_embeds is None: #inputs_embeds none
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_speech_and_text(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                speech,
                speech_lengths
            )
            result=super().forward(
                input_ids=input_ids, #none
                attention_mask=attention_mask, #none
                position_ids=position_ids, #none
                past_key_values=past_key_values, #none
                inputs_embeds=inputs_embeds, #tesnor[1,361,2048]
                labels=labels, #none
                use_cache=use_cache, #True
                output_attentions=output_attentions, #none
                output_hidden_states=output_hidden_states, #none
                return_dict=return_dict #none
            )
        return result

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        speech: Optional[torch.Tensor] = None,
        speech_lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if speech is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_speech_and_text(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                speech,
                speech_lengths
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        speech = kwargs.pop("speech", None)
        speech_lengths = kwargs.pop("speech_lengths", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if speech is not None:
            inputs['speech'] = speech
            inputs['speech_lengths'] = speech_lengths
        return inputs
#
AutoConfig.register("omni_speech_llama", OmniSpeechConfig) 
AutoModelForCausalLM.register(OmniSpeechConfig, OmniSpeechLlamaForCausalLM)
