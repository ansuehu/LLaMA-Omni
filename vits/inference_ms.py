"""
Scipt de inferencia de vits con modulo1y2.
Author: Inigo

Depues de un testeo rapido:
- Tiempo de inferencia media en gpu: 0.4s
- Tiempo dé inferencia media en cpu: 8s
"""

import matplotlib.pyplot as plt
import IPython.display as ipd

import os
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text.symbols_cast import symbols_cast
from text import text_to_sequence

from scipy.io.wavfile import write

import soundfile
import speech
import numpy as np
import time
import sys


def clean_text(text, language):
    output = speech.modulo1y2(text, mode='Word', PhTSimple='n', language=language, keep_chars = None, verbose = False)
    print(output)
    return output

def getPhones(text,language):
    #####################################
    # Extraccion fonetica de las frases #
    #####################################
    clean_text = text.lstrip()
    phones = speech.modulo1y2(clean_text, mode='Spell', PhTSimple='y', language=language, keep_chars = None, verbose = False)
    checker =  "".join(speech.modulo1y2(clean_text, mode='Word', PhTSimple='y', language=language, keep_chars = None, verbose = False))
    
    # Juntar frases muy largas si modulo1y2 lo ha separado.
    clp = ""
    for p in range(len(phones)):
        clp = clp + "".join(phones[p].split('-'))
        if p != len(phones)-1:
            clp = clp + ' | '
    slp = str(clp).split()
    
    if '?' in checker:
        slp.append('?')
    elif '!' in checker:
        slp.append('!')
    elif '.' in checker:
        slp.append('.')
    else:
        slp.append('.')
    if checker[-2] == ':':
        slp.append('.')
    if checker[-2] == ';':
        slp.append('.')
    phones = np.array(slp)
    return phones

def get_text(text, hps, language, path = False):

    if not path:
        text = getPhones(text,language)
    text_norm = text_to_sequence(text, hps.data.text_cleaners, language, inference= not path)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

# path = "/home/aholab/inigop/tts/vits/logs/finetuning_ibon_multi_eu_2_10/G_37000.pth"
path = "/home/aholab/inigoh/vits/logs/multiSpeakerEu/G_500000.pth"
hps = utils.get_hparams_from_file("/home/aholab/inigoh/vits/configs/MS.json")
device = torch.device('cuda')
language = "eu"
text = "Atzo okindegiratzen eta 10 euro aurkitu nituen."
speaker_num= 8
# speaker_num= 2

if language == "eu":
    lensym =  len(symbols)
else:
    lensym = len(symbols_cast)

net_g = SynthesizerTrn(
    lensym,
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model).to(device)
_ = net_g.eval()

_ = utils.load_checkpoint(path, net_g, None)

stn_tst = get_text(text, hps, language)
# stn_tst = get_text("/home/aholab/inigop/bips/corpus/corpus_pag_aintz_eus/AGE1127.pho.npy", hps, "eu", path= True)

with torch.no_grad():
    x_tst = stn_tst.to(device).unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
    sid = torch.LongTensor([speaker_num]).to(device)
    audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()

soundfile.write('multi8.wav', audio,hps.data.sampling_rate)