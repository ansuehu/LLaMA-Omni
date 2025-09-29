# -*- coding: utf-8 -*-
"""
Script de inferencia de VITS con modulo1y2.
Adaptado para múltiples generadores y frases.
Author: Iñigo (modificado por Mariana)
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

# Verifica qué speech se está usando (LIBBERTSO o /vits)
print("Usando speech desde:", speech.__file__)

def clean_text(text, language):
    output = speech.modulo1y2(text, mode='Word', PhTSimple='n', language=language, keep_chars=None, verbose=True)
    print(output)
    return output

def getPhones(text, language):
    #####################################
    # Extracción fonética de las frases #
    #####################################
    clean_text = text.lstrip()
    phones = speech.modulo1y2(clean_text, mode='Spell', PhTSimple='y', language=language, keep_chars=None, verbose=True)
    checker = "".join(speech.modulo1y2(clean_text, mode='Word', PhTSimple='y', language=language, keep_chars=None, verbose=True))

    clp = ""
    for p in range(len(phones)):
        clp = clp + "".join(phones[p].split('-'))
        if p != len(phones) - 1:
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
    if checker[-2] == ':' or checker[-2] == ';':
        slp.append('.')
    phones = np.array(slp)

    print(phones)
    return phones

def get_text(text, hps, language, path=False):
    if not path:
        text = getPhones(text, language)
        print(text)
    text_norm = text_to_sequence(text, hps.data.text_cleaners, language, inference=not path)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

#--------------------------- CONFIGURACIÓN MULTIPLE -------------------------------------------
textos = {

    "01": "Euskal Encounterreko adituek baieztatu dute adimen artifiziala lana, harremanak eta kultura eraldatzen ari dela, erabilera akritikoaren arriskuez ohartarazi dute.",
    "02": "Eguneroko erabilerako produktu komertzialagoek edo sintetikoagoek disruptore endokrino gehiago dituzte, bereziki produktu ekologikoekin alderatuta.",
    "03": "Zizur Nagusiak, berriz, irailaren hamarrean abiatuko ditu jaiak, eta irailaren hamalaura arte luzatuko dira.",
    "04": "Labearen tenperatura 180 gradutara jeitsi, salda bota eta beste 35-40 minutuz erre.",
    "05": "Juan Garmendia Larrañaga etnografoak jaso du Eusko Ikaskuntza.",
    "06": "Komunitate zientifikoak euforiaz eta eszeptizismoz hartu du iragarpena.",
    "07": "Lurraldearen eta eskualdearen orekaren eta saltoki bakoitzaren pisuaren arabera banatuko dira bonuak dendetan.",
    "08": "Zein magnitude fisiko ditugu jokoan oraingoan?",
    "09": "Pertsona eta agentearen lankidetza oinarri duten egiturak.",
    "10": "Zer ederra."
      

}

rutas = {
#Marina
    "marina": {
        "path": "/home/aholab/mariana/vits/logs/marina/G_612000.pth", 
        "config": "/home/aholab/mariana/vits/configs/sonora.json",
        "device": "cpu"
    },
#Alex    
    "alex": {
        "path": "/home/aholab/mariana/vits/logs/alex_sonora/G_540000.pth",
        "config": "/home/aholab/mariana/vits/configs/sonora.json",
        "device": "cpu"
# #Dariana (F2)       
#     },
#     "32": {
#         "path": "/home/aholab/mariana/vits/logs/KA_MX_F2/G_17000.pth",
#         "config": "/home/aholab/mariana/vits/configs/personalized_16.json",
#         "device": "cpu"
#     },
# #Marco (M2)    
#     "42": {
#         "path": "/home/aholab/mariana/vits/logs/KA_MX_M2/G_17000.pth",
#         "config": "/home/aholab/mariana/vits/configs/personalized_16.json",
#         "device": "cpu"
    }
    
}

language = "eu"
#language = "es"
OUTPUT_DIR = "/home/aholab/mariana/sonora/eval/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

for clave, datos in rutas.items():
    path = datos["path"]
    hps = utils.get_hparams_from_file(datos["config"])
    device = torch.device(datos["device"])

    if language == "eu":
        lensym = len(symbols)
    else:
        lensym = len(symbols_cast)
    
    # if language == "es":    
    #     lensym = len(symbols_cast)
    # else:
    #     lensym = len(symbols_cast)

    net_g = SynthesizerTrn(
        lensym,
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).to(device)
    _ = net_g.eval()
    _ = utils.load_checkpoint(path, net_g, None)

    for idx, text in textos.items():
        stn_tst = get_text(text, hps, language)
        with torch.no_grad():
            x_tst = stn_tst.to(device).unsqueeze(0)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
            tiempo = time.time()
            audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0, 0].data.cpu().float().numpy()

        nombre_archivo = f"{clave}{idx}.wav"
        soundfile.write(os.path.join(OUTPUT_DIR, nombre_archivo), audio, hps.data.sampling_rate)
        print(f"[✓] {nombre_archivo} guardado")

print("\nINFERENCIAS COMPLETADAS PARA TODOS LOS GENERADORES ;)")