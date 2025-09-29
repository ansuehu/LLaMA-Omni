_pad        = '*'
_punctuation = '!,.:;?¡¿_|' #Castellano y Euskera
# _eos = '~'
# _punctuation = '!\'(),.:;? ' #Ingles
# _special = '-'
# #_letters = 'AÁBCDEÉFGHIÍJKLMNOÓPQRSTUÚVWXYZaábcdeéfghiíjklmnoópqrstuúvwxyzñÑ' #Castellano
# #_letters = 'AÁBCDEÉFGHIÍJKLMNOÓPQRSTUÚVWXYZaábcdeéfghiíjklmnoópqrstuúüvwxyzñÑ' #Castellanocondieresis
# _letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzñÑ' #Euskera
# #_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz' #Inglés

voiced_consonant = ['m','n','J','b','d','g','jj','I','L','l','r','rr','B','D','G','gj']
vowel = ['a','e','i','j','o','u','w',"'a","'e","'i","'o","'u"]
unvoiced = ['p', 't', 'c', 'k', 'tS', 'ts','ts`', 'tz', 'f', 'T', 's','s`', 'z', 'S', 'x']
# unvoiced = ['p', 't', 'c', 'k', 'tS', 'ts', 'tz', 'f', 'T', 's', 'z', 'S', 'x'] #Castellano
all_phonemes = 	voiced_consonant + vowel + unvoiced


# Export all symbols:
# symbols = [_pad, _eos] + list(_punctuation) + all_phonemes #+ [_eos]
symbols = [_pad] + list(_punctuation) + all_phonemes #+ [_eos]
