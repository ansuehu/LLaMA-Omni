_pad        = '@'
_punctuation = '!,.:;?¡¿_|' #Castellano y Euskera
# _punctuation = '!\'(),.:;? ' #Ingles
# _special = '-'
# #_letters = 'AÁBCDEÉFGHIÍJKLMNOÓPQRSTUÚVWXYZaábcdeéfghiíjklmnoópqrstuúvwxyzñÑ' #Castellano
# #_letters = 'AÁBCDEÉFGHIÍJKLMNOÓPQRSTUÚVWXYZaábcdeéfghiíjklmnoópqrstuúüvwxyzñÑ' #Castellanocondieresis
# _letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzñÑ' #Euskera
# #_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz' #Inglés

voiced_consonant = ['m','n','J','b','d','g','jj','I','L','l','r','rr','B','D','G','gj']
vowel = ['a','e','i','j','o','u','w',"'a","'e","'i","'o","'u"]
unvoiced = ['p', 't', 'c', 'k', 'tS', 'ts', 'tz', 'f', 'T', 's', 'z', 'S', 'x']
all_phonemes = 	voiced_consonant + vowel + unvoiced

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
#_arpabet = ['@' + s for s in cmudict.valid_symbols]

# Export all symbols:
# symbols = [_pad] + list(_special) + list(_punctuation) + list(_letters) + _arpabet #+ [_eos]
symbols_cast = [_pad] + list(_punctuation) + all_phonemes #+ [_eos]
