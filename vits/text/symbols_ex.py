_pad        = '*'
_punctuation = '!,.:;?¡¿_|' #Castellano y Euskera
# _eos = '~'
# _punctuation = '!\'(),.:;? ' #Ingles
# _special = '-'
# #_letters = 'AÁBCDEÉFGHIÍJKLMNOÓPQRSTUÚVWXYZaábcdeéfghiíjklmnoópqrstuúvwxyzñÑ' #Castellano
# #_letters = 'AÁBCDEÉFGHIÍJKLMNOÓPQRSTUÚVWXYZaábcdeéfghiíjklmnoópqrstuúüvwxyzñÑ' #Castellanocondieresis
# _letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzñÑ' #Euskera
# #_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz' #Inglés

#voiced_consonant = ['m','n','J','b','d','g','jj','I','L','l','r','rr','B','D','G','gj']
#vowel = ['#', 'B', 'D', 'E', 'G', 'J', 'N', 'O', 'S', 'T','Z', 'a', 'b', 'd', 'e', 'f', 'g', 'i', 'k', 'l','m', 'n', 'o', 'p', 'r', 's', 't', 'u', 'x','j', 'w', 'L','c','s`',]
all_phonemes = ["@","0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "'@","'0", "'1", "'2", "'3", "'4", "'5", "'6", "'7", "'8", "'9", "'a", "'b", "'c", "'d", "'e", "'f", "'g", "'h", "'i", "'j", "'k", "'l", "'m", "'n", "'o", "'p", "'q", "'r", "'s", "'t", "'u", "'v", "'w", "'x", "'y", "'z", "'A", "'B", "'C", "'D", "'E", "'F", "'G", "'H", "'I", "'J", "'K", "'L", "'M", "'N", "'O", "'P", "'Q", "'R", "'S", "'T", "'U", "'V", "'W", "'X", "'Y", "'Z"]

# unvoiced = ['p', 't', 'c', 'k', 'tS', 'ts', 'tz', 'f', 'T', 's', 'z', 'S', 'x'] #Castellano
#other =['A','R','C','F','H','I','K']

#'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\', ']', '^', '_', '`', '{', '|', '}', '~',

# Export all symbols:
# symbols = [_pad, _eos] + list(_punctuation) + all_phonemes #+ [_eos]
symbols_ex = [_pad] + list(_punctuation) + all_phonemes #+ [_eos]
#symbols_ex =  all_phonemes #+ [_eos]
