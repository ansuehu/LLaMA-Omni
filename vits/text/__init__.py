""" from https://github.com/keithito/tacotron """
from text import cleaners
from text.symbols import symbols
from text.symbols_cast import symbols_cast
import numpy as np


# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

_symbol_to_id_cast = {s: i for i, s in enumerate(symbols_cast)}
_id_to_symbol_cast = {i: s for i, s in enumerate(symbols_cast)}

def string_to_sequence(text, cleaner_names):
  sequence = []
  sequence += _symbols_to_sequence(text.tolist())
  #print(sequence)
  return sequence

def text_to_sequence(text, cleaner_names, language, inference=False):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

    The text can optionally have ARPAbet sequences enclosed in curly braces embedded
    in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through

    Returns:
      List of integers corresponding to the symbols in the text
  '''
  
  sequence = []
  if inference:
    a = text
  else:
    a = np.load(text).tolist()
    
  if language == 'eu':
    sequence += _symbols_to_sequence(a)
  elif language == 'es':
    sequence += _symbols_to_sequence_cast(a)

  return sequence

# def text_to_sequence(text, cleaner_names):
#   '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
#     Args:
#       text: string to convert to a sequence
#       cleaner_names: names of the cleaner functions to run the text through
#     Returns:
#       List of integers corresponding to the symbols in the text
#   '''
#   sequence = []

#   clean_text = _clean_text(text, cleaner_names)
#   for symbol in clean_text:
#     symbol_id = _symbol_to_id[symbol]
#     sequence += [symbol_id]
#   return sequence

def cleaned_text_to_sequence(text, language):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

    The text can optionally have ARPAbet sequences enclosed in curly braces embedded
    in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through

    Returns:
      List of integers corresponding to the symbols in the text
  '''
  
  sequence = []
  a = np.load(text).tolist()
    
  if language == 'eu':
    sequence += _symbols_to_sequence(a)
  elif language == 'es':
    sequence += _symbols_to_sequence_cast(a)

  return sequence

# def cleaned_text_to_sequence(cleaned_text):
#   '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
#     Args:
#       text: string to convert to a sequence
#     Returns:
#       List of integers corresponding to the symbols in the text
#   '''
#   sequence = [_symbol_to_id[symbol] for symbol in cleaned_text]
#   return sequence


def sequence_to_text(sequence):
  '''Converts a sequence of IDs back to a string'''
  result = ''
  for symbol_id in sequence:
    s = _id_to_symbol[symbol_id]
    result += s
  return result


def _clean_text(text, cleaner_names):
  for name in cleaner_names:
    cleaner = getattr(cleaners, name)
    if not cleaner:
      raise Exception('Unknown cleaner: %s' % name)
    text = cleaner(text)
  return text

def _symbols_to_sequence(symbols):
  return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]

def _symbols_to_sequence_cast(symbols):
  return [_symbol_to_id_cast[s] for s in symbols if _should_keep_symbol(s)]

def _should_keep_symbol(s):
  return s in _symbol_to_id and s is not '@' and s is not '~'