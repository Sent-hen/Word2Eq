import torch
import numpy as np

START_TOKEN = ''
END_TOKEN = ''
PADDING_TOKEN = ''

equation_vocabulary = [
    START_TOKEN, 
    '1', 
    '2', 
    '3', 
    '4', 
    '5', 
    '6', 
    '7', 
    '8', 
    '9', 
    '0', 
    ' ', 
    '*', 
    '+', 
    ',', 
    '-', 
    '.', 
    '/', 
    '=',
    PADDING_TOKEN,
    END_TOKEN,
]

index_to_equation = {k:v for k,v in enumerate(equation_vocabulary)}
equation_to_index = {v:k for k,v in enumerate(equation_vocabulary)}

class BPETokenizer():
    pass

class EquationVocabularyTokenizer():
    pass

if __name__ == '__main__':
    print(index_to_equation[18])