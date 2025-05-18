# import ModelInterfaces
import torch
import numpy as np
import epitran
import eng_to_ipa


def get_phonem_converter(language: str):
    if language == 'en':
        return EngPhonemConverter()
    else:
        raise ValueError('Only English language is supported')

    return phonem_converter






class EngPhonemConverter():

    def __init__(self,) -> None:
        super().__init__()

    def convertToPhonem(self, sentence: str) -> str:
        phonem_representation = eng_to_ipa.convert(sentence)
        phonem_representation = phonem_representation.replace('*','')
        return phonem_representation
