import torch
import torch.nn as nn
import pickle

from AIModels import NeuralASR 

def getASRModel(language: str,use_whisper:bool=True):


    if use_whisper:
        from whisper import WhisperASRModel
        return WhisperASRModel()
    if language == 'en':
        model, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                                model='silero_stt',
                                                language='en',
                                                device=torch.device('cpu'))
        model.eval()
        return NeuralASR(model, decoder)
    else:
        raise ValueError('Language not implemented')
