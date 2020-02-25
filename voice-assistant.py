# -*- coding: utf-8 -*-
# coding=utf-8
# @author: cer
import numpy as np
import pickle

import data.load
import io
from metrics.accuracy import conlleval
from data_loader import *
import my_metrics as metric
from keras.models import Sequential,Model
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras.layers.core import Dense, Dropout
from keras.layers.wrappers import TimeDistributed
from keras.layers import Convolution1D, MaxPooling1D, Bidirectional,Input, Embedding, LSTM, Dense
import progressbar
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import requests
import speech_recognition as sr
import test as t

sample_rate = 48000
chunk_size = 2048
r = sr.Recognizer()


model = t.NLU_Model()
actuator = t.RuleEngine()
# if input("Text mode? y/n\n")== 'y':
if 0:
    while 1:
        input_text = input("Hi how can I help you\n")
        slots = model.NLU(input_text)
        actuator.execute(slots)
else:
    with sr.Microphone(sample_rate=sample_rate,
                       chunk_size=chunk_size) as source:
        r.adjust_for_ambient_noise(source)
        r.dynamic_energy_threshold=True
        while 1:
            print("Hi how can I help you\n")
            audio = r.listen(source,phrase_time_limit=5)
            print("recording finished")
            try:
                input_text = r.recognize_google(audio, language="fa-IR")
                print("you said: " + input_text)
                slots = model.NLU(input_text)
                actuator.execute(slots)

            # error occurs when google could not understand what was said
            except sr.UnknownValueError:
                print("Google Speech Recognition could not understand audio")

            except sr.RequestError as e:
                print("Could not request results from Google Speech Recognition service; {0}".format(e))