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


with open('word2index.pkl', 'rb') as f:
    word2index = pickle.load(f)
with open('index2slot.pkl', 'rb') as f:
    index2slot =  pickle.load(f)
with open('index2intent.pkl', 'rb') as f:
    index2intent =  pickle.load(f)
with open('index2word.pkl', 'rb') as f:
    index2word = pickle.load(f)


### Model
n_classes = len(index2slot)
n_intent = len(index2intent)
n_vocab = len(index2word)
print(index2slot)
# print(index2word)

class NLU_Model():
    model = None
    def __init__(self):
        main_input = Input(name='main_input',shape=(None,))
        x = Embedding(output_dim=50, input_dim=n_vocab)(main_input)
        lstm_out = Bidirectional(LSTM(units=50,return_sequences=True))(x)
        lstm_out_slot = Bidirectional(LSTM(units=50,return_sequences=True))(lstm_out)
        intent_out = Bidirectional(LSTM(units=50,return_sequences=False))(lstm_out)
        main_out = TimeDistributed(Dense(n_classes, activation='softmax'))(lstm_out_slot)
        intent_out = Dense(n_intent, activation='softmax')(intent_out)
        model = Model(inputs= main_input , outputs= [main_out,intent_out])
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        print(model.summary())
        model.load_weights('best_model_weights.h5',by_name=True)
        self.model = model

    def NLU(self,input_utterance):
        input_token = input_utterance.split(" ")
        print(input_token)
        input_index = list(map(lambda i: word2index[i] if i in word2index else word2index["<UNK>"], input_token))
        print(input_index)
        input_index = np.array(input_index)
        input_index = input_index[np.newaxis, :]
        pred = self.model.predict_on_batch(input_index)
        slot_pred = np.argmax(np.array(pred[0]), -1)[0]
        slot_pred = list(map(lambda x: index2slot[x], slot_pred))
        print(slot_pred)
        return (slot_pred)

class RuleEngine:
    def execute(self,slots):
        if 'device_lamp' in slots:
            if "command_off" in slots:
                requests.get('http://192.168.1.242:82/led00')
            if "command_on" in slots:
                requests.get('http://192.168.1.242:82/led10')
            if "command_decrease" in slots:
                requests.get('http://192.168.1.242:82/led10')
            if "command_increase" in slots:
                requests.get('http://192.168.1.242:82/led11')
        if "device_tv" in slots:
            if "command_off" in slots:
                requests.get('http://192.168.1.242:82/ir_send1')
            if "command_on" in slots:
                requests.get('http://192.168.1.242:82/ir_send1')
        if "volume" in slots:
            if "command_decrease" in slots:
                requests.get('http://192.168.1.242:82/ir_send13')
                requests.get('http://192.168.1.242:82/ir_send13')
                requests.get('http://192.168.1.242:82/ir_send13')
            if "command_increase" in slots:
                requests.get('http://192.168.1.242:82/ir_send12')
                requests.get('http://192.168.1.242:82/ir_send12')
                requests.get('http://192.168.1.242:82/ir_send12')
        if "command_change_channel" in slots:
            if "command_decrease" in slots:
                requests.get('http://192.168.1.242:82/ir_send15')
            if "command_increase" in slots:
                requests.get('http://192.168.1.242:82/ir_send14')
            if "channel_num_1" in slots:
                requests.get('http://192.168.1.242:82/ir_send2')
            if "channel_num_2" in slots:
                requests.get('http://192.168.1.242:82/ir_send3')
            if "channel_num_3" in slots:
                requests.get('http://192.168.1.242:82/ir_send4')
            if "channel_num_4" in slots:
                requests.get('http://192.168.1.242:82/ir_send5')
            if "channel_num_5" in slots:
                requests.get('http://192.168.1.242:82/ir_send6')
            if "channel_num_6" in slots:
                requests.get('http://192.168.1.242:82/ir_send7')
            if "channel_num_7" in slots:
                requests.get('http://192.168.1.242:82/ir_send8')
            if "channel_num_8" in slots:
                requests.get('http://192.168.1.242:82/ir_send9')
            if "channel_num_9" in slots:
                requests.get('http://192.168.1.242:82/ir_send10')
            if "channel_num_10" in slots:
                requests.get('http://192.168.1.242:82/ir_send2')
                requests.get('http://192.168.1.242:82/ir_send11')
            if "channel_num_11" in slots:
                requests.get('http://192.168.1.242:82/ir_send2')
                requests.get('http://192.168.1.242:82/ir_send2')
            if "channel_num_12" in slots:
                requests.get('http://192.168.1.242:82/ir_send2')
                requests.get('http://192.168.1.242:82/ir_send3')
            if "channel_num_13" in slots:
                requests.get('http://192.168.1.242:82/ir_send2')
                requests.get('http://192.168.1.242:82/ir_send4')
            if "channel_num_14" in slots:
                requests.get('http://192.168.1.242:82/ir_send2')
                requests.get('http://192.168.1.242:82/ir_send5')
            if "channel_num_15" in slots:
                requests.get('http://192.168.1.242:82/ir_send2')
                requests.get('http://192.168.1.242:82/ir_send6')
            if "channel_num_16" in slots:
                requests.get('http://192.168.1.242:82/ir_send2')
                requests.get('http://192.168.1.242:82/ir_send7')
            if "channel_num_17" in slots:
                requests.get('http://192.168.1.242:82/ir_send2')
                requests.get('http://192.168.1.242:82/ir_send8')
            if "channel_num_18" in slots:
                requests.get('http://192.168.1.242:82/ir_send2')
                requests.get('http://192.168.1.242:82/ir_send9')
            if "channel_num_19" in slots:
                requests.get('http://192.168.1.242:82/ir_send2')
                requests.get('http://192.168.1.242:82/ir_send10')
            if "channel_num_20" in slots:
                requests.get('http://192.168.1.242:82/ir_send3')
                requests.get('http://192.168.1.242:82/ir_send11')

