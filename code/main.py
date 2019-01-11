# -*- coding: utf-8 -*-
"""
  @Author: zzn
  @Date: 2018-11-12 14:13:52
  @Last Modified by:   zzn
  @Last Modified time: 2018-11-12 14:13:52
"""
import os
import pickle
import pandas as pd
import numpy as np
from collections import Counter
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM
from keras_contrib.layers import CRF
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


def train_txt2csv():
    file_names = [name.replace('.txt', '') for name in os.listdir(
        '../data/train/') if name.endswith('.txt')]
    train_file = open('../data/train.csv', 'w', encoding='utf-8')
    split_chars = ['。', '！', '？', '，']
    for name in file_names:
        text_file = open('../data/train/{}.txt'.format(name),
                         encoding='utf-8')
        text_string = text_file.read()
        text_file.close()
        text = []
        for char in text_string:

            if char == ' ':
                text.append('space')
            elif char == '\n':
                text.append('newline')
            else:
                text.append(char)
        tag = ['O']*len(text)
        tag_file = pd.read_csv(
            '../data/train/{}.ann'.format(name), header=None, sep='\t')
        for idx in range(len(tag_file)):
            label = tag_file.iloc[idx][1].split(' ')
            class_label, start, end = label[0], int(label[1]), int(label[-1])
            tag[start] = 'B-{}'.format(class_label)
            for i in range(start+1, end):
                tag[i] = 'I-{}'.format(class_label)
        cur_sentence = []
        cur_tags = []
        for k, char in enumerate(text):
            if char in split_chars:
                cur_sentence.append(char)
                cur_tags.append(tag[k])
                train_file.write(' '.join(cur_sentence) +
                                 '\t'+' '.join(cur_tags)+'\n')
                cur_sentence = []
                cur_tags = []
            else:
                cur_sentence.append(char)
                cur_tags.append(tag[k])
    train_file.close()


def get_train_data(max_length=500):
    with open('../data/train.csv', encoding='utf-8') as f:
        texts_tags = [line.strip() for line in f.readlines()]
    texts = [text_tag.split('\t')[0].split(' ') for text_tag in texts_tags]
    tags = [text_tag.split('\t')[1].split(' ') for text_tag in texts_tags]
    words = []
    for text in texts:
        words += text
    word_counts = Counter(words)
    vocab = [w for w, f in iter(word_counts.items()) if f >= 2]
    word2idx = dict((w, i+2) for i, w in enumerate(vocab))
    word2idx['pad'] = 0
    word2idx['unk'] = 1
    y_tags = ['O', 'B-Disease', 'I-Disease', 'B-Reason', 'I-Reason',
              "B-Symptom", "I-Symptom", "B-Test", "I-Test", "B-Test_Value",
              "I-Test_Value", "B-Drug", "I-Drug", "B-Frequency", "I-Frequency",
              "B-Amount", "I-Amount", "B-Treatment", "I-Treatment", "B-Operation",
              "I-Operation", "B-Method", "I-Method", "B-SideEff", "I-SideEff",
              "B-Anatomy", "I-Anatomy", "B-Level", "I-Level", "B-Duration", "I-Duration"]
    with open('../data/dict.pkl', 'wb') as outp:
        pickle.dump((word2idx, y_tags), outp)
    x = [[word2idx.get(word, 1) for word in text] for text in texts]
    y = [[y_tags.index(t) for t in tag] for tag in tags]
    if max_length is None:
        max_length = max(len(text) for text in texts)
    x = pad_sequences(x, max_length)

    y = pad_sequences(y, max_length, value=-1)
    y = np.expand_dims(y, 2)
    return x, y


def build_model():
    x, y = get_train_data(500)
    tr_x, val_x, tr_y, val_y = train_test_split(
        x, y, test_size=0.1, random_state=2018)
    with open('../data/dict.pkl', 'rb') as f:
        (word2idx, y_tags) = pickle.load(f)

    model = Sequential()
    model.add(Embedding(len(word2idx), 200, mask_zero=True))
    model.add(Bidirectional(LSTM(100, return_sequences=True,
                                 recurrent_dropout=0.1, dropout=0.1)))
    model.add(Bidirectional(LSTM(100, return_sequences=True,
                                 recurrent_dropout=0.1, dropout=0.1)))
    crf = CRF(len(y_tags), sparse_target=True)

    model.add(crf)
    model.summary()
    model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])

    return model, tr_x, tr_y, val_x, val_y


def predict_for_txt(filename, model, max_length=3000):
    f = open('../data/test_b/{}.txt'.format(filename), encoding='utf-8')
    txt_string = f.read()
    f.close()
    with open('../data/dict.pkl', 'rb') as f:
        (word2idx, y_tags) = pickle.load(f)

    texts = []
    index_texts = []
    texts_offset = []
    split_chars = ['。', '！', '？', '，']
    cur_text = []
    index_text = []
    texts_offset.append(0)
    texts_len = []
    for char in txt_string:
        if char == '\n':
            char = 'newline'
            idx = word2idx.get(char, 1)
            cur_text.append(char)
            index_text.append(idx)
        elif char == ' ':
            char = 'space'
            idx = word2idx.get(char, 1)
            cur_text.append(char)
            index_text.append(idx)
        elif char in split_chars:
            char = char
            idx = word2idx.get(char, 1)
            cur_text.append(char)
            index_text.append(idx)
            texts.append(cur_text)
            index_texts.append(index_text)
            texts_len.append(len(cur_text))
            texts_offset.append(len(cur_text))
            cur_text = []
            index_text = []
        else:
            char = char
            idx = word2idx.get(char, 1)
            cur_text.append(char)
            index_text.append(idx)
    texts.append(cur_text)
    index_texts.append(index_text)
    texts_len.append(len(cur_text))
    texts_offset = np.cumsum(texts_offset)
    index_texts = pad_sequences(index_texts, maxlen=max_length)
    proba = model.predict(index_texts, batch_size=32)
    proba = [p[-texts_len[i]:] for i, p in enumerate(proba)]
    results = [np.argmax(p, axis=1) for p in proba]
    results_tags = []
    for res in results:
        cur_res = []
        for r in res:
            if y_tags[r] == 'O':
                r = y_tags[r]
            else:
                r = y_tags[r].split('-')[1]
            cur_res.append(r)
        results_tags.append(cur_res)
    result_f = open('../submit/{}.ann'.format(filename),
                    'w', encoding='utf-8')
    num = 0
    for i, result_tag in enumerate(results_tags):
        pre = result_tag[0]
        start = 0
        for j in range(1, len(result_tag)):
            cur = result_tag[j]
            if cur != pre:
                end = j
                if pre != 'O':
                    num += 1
                    entry = texts[i][start:end]
                    entry = ''.join(entry)
                    entry = entry.replace('newline', ' ').replace('space', ' ')
                    result_f.write('T{}\t{} {} {}\t{}\n'.format(
                        num, pre, start+texts_offset[i], end+texts_offset[i], entry))
                start = j
                pre = cur
    result_f.close()


def main_():

    print('convert train .txt data to train .csv data...')
    train_txt2csv()
    print('build lstm-crf model...')

    model, tr_x, tr_y, val_x, val_y = build_model()
    print('start training...')
    model_path = '../data/best_.model'
    checkpiont = ModelCheckpoint(
        model_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    lr_reduce = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=2, verbose=1, mode='min')
    early_stop = EarlyStopping(
        monitor='val_loss', patience=10, verbose=1, mode='min')
    model.fit(tr_x, tr_y, batch_size=128, epochs=200, validation_data=[
              val_x, val_y], callbacks=[checkpiont, lr_reduce, early_stop])
    print('end training!')
    print('start testing...')
    model.load_weights(model_path)
    test_filenames = [name.replace('.txt', '')
                      for name in os.listdir('../data/test_b/')]
    for name in test_filenames:
        predict_for_txt(name, model)
    print('testing end!')


if __name__ == '__main__':
    main_()
