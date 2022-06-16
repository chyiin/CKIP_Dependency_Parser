
import networkx as nx
import numpy as np
import streamlit as st
from PIL import Image
import rpyc
import json

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import nltk
from nltk.tokenize import word_tokenize
from transformers import XLNetTokenizer, BertTokenizer
from XLNet_encoder import XLNet_Encoder1
from XLNet_punct import XLNet_Punct
from BERT_encoder import BERT_Encoder1
from BERT_punct import BERT_Punct
import matplotlib.pyplot as plt
from opencc import OpenCC
from mst import fast_parse

s2t = OpenCC('s2t')
t2s = OpenCC('t2s')
nltk.download('punkt')   

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
if torch.cuda.is_available():
    torch.cuda.set_device(1)

def tokenize_and_preserve_labels(tokenizer, subsent):
    
    tokenized_sentence, labels_idx, seq = [], [], [0]
    n, idx = 0, 0
    
    for word in subsent:

        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)
        n = n + n_subwords
        tokenized_sentence.extend(tokenized_word)
        labels_idx.extend([idx] * n_subwords)
        seq.append(n)
        idx = idx + 1
        
    return tokenized_sentence, torch.tensor(labels_idx), seq[:-1]

@st.cache(allow_output_mutation=True)
class english_parser():

    def __init__(self):

        self.ids_to_labels = {0: 'root', 1: 'prep', 2: 'det', 3: 'nn', 4: 'num', 5: 'pobj', 6: 'punct', 7: 'poss', 8: 'possessive', 9: 'amod', 10: 'nsubj', 11: 'appos', 12: 'dobj', 13: 'dep', 14: 'cc', 15: 'conj', 16: 'nsubjpass', 17: 'partmod', 18: 'auxpass', 19: 'advmod', 20: 'ccomp', 21: 'aux', 22: 'cop', 23: 'xcomp', 24: 'quantmod', 25: 'tmod', 26: 'neg', 27: 'infmod', 28: 'rcmod', 29: 'pcomp', 30: 'mark', 31: 'advcl', 32: 'predet', 33: 'csubj', 34: 'mwe', 35: 'parataxis', 36: 'npadvmod', 37: 'number', 38: 'acomp', 39: 'prt', 40: 'iobj', 41: 'preconj', 42: 'expl', 43: 'discourse', 44: 'csubjpass'}

        self.tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')
        self.model = XLNet_Encoder1(hidden_size=1024, pretrained="xlnet-large-cased").cuda()
        self.model.load_state_dict(torch.load(f'demo_model/xlnet_encoder.pt')) 

        self.punct_model = XLNet_Punct(loaded_model='demo_model/xlnet_encoder.pt', hidden_size=1024, num_labels=len(self.ids_to_labels)).cuda()
        self.punct_model.load_state_dict(torch.load(f'demo_model/xlnet_punct_encoder.pt')) 

    def output(self, input_text):

        input_sentence = word_tokenize(input_text)
        input = ['root'] + input_sentence
        sentence_index = []
        for i in range(len(input)):
            sentence_index.append((i, input[i]))

        input_token, input_idx, input_seqs = tokenize_and_preserve_labels(self.tokenizer, input)

        input_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(txt) for txt in input_token])
        attention_masks = torch.tensor([float(i != 0.0) for i in input_ids])

        input_data = TensorDataset(input_ids.unsqueeze(0), attention_masks.unsqueeze(0), input_idx.unsqueeze(0))
        input_sampler = SequentialSampler(input_data)
        input_dataloader = DataLoader(input_data, sampler=input_sampler, batch_size=1, shuffle=False)

        self.model.eval()

        punct_predict, predict = [], []
        for step, batch in enumerate(input_dataloader):            
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_idx = batch
            with torch.no_grad():

                output = self.model(input_ids=b_input_ids, token_type_ids=None, attention_mask=b_input_mask, batch=batch)
                punct_output = self.punct_model(input_ids=b_input_ids, token_type_ids=None, attention_mask=b_input_mask, batch=batch)

            # label_indices = np.argmax(output[1].to('cpu').numpy(), axis=2)
            punct_label_indices = np.argmax(punct_output[1].to('cpu').numpy(), axis=2)
            seq_output = torch.index_select(output[1][0], 0, torch.tensor(input_seqs).cuda())
            seq_output = torch.index_select(seq_output, 1, torch.tensor(input_seqs).cuda())
            # print(seq_output)
            # with open('output.json', 'w') as f:
            #     json.dump(seq_output.to('cpu').numpy().tolist(), f)
            # input('1')
            label_indices = fast_parse(torch.transpose(seq_output, 0, 1).fill_diagonal_(-100000).to('cpu').numpy(), one_root=True)
            # seq_output = torch.transpose(seq_output, 0, 1).fill_diagonal_(-100000).to('cpu')
            # seq_output_a = seq_output.argmax(1)
            # seq_output_m = torch.zeros(seq_output.shape).scatter(1, seq_output_a.unsqueeze(1), 100000)
            # seq_output[0] = seq_output_m[0]
            # label_indices = fast_parse(seq_output.numpy(), one_root=True)
            
            final_predict = []
            for label_idx in label_indices:
                final_predict.append(int(b_idx[0][input_seqs[label_idx]].cpu()))
            final_predict[0] = -1

            punct_new_labels = []
            for label_idx in punct_label_indices[0]:
                punct_new_labels.append(self.ids_to_labels[label_idx])
            punct_predict.append(punct_new_labels)

        punct_final_predict = []
        for i in input_seqs:
            punct_final_predict.append(punct_predict[0][i])

        parse = []
        for i in range(len(input[1:])):
            parse.append((final_predict[1:][i], i+1, punct_final_predict[1:][i]))
  
        return parse, sentence_index

@st.cache(allow_output_mutation=True)
class chinese_parser():

    def __init__(self):
        
        self.conn = rpyc.classic.connect('localhost')
        self.conn.execute('from ckiptagger import data_utils, construct_dictionary, WS, POS, NER')
        self.conn.execute('ws = WS("./data")')
        self.ids_to_labels = {0: 'root', 1: 'nn', 2: 'conj', 3: 'cc', 4: 'nsubj', 5: 'dep', 6: 'punct', 7: 'lobj', 8: 'loc', 9: 'comod', 10: 'asp', 11: 'rcmod', 12: 'etc', 13: 'dobj', 14: 'cpm', 15: 'nummod', 16: 'clf', 17: 'assmod', 18: 'assm', 19: 'amod', 20: 'top', 21: 'attr', 22: 'advmod', 23: 'tmod', 24: 'neg', 25: 'prep', 26: 'pobj', 27: 'cop', 28: 'dvpmod', 29: 'dvpm', 30: 'lccomp', 31: 'plmod', 32: 'det', 33: 'pass', 34: 'ordmod', 35: 'pccomp', 36: 'range', 37: 'ccomp', 38: 'xsubj', 39: 'mmod', 40: 'prnmod', 41: 'rcomp', 42: 'vmod', 43: 'prtmod', 44: 'ba', 45: 'nsubjpass'}
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        self.model = BERT_Encoder1(hidden_size=1024, pretrained="hfl/chinese-roberta-wwm-ext-large").cuda()
        self.model.load_state_dict(torch.load(f'demo_model/bert_encoder.pt')) 

        self.punct_model = BERT_Punct(loaded_model='demo_model/bert_encoder.pt', hidden_size=1024, num_labels=len(self.ids_to_labels)).cuda()
        self.punct_model.load_state_dict(torch.load(f'demo_model/bert_punct_encoder.pt')) 

    def output(self, input_text):

        input_text = s2t.convert(input_text)
        input_sent = self.conn.eval(f'ws(["{input_text}"])')[0]
        input_sentence = []
        for i in input_sent:
            input_sentence.append(t2s.convert(i))
        input = ['root'] + list(input_sentence)
        parse_input = ['root'] + list(input_sent)
        sentence_index = []
        for i in range(len(parse_input)):
            sentence_index.append((i, parse_input[i]))

        input_token, input_idx, input_seqs = tokenize_and_preserve_labels(self.tokenizer, input)

        input_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(txt) for txt in input_token])
        attention_masks = torch.tensor([float(i != 0.0) for i in input_ids])

        input_data = TensorDataset(input_ids.unsqueeze(0), attention_masks.unsqueeze(0), input_idx.unsqueeze(0))
        input_sampler = SequentialSampler(input_data)
        input_dataloader = DataLoader(input_data, sampler=input_sampler, batch_size=1, shuffle=False)

        self.model.eval()

        punct_predict, predict = [], []
        for step, batch in enumerate(input_dataloader):            
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_idx = batch
            with torch.no_grad():

                output = self.model(input_ids=b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
                punct_output = self.punct_model(input_ids=b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

            # label_indices = np.argmax(output[1].to('cpu').numpy(), axis=2)
            # label_indices = fast_parse(torch.transpose(output[1], 1, 2)[0].fill_diagonal_(-100000).to('cpu').numpy(), one_root=True)
            punct_label_indices = np.argmax(punct_output[1].to('cpu').numpy(), axis=2)
            seq_output = torch.index_select(output[1][0], 0, torch.tensor(input_seqs).cuda())
            seq_output = torch.index_select(seq_output, 1, torch.tensor(input_seqs).cuda())
            label_indices = fast_parse(torch.transpose(seq_output, 0, 1).fill_diagonal_(-100000).to('cpu').numpy(), one_root=True)
          
            final_predict = []
            for label_idx in label_indices:
                final_predict.append(int(b_idx[0][input_seqs[label_idx]].cpu()))
            final_predict[0] = -1

            punct_new_labels = []
            for label_idx in punct_label_indices[0]:
                punct_new_labels.append(self.ids_to_labels[label_idx])
            punct_predict.append(punct_new_labels)

        punct_final_predict = []
        for i in input_seqs:
            punct_final_predict.append(punct_predict[0][i])

        parse = []
        for i in range(len(input[1:])):
            parse.append((final_predict[1:][i], i+1, punct_final_predict[1:][i]))
  
        return parse, sentence_index

if __name__ == '__main__':

    st.set_page_config(page_title="Dependency Parser")
    st.title('Dependency Parser')
    st.write("")

    option = st.selectbox('Select Language:', ('English', 'Chinese'))

    if option == 'English':

        form = st.form(key='my_form')
        inp_text = form.text_input(label='Please submit sentence ...')
        submit_button = form.form_submit_button(label='Submit')

        if inp_text == '':
            input_text = 'How are you?' # 'Not all those who wrote appose the changes.'
        else:
            input_text = inp_text

        st.write("")
        st.write("Dependency Parse Tree: ")

        with st.spinner('In progress ...'):
        
            output, indexx = english_parser().output(input_text)
            g=nx.DiGraph()
            for ind in indexx:
                g.add_node(ind[0], label=f'<{ind[1]}>')
            for tok_ind in output:
                g.add_edge(tok_ind[0], tok_ind[1], label=f' {tok_ind[2]} ')
            p=nx.drawing.nx_pydot.to_pydot(g)
            st.graphviz_chart(f"{p}", use_container_width=True)

    else:

        form = st.form(key='my_form')
        inp_text = form.text_input(label='請輸入句子 ...')
        submit_button = form.form_submit_button(label='確定')

        if inp_text == '':
            input_text = '你好嗎？'
        else:
            input_text = inp_text

        st.write("")
        st.write("剖析樹: ")

        with st.spinner('In progress ...'):

            output, indexx = chinese_parser().output(input_text)
            g=nx.DiGraph()
            for ind in indexx:
                g.add_node(ind[0], label=f'<{ind[1]}>')
            for tok_ind in output:
                g.add_edge(tok_ind[0], tok_ind[1], label=f' {tok_ind[2]} ')
            p=nx.drawing.nx_pydot.to_pydot(g)
            st.graphviz_chart(f"{p}", use_container_width=True)
