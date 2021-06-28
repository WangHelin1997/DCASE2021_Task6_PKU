# from bert_embedding import BertEmbedding
# from bert_embedding import BertEmbedding
# import numpy as np
# bert_abstract = """We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers.
#  Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations by jointly conditioning on both left and right context in all layers.
#  As a result, the pre-trained BERT representations can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications. 
# BERT is conceptually simple and empirically powerful. 
# It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE benchmark to 80.4% (7.6% absolute improvement), MultiNLI accuracy to 86.7 (5.6% absolute improvement) and the SQuAD v1.1 question answering Test F1 to 93.2 (1.5% absolute improvement), outperforming human performance by 2.0%."""
 

# bert_embedding = BertEmbedding()
# result1 = bert_embedding(bert_abstract)
# print(len(result1))

# import torch
# import torchtext.vocab as vocab

# cache_dir = "./pretrainmodel/glove"
# # glove = vocab.pretrained_aliases["glove.6B.50d"](cache=cache_dir)
# glove = vocab.GloVe(name='6B', dim=300, cache=cache_dir) # 与上面等价
# print(type(glove))
# print(glove.vectors[3336],type(glove.vectors[3336]))


import torch
import torchtext.vocab as vocab
import pickle
cache_dir = "./pretrainmodel/glove"
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
# glove = vocab.pretrained_aliases["glove.6B.50d"](cache=cache_dir)
glove = vocab.GloVe(name='840B', dim=300, cache=cache_dir) # 与上面等价
with open('./create_dataset/data/pickles/words_list.p','rb') as f:
    dict_words = pickle.load(f)
pre_vec = {}
glove_keys = glove.stoi.keys()
for k in dict_words:
    if k =='<eos>' :
        k_ = 'eos'
        pre_vec[k] = glove.vectors[glove.stoi[k_]]
    elif k =='<sos>':
        k_ = 'sos'
        pre_vec[k] = glove.vectors[glove.stoi[k_]]
    elif k == 'ribbiting':
        k_ = 'ribbit'
        pre_vec[k] = glove.vectors[glove.stoi[k_]]
    elif k not in glove_keys:
        verb = lemmatizer.lemmatize(k, pos='v')
        noun = lemmatizer.lemmatize(k, pos='n')
        if verb in glove_keys :
            pre_vec[k] = glove.vectors[glove.stoi[verb]]
        elif noun in glove_keys:
            pre_vec[k] = glove.vectors[glove.stoi[noun]]
        else:
            print(k)
    else:
        pre_vec[k] = glove.vectors[glove.stoi[k]]
pre_vec['<pad>'] = glove.vectors[glove.stoi['pad']]

with open('./create_dataset/data/pickles/words_list_glove.p', 'wb') as f:
    pickle.dump(pre_vec, f)