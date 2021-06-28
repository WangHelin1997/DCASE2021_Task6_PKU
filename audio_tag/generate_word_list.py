import pickle 
import numpy as np 
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer

from collections import Counter
from ast import literal_eval
stop_words = ['time','times','object', 'make', 'play', 'distance', 'go', 'something', 'turn', 'get', 'come', 'follow', 'use',
 'piece', 'have','approach', 'rate', 'take', 'begin', 'end', 'forth', 'slow', 'continue', 'lot', 'work', 'place',
 'create', 'night', 'set', 'increase', 'page', 'others', 'put', 'item', 'way', 'variety', 'passing', 'do', 'repeat',
 'occur', 'leave', 'try', 'speaks', 'side', 'past', 'accelerate', 'overhead', 'path', 'thing', 'signal', 'become',
 'kind', 'pattern', 'day', 'cause', 'frequency', 'couple', 'second', 'closing', 'note', 'interval',
 'vary', 'proximity', 'number', 'cover', 'system', 'effect','start','go','someone',
 'back','become','change','come','do',"amount",'has','other','top','background','person','people']

def dict_add(word_freq_dict, w, f):
    if w in word_freq_dict.keys():
        word_freq_dict[w] += f
    else:
        word_freq_dict[w] = f
    return word_freq_dict
word_list = pickle.load(open("../create_dataset/data/pickles/words_list.p","rb"))
train_csv = "../create_dataset/data/clotho_csv_files/clotho_captions_development.csv"
test_csv = "../create_dataset/data/clotho_csv_files/clotho_captions_evaluation.csv"
val_csv = "../create_dataset/data/clotho_csv_files/clotho_captions_validation.csv"
df_train = pd.read_csv(train_csv)
df_test = pd.read_csv(test_csv)
df_val = pd.read_csv(val_csv)
data1 = df_train.values
data2 = df_test.values
data3 = df_val.values
alldata = np.concatenate((data1,data2,data3))
lemmatizer = WordNetLemmatizer()
all_keywords = Counter()
count = 0
for data in alldata:
    data = data[1:]
    # keywords = set()
    a_keywords = []

    for text in data:
        text=nltk.word_tokenize(text)
        text_list = nltk.pos_tag(text)
        # print(text_list)
        keywords = []
        for word, tag in text_list:
            word = word.lower()
            if tag.startswith("VB"):
                w = lemmatizer.lemmatize(word, pos='v')
                if w !='be':
                    keywords.append(word)
            elif tag.startswith("NN"):
                w = lemmatizer.lemmatize(word, pos='n')
                keywords.append(word)
            else:
                continue
        tmp = Counter(keywords)
        all_keywords += tmp
    print(count)
    count += 1

word_freq_dict = {}
selected_words = all_keywords.most_common()
for w, f in selected_words:
    if w[-3:] == 'ing' and w[:-3] in word_list:  # playing
        word_freq_dict = dict_add(word_freq_dict, w[:-3], f)
    elif w[-3:] == 'ing' and w[:-4] in word_list:  # running
        word_freq_dict = dict_add(word_freq_dict, w[:-4], f)
    elif w[-2:] == 'ly' and w[:-2] in word_list:  # slowly
        word_freq_dict = dict_add(word_freq_dict, w[:-2], f)
    elif w[-2:] == 'ly' and w[:-3] in word_list:  # lly
        word_freq_dict = dict_add(word_freq_dict, w[:-3], f)
    elif w[-1] == 's' and w[:-1] in word_list:
        word_freq_dict = dict_add(word_freq_dict, w[:-1], f)
    elif w[-2:] == 'es' and w[:-2] in word_list:
        word_freq_dict = dict_add(word_freq_dict, w[:-2], f)
    elif w[-1] == 'd' and w[:-1] in word_list:
        word_freq_dict = dict_add(word_freq_dict, w[:-1], f)
    elif w[-2:] == 'ed' and w[:-2] in word_list:
        word_freq_dict = dict_add(word_freq_dict, w[:-2], f)
    elif w[-3:] == 'ing' and w[:-3]+'e' in word_list:
        word_freq_dict = dict_add(word_freq_dict, w[:-3]+'e', f)
    else:
        word_freq_dict = dict_add(word_freq_dict, w, f)
word_freq = sorted([(w, f) for w, f in word_freq_dict.items()], key=lambda x: x[1], reverse=True)
word_freq = [(w, f) for w, f in word_freq if len(w) > 2]
word_list_f = []
for w, f in word_freq:
    if w not in stop_words:
        word_list_f.append((w,f))
    else:
        print(w)

word_list_final = [w for w, f in word_list_f][:300]

# TaggingToembs
Tag2emb = []
for w in word_list_final:
    Tag2emb.append(word_list.index(w))
print(word_list_final)
pickle.dump(word_list_final, open('word_list_pretrain_rules.p', 'wb'))
pickle.dump(Tag2emb, open('TaggingToEmbs.p', 'wb'))