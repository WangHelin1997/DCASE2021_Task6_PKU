import re
import pickle
import numpy as np 
import pandas as pd
from cider import CiderScorer
words_list = pickle.load(open("../create_dataset/data/pickles/words_list.p","rb"))

def get_doc_freq(refs):
    tmp = CiderScorer()
    tmp.cook_append(None, refs)
    tmp.compute_doc_freq()
    return tmp.document_frequency, len(tmp.crefs)
def bulit_dict(captions_data):
    refs_words = []
    refs_idxs = []
    p = re.compile(r"[!?',;:.]")
    for data in captions_data:
        data = data[1:]
        for text in data:
            text = p.sub(' ',text.lower())
            text = text.split()
            refs_words.append(' '.join(text))
            refs_idxs.append(' '.join([str(words_list.index(_)) for _ in text]))

    ngram_words, count_refs = get_doc_freq(refs_words)
    ngram_idxs, count_refs = get_doc_freq(refs_idxs)
    with open("output_pkl_words.p","wb") as f:
        pickle.dump({'document_frequency': ngram_words, 'ref_len': count_refs}, f)
    with open("output_pkl_inds.p","wb") as f:
        pickle.dump({'document_frequency': ngram_idxs, 'ref_len': count_refs}, f)

def main(split):
    train_csv = "../create_dataset/data/clotho_csv_files/clotho_captions_development.csv"
    test_csv = "../create_dataset/data/clotho_csv_files/clotho_captions_evaluation.csv"
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)
    data1 = df_train.values
    data2 = df_test.values
    if split == "train":
        data = data1
    elif split == "val":
        data = data2
    elif split == "all":
        data = np.concatenate((data1,data2))
    bulit_dict(data)

if __name__ == "__main__":
    main("train")
