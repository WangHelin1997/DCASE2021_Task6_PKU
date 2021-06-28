import csv
import pickle
from captions_functions import clean_sentence,get_words_counter,get_sentence_words
import copy
import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
# with open('word_list_pretrain_rules.p', 'rb') as f:
#     testtag = pickle.load(f)
# _test contains background people person 
with open('word_list_pretrain_rules.p', 'rb') as f:
    testtag = pickle.load(f)
def c_wordList(x):
    WordList = copy.copy(x)
    for w in x:
        if w[-3:] == 'ing' and w[:-3] in testtag: #playing
            WordList.remove(w)
            WordList.add(w[:-3])
        elif w[-3:] == 'ing' and w[:-4] in testtag: #running
            WordList.remove(w)
            WordList.add(w[:-4])
        elif w[-2:] == 'ly' and w[:-2] in testtag: # slowly
            WordList.remove(w)
            WordList.add(w[:-2])
        elif w[-2:] == 'ly' and w[:-3] in testtag: # lly
            WordList.remove(w)
            WordList.add(w[:-3])
        elif w[-1] == 's' and w[:-1] in testtag:
            WordList.remove(w)
            WordList.add(w[:-1])
        elif w[-2:] == 'es' and w[:-2] in testtag:
            WordList.remove(w)
            WordList.add(w[:-2])
        elif w[-1] == 'd' and w[:-1] in testtag:
            WordList.remove(w)
            WordList.add(w[:-1])
        elif w[-2:] == 'ed' and w[:-2] in testtag:
            WordList.remove(w)
            WordList.add(w[:-2])
        elif w[-3:] == 'ing' and w[:-3]+'e' in testtag:
            WordList.remove(w)
            WordList.add(w[:-3]+'e')
    return WordList

def gen_tag(split):
    audioAttributes = [] 
    allWordList = []
    all_count = 0
    with open('../create_dataset/data/clotho_csv_files/clotho_captions_{}.csv'.format(split), 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        file_names = [row['file_name'] for row in reader]
    with open('../create_dataset/data/clotho_csv_files/clotho_captions_{}.csv'.format(split), 'r') as f:
        reader = csv.reader(f)
        count = 0
        for row in reader:
            curAttributes = []  
            count += 1  
            if count == 1:
                continue
            curCaptionList = row[1:]  
            curWordList = []
            for caption in curCaptionList:
                curWordList.extend(get_sentence_words(caption))
            curWordList = c_wordList(set(curWordList))
            allWordList.append(list(curWordList))

    allAttributes = {}
    for file_name, audioList in zip(file_names, allWordList):
        
        curAttributes = []
        for attribute in testtag:
            if attribute in audioList:
                curAttributes.append(attribute)
        allAttributes[file_name] = curAttributes

    allAttributesNum = {}
    for file_name, audioList in zip(file_names, allWordList):
        curAttributes = [0 for s in range(len(testtag))]
        count =0
        for i, attribute in enumerate(testtag):
            if attribute in audioList:
                curAttributes[i] = 1
                count+=1
        
        if count <= 2 :
            all_count+=1
            print("there is no word in ",file_name)
        allAttributesNum[file_name] = curAttributes

    with open('audioTagNum_{}_fin_nv.pickle'.format(split), 'wb') as f:
        pickle.dump(allAttributesNum, f)

    with open('audioTagName_{}_fin_nv.pickle'.format(split), 'wb') as f:
        pickle.dump(allAttributes, f)
    print(all_count)
    # with open('Tag_fin.pickle','wb') as f:
    #     pickle.dump(curWordList,f)
if __name__ == '__main__':
    gen_tag('development')
    gen_tag('evaluation')
    gen_tag('validation')

