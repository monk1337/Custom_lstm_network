from tqdm import tqdm as tqdm_notebook
from sklearn import preprocessing 
import numpy as np
import ast
import pandas as pd
import spacy
import operator
import pickle as pk
import spacy
nlp = spacy.load('en')


read_csv = pd.read_csv('./Data/tweet2.csv')
dataset = []
for m in range(len(read_csv)):
    dataset.append(read_csv.iloc[m]['cleaned_tweet_decrypt_emojis'])




def sentiment_vector_generation():
    
    read_csv = pd.read_csv('./Data/tweet2.csv') #reading tweets file

    def sentiment_dict():
        sentiment_dict = {}

        #reading_first_file
        with open('./Data/NRC-Hashtag-Emotion-Lexicon-v0.2.txt','r') as f:
            for line in tqdm_notebook(f):
                data = line.split()
                if data[1] not in sentiment_dict:
                    sentiment_dict[data[1].lower()]=[float(data[2])]
                else:
                    sentiment_dict[data[1].lower()].append(float(data[2]))

        with open('./Data/NRC-VAD-Lexicon.txt','r') as f:
            i=0
            for line in tqdm_notebook(f):
                data = line.split()
                if i>=1:
                    try:
                        if data[0].lower() not in sentiment_dict:
                            sentiment_dict[data[0].lower()]= [float(data[1])]
                        else:
                            sentiment_dict[data[0].lower()].append(float(data[1]))
                    except Exception:
                        sentiment_dict[data[0].lower()]= [float(data[2])]

                i+=1

        #third_sentiment_file
        with open('./Data/SemEval2015-English-Twitter-Lexicon.txt','r') as f:
            for line in f:
                data = line.split()
                if data[1] not in sentiment_dict:
                    sentiment_dict[data[1].lower()] = [float(data[0])]
                else:
                    sentiment_dict[data[1].lower()].append(float(data[0]))

        #forth_sentinet
        with open('./Data/senticnet5.txt','r') as f:
            i=0
            for line in f:
                if i>=1:
                    data = line.split()
                    if data[0] not in sentiment_dict:
                        sentiment_dict[data[0].lower()] = [float(data[2])]
                    else:
                        sentiment_dict[data[0].lower()].append(float(data[2]))
                i+=1

        return sentiment_dict
    
    
    

    def create_vocab():
        sentiment = []
        tweets = []
        vocab  = []
        vocabs = {}

        for m in range(len(read_csv)):
            try:
                for g in read_csv.iloc[m]['cleaned_tweet_decrypt_emojis'].split():
                    if g not in vocab:
                        vocab.append(g)
                        vocabs[g] = 1
                    else:
                        vocabs[g] +=1
            except Exception:
                pass

        sorted_longs = sorted(vocabs.items(), key=operator.itemgetter(1),reverse=True)
        sotted_list = []

        for m in sorted_longs:
            if m[1]>=2:
                sotted_list.append(m[0])
        with open('word_vocab_.pkl','wb') as f:
            pk.dump(sotted_list,f)

        return sotted_list
    
    def sentiment_vector_generation_():
        final_vector = { 'words' : []  ,  'sentiment_vector' : [] }

        sentiment_vocab_ = create_vocab()
        sentiment_dict_  = sentiment_dict()

        for m in sentiment_vocab_ :
            if m in sentiment_dict_:
                final_vector['words'].append(m)
                final_vector['sentiment_vector'].append(sentiment_dict_[m])
            else:
                final_vector['words'].append(m)
                final_vector['sentiment_vector'].append(np.zeros(6))

        final_vector_s = { 'words' : []  ,  'sentiment_vector' : [] }

        for m in zip(*(final_vector['words'],final_vector['sentiment_vector'])):
            if len(m[1])>=6:
                final_vector_s['words'].append(m[0])
                final_vector_s['sentiment_vector'].append(preprocessing.normalize([m[1]],norm='l2')[0][:6].tolist())
            else:
                act = len(m[1])
                hw_many = 6-len(m[1])
                append_list = [0]*hw_many
                m[1].extend(append_list)
                normalize = preprocessing.normalize([m[1]],norm='l2')[0]
                final_vector_s['words'].append(m[0])
                final_vector_s['sentiment_vector'].append(normalize.tolist())
                
        save_file = pd.DataFrame(final_vector_s).to_csv('final_sentiment_vector.csv')
        read_file_ss = pd.read_csv('final_sentiment_vector.csv')
        
        dict_ = {}
        sentiment_matrix_ = []
        
                
        dict_['UNK'] = [ 0.71481943,  0.98741437,  0.85255514,  0.83739983, -0.53904541, 0.87657205]
        
        for m in range(len(read_file_ss)):
            dict_ [read_file_ss.iloc[m]['words']] = [float(m) for m in ast.literal_eval(read_file_ss.iloc[m]['sentiment_vector'])]
            sentiment_matrix_.append([float(m) for m in ast.literal_eval(read_file_ss.iloc[m]['sentiment_vector'])])

            
        np.save('senti_matrix.npy',sentiment_matrix_)

        return save_file
    
    return sentiment_vector_generation_()





def word_embedding_matrix(embedding_path,dim):
    
    #first and second vector are pad and unk words
    
    with open('word_vocab_.pkl','rb') as f:
        vocab = pk.load(f)
        
    
    with open(embedding_path,'r') as f:
        word_vocab =[]
        embedding_matrix = []
        word_vocab.extend(['PAD','UNK'])
        embedding_matrix.append(np.random.uniform(-1.0, 1.0, (1,dim))[0])
        embedding_matrix.append(np.random.uniform(-1.0, 1.0, (1,dim))[0])

        
        for line in f:
            if line.split()[0] in vocab:
                word_vocab.append(line.split()[0])
                embedding_matrix.append([float(i) for i in line.split()[1:]])
                
        np.save('word_embedding.npy',np.reshape(embedding_matrix,[-1,dim]).astype(np.float32))
        
        int_to_vocab = {}
        symbols = {0: 'PAD',1: 'UNK'}


        for index_no,word in enumerate(word_vocab):
            int_to_vocab[index_no] = word
            
        int_to_vocab.update(symbols)

        
        vocab_to_int = {word:index_no for index_no , word in int_to_vocab.items()}
        
        
        with open('int_to_vocab.pkl','wb') as f:
            pk.dump(int_to_vocab,f)

        with open('vocab_to_int.pkl','wb') as f:
            pk.dump(vocab_to_int,f)
            

        with open('word_vocab.pkl','wb') as f:
            pk.dump(word_vocab,f)

def pos_embedding_generation():

        
    nlp = spacy.load('en')

    
    def spacy_PoS(sentence):
        return [ w.pos_ for w in nlp(sentence) ]
    
    pos_vocab = []
    for m in tqdm_notebook(dataset):
        try:
            pos_vocab.extend(spacy_PoS(m))
            
        except Exception as e:
            pass
        
    
    
    with open('pos_vocab.pkl','wb') as f:
        pk.dump(pos_vocab,f)
        
    Pos_matrix = []
    
    for m in range(16):
        Pos_matrix.append(np.random.uniform(-1.0, 1.0, (1,50))[0])
        
    np.save('pos_matrix.npy',Pos_matrix)
        
    return pos_vocab

sentim = sentiment_vector_generation()
word_embed = word_embedding_matrix('./Data/glove.6B.300d.txt',300)
pos_embd= pos_embedding_generation()


raw_data = pd.read_csv('./Data/tweet2.csv')
nlp = spacy.load('en')


def spacy_PoS(sentence):
    return [ w.pos_ for w in nlp(sentence) ]

with open('word_vocab.pkl','rb') as f:
    word_vocab = pk.load(f)



word_to_int = {n: m  for m,n in enumerate(word_vocab)}
int_to_word = {m:n  for m,n in enumerate(word_vocab)}


with open('word_vocab.pkl','rb') as f:
    sentiment_vocab = pk.load(f)

senti_to_int = {n: m  for m,n in enumerate(sentiment_vocab)}
int_to_senti = {m:n  for m,n in enumerate(sentiment_vocab)}



with open('pos_vocab.pkl','rb') as f:
    pos_vocab = pk.load(f)

final_tags = list(set(pos_vocab))

pos_to_int = {n: m  for m,n in enumerate(final_tags)}
int_to_pos = {m:n  for m,n in enumerate(final_tags)}


labels_vocab =  {'positive':0  ,'negative':1 , 'neutral' : 2 }
reverse_vocab = {0: 'positive' ,1:'negative' , 2: 'neutral' }


#all_embedding_matrix

word_embedding_matrix  = np.load('word_embedding.npy')  #300 dim

pos_embedding_matrix   = np.load('pos_matrix.npy')      # 50 dim

senti_embedding_matrix = np.load('senti_matrix.npy')    #6 dim




word_embedding_encoded = []
pos_encoded_           = []
sentiment_encodeds      = []
labels_encoded_        = []

def padd_local(seq_,le_n):
    if len(seq_)<=35:
        max_count = 35 - len(seq_) 
        pad_count = [0] * max_count
        return seq_ + pad_count
    else:
        seq_ = seq_[:35]
        return seq_

for m in tqdm_notebook(range(len(raw_data))):

    try:
        tokensize_normal    = raw_data.iloc[m]['cleaned_tweet_decrypt_emojis'].split()
        pos_tags            = spacy_PoS(raw_data.iloc[m]['cleaned_tweet_decrypt_emojis'])
        sentiment_encoded   = raw_data.iloc[m]['cleaned_tweet_decrypt_emojis'].split()
        labels_encoded_.append(labels_vocab[raw_data.iloc[m]['airline_sentiment']])


        new_sen =[]
        new_pos =[]
        new_seni=[]
        for m in tokensize_normal:
            if m.lower() in word_vocab:
                new_sen.append(word_to_int[m.lower()])
            else:
                new_sen.append(word_to_int['UNK'])

        for k in pos_tags:
            new_pos.append(pos_to_int[k])


        for j in sentiment_encoded:
            if j in sentiment_vocab:
                new_seni.append(senti_to_int[j])
            else:
                new_seni.append(senti_to_int['UNK'])

        word_embedding_encoded.append(new_sen)
        pos_encoded_.append(new_pos)
        sentiment_encodeds.append(new_seni)
    except Exception:
        pass

#         print(pos_to_int)

word_embedd_lookup = []
pos_embedd_lookup  =[]
senti_embedd_lookup =[]
Improved_vector = []
Labels_encoded  = []

for word_ in tqdm_notebook(zip(word_embedding_encoded,pos_encoded_,sentiment_encodeds,labels_encoded_)):
    max_len = max([len(word_[0]),len(word_[1]),len(word_[2]),word_[3]])

    word_vector = padd_local(word_[0],max_len)
    pos_vector  = padd_local(word_[1],max_len)
    senti_vector = padd_local(word_[2],max_len)
    
    
    
    labels       = word_[3]

    word_look =[]
    pos_look  =[]
    senti_look =[]


    for local_word in word_vector:
        word_look.append(word_embedding_matrix[local_word])
    for local_pos in pos_vector:
        pos_look.append(pos_embedding_matrix[local_pos])
    for local_senti in senti_vector:
        senti_look.append(senti_embedding_matrix[local_senti])

    Improved_vector.append(np.column_stack((word_look,pos_look,senti_look)))
    
    Labels_encoded.append(labels)

#     print(np.column_stack((word_look,pos_look,senti_look)).shape)
with open('Improved_vector_pad.pkl','wb') as f:
    pk.dump(Improved_vector,f)
    
#     print(np.column_stack((word_look,pos_look,senti_look)).shape)
with open('labels_pad.pkl','wb') as f:
    pk.dump(Labels_encoded,f)