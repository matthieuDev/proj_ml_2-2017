import numpy as np
import pickle

from scipy.sparse import rand
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.kernel_ridge import KernelRidge
from scipy.sparse import coo_matrix

import implementations as imp
import proj1_helpers as ph
import glove as gl

from nltk import pos_tag
#nltk.download()
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

tweets_train = []
smile = []
tweet_train_pos = []
tweet_train_neg = []
print(len(tweets_train))
with open('twitter-datasets/train_neg_full.txt', encoding='UTF-8') as f :
    tweet_train_neg = list(set(f.readlines()))
    
print(len(tweets_train))
with open('twitter-datasets/train_pos_full.txt', encoding='UTF-8') as f :
    tweet_train_pos =list(set(f.readlines()))
	
print(len(tweet_train_pos)+len(tweet_train_neg))

set_neg = set(tweet_train_neg)
duplicate = [ x for x in tweet_train_pos if x in set_neg]

tweet_train_pos = [ x for x in tweet_train_pos if x not in duplicate]
tweet_train_neg = [ x for x in tweet_train_neg if x not in duplicate]

tweets_train = np.append(tweet_train_pos,tweet_train_neg)
smile = np.append([1]*len(tweet_train_pos),[-1]*len(tweet_train_neg))

print(len(tweet_train_pos)+len(tweet_train_neg))

stemmer = PorterStemmer()
lemmatiser = WordNetLemmatizer()

def ke_free_lemm (word , type_) :
    try :
        res = lemmatiser.lemmatize(word , type_)
        print(res,word ,type_)
        return 
        
    except KeyError :
        return word

def lemmatize (tweet):
    
    return [ lemmatiser.lemmatize(x) for x in tweet ]

def split_ok (tweet) :
    res = tweet[:-1].split(' ')
    res = [ x for x in res if x != '']
    return res
	
	
print ( 'begin')
tweets_train_nlp = [ [ ke_free_lemm(word[0],word[1]) for word in pos_tag(split_ok (x))] \
                    for x in tweets_train\
                    ]
#tweets_train_nlp = [  pos_tag(x[:-2].split(' ')) for x in tweets_train [:]  ]
print(len(tweets_train_nlp))

with open('tweets_nlp_lem', 'wb') as fp:
    pickle.dump(tweets_train_nlp, fp)
