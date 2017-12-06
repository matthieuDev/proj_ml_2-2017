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

tweets_train = pickle.load(open('twitter-datasets/full_train_clean.pkl', 'rb'))

smile = pickle.load(open('twitter-datasets/train_smile.pkl', 'rb')) 

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
tweets_train_nlp = [ [ stemmer.stem(ke_free_lemm(word[0],word[1])) for word in pos_tag(split_ok (x))] \
                    for x in tweets_train\
                    ]
#tweets_train_nlp = [  pos_tag(x[:-2].split(' ')) for x in tweets_train [:]  ]
print(len(tweets_train_nlp))

with open('tweets_nlp', 'wb') as fp:
    pickle.dump(tweets_train_nlp, fp)
