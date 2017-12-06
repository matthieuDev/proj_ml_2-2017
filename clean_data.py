import numpy as np
import pickle

tweets_train = []
smile = []
tweet_train_pos = []
tweet_train_neg = []

with open('twitter-datasets/train_neg_full.txt', encoding='UTF-8') as f :
    tweet_train_neg = list(set(f.readlines()))
    
print(len(tweets_train))
with open('twitter-datasets/train_pos_full.txt', encoding='UTF-8') as f :
    tweet_train_pos =list(set(f.readlines()))
	


set_neg = set(tweet_train_neg)
duplicate = [ x for x in tweet_train_pos if x in set_neg]

tweet_train_pos = [ x for x in tweet_train_pos if x not in duplicate]
tweet_train_neg = [ x for x in tweet_train_neg if x not in duplicate]

tweets_train = np.append(tweet_train_pos,tweet_train_neg)
smile = np.append([1]*len(tweet_train_pos),[-1]*len(tweet_train_neg))

with open('twitter-datasets/train_neg_full_clean.pkl', 'wb') as f:
	pickle.dump(tweet_train_neg, f)

with open('twitter-datasets/train_pos_full_clean.pkl', 'wb') as f:
	pickle.dump(tweet_train_pos, f)		

with open('twitter-datasets/full_train_clean.pkl', 'wb') as f:
	pickle.dump(tweets_train, f)	


with open('twitter-datasets/train_smile.pkl', 'wb') as f:
	pickle.dump(smile, f)

