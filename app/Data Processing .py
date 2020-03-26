
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings("ignore")
import ftfy
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import re
import time

from math import exp
from numpy import sign

from sklearn.metrics import  classification_report, confusion_matrix, accuracy_score
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from nltk import PorterStemmer

from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv1D, Dense, Input, LSTM, Embedding, Dropout, Activation, MaxPooling1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# In[69]:
EMBEDDING_FILE = 'GoogleNews-vectors-negative300.bin.gz'
print("Loading embedding file..")
word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)

while(True):
    df = pd.read_csv("twitter2.csv")
    df = df[['tweet', 'location']]
    df


    # In[70]:


    df = df.dropna()
    df


    # In[118]:


    import re

    # Expand Contraction
    cList = {
      "ain't": "am not",
      "aren't": "are not",
      "can't": "cannot",
      "can't've": "cannot have",
      "'cause": "because",
      "could've": "could have",
      "couldn't": "could not",
      "couldn't've": "could not have",
      "didn't": "did not",
      "doesn't": "does not",
      "don't": "do not",
      "hadn't": "had not",
      "hadn't've": "had not have",
      "hasn't": "has not",
      "haven't": "have not",
      "he'd": "he would",
      "he'd've": "he would have",
      "he'll": "he will",
      "he'll've": "he will have",
      "he's": "he is",
      "how'd": "how did",
      "how'd'y": "how do you",
      "how'll": "how will",
      "how's": "how is",
      "I'd": "I would",
      "I'd've": "I would have",
      "I'll": "I will",
      "I'll've": "I will have",
      "I'm": "I am",
      "I've": "I have",
      "isn't": "is not",
      "it'd": "it had",
      "it'd've": "it would have",
      "it'll": "it will",
      "it'll've": "it will have",
      "it's": "it is",
      "let's": "let us",
      "ma'am": "madam",
      "mayn't": "may not",
      "might've": "might have",
      "mightn't": "might not",
      "mightn't've": "might not have",
      "must've": "must have",
      "mustn't": "must not",
      "mustn't've": "must not have",
      "needn't": "need not",
      "needn't've": "need not have",
      "o'clock": "of the clock",
      "oughtn't": "ought not",
      "oughtn't've": "ought not have",
      "shan't": "shall not",
      "sha'n't": "shall not",
      "shan't've": "shall not have",
      "she'd": "she would",
      "she'd've": "she would have",
      "she'll": "she will",
      "she'll've": "she will have",
      "she's": "she is",
      "should've": "should have",
      "shouldn't": "should not",
      "shouldn't've": "should not have",
      "so've": "so have",
      "so's": "so is",
      "that'd": "that would",
      "that'd've": "that would have",
      "that's": "that is",
      "there'd": "there had",
      "there'd've": "there would have",
      "there's": "there is",
      "they'd": "they would",
      "they'd've": "they would have",
      "they'll": "they will",
      "they'll've": "they will have",
      "they're": "they are",
      "they've": "they have",
      "to've": "to have",
      "wasn't": "was not",
      "we'd": "we had",
      "we'd've": "we would have",
      "we'll": "we will",
      "we'll've": "we will have",
      "we're": "we are",
      "we've": "we have",
      "weren't": "were not",
      "what'll": "what will",
      "what'll've": "what will have",
      "what're": "what are",
      "what's": "what is",
      "what've": "what have",
      "when's": "when is",
      "when've": "when have",
      "where'd": "where did",
      "where's": "where is",
      "where've": "where have",
      "who'll": "who will",
      "who'll've": "who will have",
      "who's": "who is",
      "who've": "who have",
      "why's": "why is",
      "why've": "why have",
      "will've": "will have",
      "won't": "will not",
      "won't've": "will not have",
      "would've": "would have",
      "wouldn't": "would not",
      "wouldn't've": "would not have",
      "y'all": "you all",
      "y'alls": "you alls",
      "y'all'd": "you all would",
      "y'all'd've": "you all would have",
      "y'all're": "you all are",
      "y'all've": "you all have",
      "you'd": "you had",
      "you'd've": "you would have",
      "you'll": "you you will",
      "you'll've": "you you will have",
      "you're": "you are",
      "you've": "you have"
    }

    c_re = re.compile('(%s)' % '|'.join(cList.keys()))

    def expandContractions(text, c_re=c_re):
        def replace(match):
            return cList[match.group(0)]
        return c_re.sub(replace, text)

    def clean_tweets(tweets, df):
        cleaned_tweets = []
        i = 0
        for tweet in tweets:
            new_tweet = str(tweet)
            # if url links then dont append to avoid news articles
            # also check tweet length, save those > 10 (length of word "depression")
            if re.match("(\w+:\/\/\S+)", tweet) == None and len(tweet) > 10:
                #remove hashtag, @mention, emoji and image URLs
                new_tweet = ' '.join(re.sub("(@[A-Za-z0-9]+)|(\#[A-Za-z0-9]+)|(<Emoji:.*>)|(pic\.twitter\.com\/.*)", " ", tweet).split())
                
                #fix weirdly encoded texts
                new_tweet = ftfy.fix_text(new_tweet)
                
                #expand contraction
                new_tweet = expandContractions(new_tweet)

                #remove punctuation
                new_tweet = ' '.join(re.sub("([^0-9A-Za-z \t])", " ", new_tweet).split())

                #stop words
                stop_words = set(stopwords.words('english'))
                word_tokens = nltk.word_tokenize(new_tweet) 
                filtered_sentence = [w for w in word_tokens if not w in stop_words]
                new_tweet = ' '.join(filtered_sentence)

                #stemming words
                new_tweet = PorterStemmer().stem(new_tweet)         
                cleaned_tweets.append(new_tweet)
                i += 1
            else:
                df = df.drop(df.index[i])
            
        return cleaned_tweets, df


    # In[120]:


    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')

    df_arr = [x for x in df["tweet"]]
    X, df = clean_tweets(df_arr, df)
    print(len(X))
    df


    # In[73]:


    MAX_NB_WORDS = 20000
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(X)


    # In[74]:


    sequence = tokenizer.texts_to_sequences(X)


    # In[75]:


    word_index = tokenizer.word_index
    print('Found %s unique tokens' % len(word_index))


    # In[76]:


    MAX_SEQUENCE_LENGTH = 140
    data = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)
    print('Shape of data_d tensor:', data.shape)



    # In[77]:


    nb_words = min(MAX_NB_WORDS, len(word_index))
    EMBEDDING_DIM = 300
    embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))

    for (word, idx) in word_index.items():
        print(word, idx)
        if word in word2vec.vocab and idx < MAX_NB_WORDS:
            embedding_matrix[idx] = word2vec.word_vec(word)


    # In[78]:


    from keras.models import model_from_json

    # Model reconstruction from JSON file
    with open('model_architecture.json', 'r') as f:
        model = model_from_json(f.read())

    # Load weights into the new model
    model.load_weights('model_weights.h5')


    # In[79]:


    labels_pred = model.predict(data)
    labels_pred = np.round(labels_pred.flatten())


    # In[80]:


    print(X)
    labels_pred


    # In[81]:


    i = 0
    locations = [] 

    for x in np.nditer(labels_pred):
        if(x == 1):
            locations.append(df.iloc[i]["location"])
        i+=1

    locations
    print(len(locations))


    # In[82]:


    top = 49.3457868 # north lat
    left = -124.7844079 # west long
    right = -66.9513812 # east long
    bottom =  24.7433195 # south lat

    def cull(lat, lng):
        if bottom <= lat <= top and left <= lng <= right:
            return True 
        return False 


    # In[83]:


    from geopy.geocoders import Nominatim, ArcGIS
    from geopy.extra.rate_limiter import RateLimiter

    import pycountry

    df_coord = pd.DataFrame(columns=('Latitude', 'Longitude'))

    geolocator = ArcGIS(timeout=10)

    i = 0
    for location in locations:
        if location:
            loc = geolocator.geocode(location, exactly_one=True)
            if loc:
                print(loc.address)
                if(cull(loc.latitude, loc.longitude)):
                    df_coord.loc[i] = (loc.latitude, loc.longitude)
                    print(loc.latitude, loc.longitude)
                    i+=1


    # In[84]:


    df_coord


    # In[57]:


    i = 0
    demo_locations = [] 

    for x in np.nditer(labels_pred):
        demo_locations.append(df.iloc[i]["location"])
        i+=1

    demo_locations
    print(len(demo_locations))
    with open('coords.csv', 'a', encoding='utf-8') as f:
        df_coord.to_csv(f, header=False, encoding='utf-8')


