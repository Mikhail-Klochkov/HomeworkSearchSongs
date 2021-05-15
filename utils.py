
import pandas as pd
import os 
import numpy as np 
import random 
import nltk 
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
from collections import Counter
from nltk.stem import WordNetLemmatizer
import re
import tqdm
from tqdm import tqdm_pandas
from tqdm import tqdm
import string

expr = r'[^a-zA-z\s]'
parser=re.compile(expr)

nltk.download('stopwords')
nltk.download('wordnet')
stopwords_ = set(stopwords.words('english'))

FREQWORDS = None
RAREWORDS = None

def load_data():
    path_files = []
    tables = []
    for roots, _, files in os.walk('./'):
        for file in files:
            if file.find('lyric') != -1 or file.find('artist') != -1:
                #print("root: {}\ndirs: {}\nfiles: {}\n".format(roots, 
                #                                               dirs, 
                #                                               files))
                path_files.append(roots + '/' + file)


    for path in path_files:
        #with open(path, 'r', encoding = 'ISO-8859-1') as f:
        try:
            print("load path: {}".format(path))
            tables.append(pd.read_csv(path))
        except:
            print('Something wrong! ')

    # -- we work with english words -- #
    df_artist, df_songs = tables
    col_name = 'Idiom'
    if col_name in set(df_songs.columns):
        df_en_songs = df_songs[df_songs[col_name] == "ENGLISH"]

    return df_artist, df_en_songs


def clean_regex(text):
  return re.sub(expr, '', text)

class Preprocess(object):

  def __init__(self, 
               stemmed = False,
               lemmatize = False):

    self._stemmed = stemmed
    self._lemmatize = lemmatize
    self._shortcutmethod = None
    if self._stemmed:
      self._shortcutmethod = PorterStemmer().stem
    if self._lemmatize:
      self._shortcutmethod = WordNetLemmatizer().lemmatize

    # -- trivial -- #
    if self._shortcutmethod is None:
      self._shortcutmethod = lambda x: x

  def cleantext(self, 
                text,
                stopwords = True):
    """
    deleted all empty or ' ' just spaces and one 
    """
    """
    if(not stopwords):
        stopwords_ = set()
    """
    text = text.lower()
    return ' '.join([self._shortcutmethod(item) 
            for item in clean_regex(text).split(' ') 
            if (item != '') and (item != ' ') and 
            (len(item) != 1) and (item not in (stopwords_ if stopwords else set()))
            ])

def returnFrequentList(df, col_name_text):
  """
  return Counter of all words in dataset frequent vocabulary
  """
  cnt = Counter()
  for row in df.itertuples(index = True, name = 'Pandas'):
    text_current = getattr(row, col_name_text) 
    words = text_current.split(' ')
    for word in words:
      cnt[word] += 1
  return cnt 

class ClassPreprocessed(object):

    def __init__(self, 
                 text_column = None,
                 DRAREFREQ_words = False,
                 numfreqwords = 10,
                 numrarewords = 20,
                 bar_tqdm = True,
                 stopwords = True,
                 lemmatize = True):

        self._text_column = text_column
        self._DRAREFREQ_words = DRAREFREQ_words
        self._numfreqwords = numfreqwords
        self._numrarewords = numrarewords
        self._bar_tqdm = bar_tqdm
        self._stopwords = stopwords
        self._lemmatize = lemmatize

    def preprocessed(self, df, verbose = 0):

        df_new = df.copy(deep = True)
        if verbose > 0:
            print(f'We preprocessed text column: {self._text_column}\n \
                   Delete frequent or rare words: {self._DRAREFREQ_words}\n \
                   Remove stopwords: {self._stopwords}\n \
                   Lemmatizing : {self._lemmatize}\n')

        prep_cls = Preprocess(lemmatize = self._lemmatize,
                              stemmed = self._lemmatize)
        if(self._bar_tqdm):
            df_new[self._text_column] = df_new[self._text_column].progress_apply(lambda text: prep_cls.cleantext(text))
        else:
            df_new[self._text_column] = df_new[self._text_column].apply(lambda text: prep_cls.cleantext(text))

        # -- if stopwords is set() -- #
        if not self._stopwords:
            global stopwords_
            stopwords_ = set(stopwords.words('english'))
    
        # -- RARE and FREQ words -- #    
        global FREQWORDS, RAREWORDS
        if(self._DRAREFREQ_words):
            counter = returnFrequentList(df_new, 
                                col_name_text = self._text_column)

            FREQWORDS = set([word
                            for word, nums in 
                            counter.most_common(self._numfreqwords)
                            ])

            RAREWORDS = set([word
                            for word, nums in 
                            counter.most_common()[-self._numrarewords: ]
                            ])
            def filterwords(text, nonrelevantsetofwords):
                return ' '.join([w for w in text.split(' ') if w not in nonrelevantsetofwords])

            df_new[self._text_column] = df_new[self._text_column].apply(lambda text: filterwords(text, FREQWORDS))
            df_new[self._text_column] = df_new[self._text_column].apply(lambda text: filterwords(text, RAREWORDS))

        return df_new
