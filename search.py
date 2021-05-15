import random
import pandas as pd 
import os
import re
import numpy as np
from utils import load_data, ClassPreprocessed, FREQWORDS, RAREWORDS, clean_regex, Preprocess
import tqdm
from tqdm import tqdm, tqdm_pandas

class Document:
    def __init__(self, 
                 title, 
                 text,
                 Artist_name = None,
                 Genre = None, 
                 Popularity = None):
        # можете здесь какие-нибудь свои поля подобавлять
        self.title = title
        self.text = text
        self._artist = Artist_name
        self._genre = Genre
        self._popularity = Popularity
    
    def format(self, query):
        # возвращает пару тайтл-текст, отформатированную под запрос
        # -- может быть слишком длинным -- #
        # -- ограничение на название 50, на текст на 120 символов -- #
        # -- Добавим дополнительную информацию -- #
        dop_info = " "
        if(self._artist is not None):
            dop_info = f" | Artist: {self._artist} | Genre : {self._genre} | Popularity: {self._popularity}"
        return ["Name: " + self.title[:50] + ('...' if len(self.title) > 50 else '') + dop_info, self.text[:120] + ('...' if len(self.text) > 120 else '')]

invertindextitles = {}
invertindextext = {}
invertIndexGenres = {}
invertIndexArtistName = {}
index = []
dictionary_fastWV = {}
index_popularity = {}
# -- Если не можем из словаря emmbeddings найти нужное представление -- #
default = 0.

def build_index():
    # считывает сырые данные и строит индекс
    index.append(Document('The Beatles — Come Together', 'Here come old flat top\nHe come groovin\' up slowly'))
    index.append(Document('The Rolling Stones — Brown Sugar', 'Gold Coast slave ship bound for cotton fields\nSold in the market down in New Orleans'))
    index.append(Document('МС Хованский — Батя в здании', 'Вхожу в игру аккуратно,\nОна еще не готова.'))
    index.append(Document('Физтех — Я променял девичий смех', 'Я променял девичий смех\nНа голос лектора занудный,'))

# -- for test -- #
def build_indexMy(df, 
                  title_col = 'SName',
                  text_col = 'Lyric',
                  ):
    bools = []
    for col in [title_col, text_col]:
        if col in set(df.columns):
            bools.append(True)
        else:
            bools.append(False)

    assert(all(bools))

    bound = 0
    if df.shape[0] > 100:
        bound = 100
    else:
        bound = df.shape[0]

    for row in df.iloc[:bound, :].itertuples():
        title = getattr(row, title_col)
        text = getattr(row, title_col)
        index.append(Document(title, text))

def record(title: str, text: str, idx: int):
    """
    idx - it's index in global data set of this row
    get one title and one text and record all term in dicts inverindextitles, and invertindextext
    """
    # -- for changed -- #
    global invertindextext
    global invertindextitles
    words_title = title.split(" ")
    words_text = text.split(" ")
    # -- for title information -- #
    for pos, wtitle in enumerate(words_title, 0):
        if(invertindextitles.get(wtitle, -1) != -1 and wtitle != ''):
            invertindextitles[wtitle]['index'].append(idx)
        else:
            invertindextitles[wtitle] = {'index': [idx]}
    
    # -- for text -- #
    for pos, wtext in enumerate(words_text, 0):
        try:
            if(invertindextext.get(wtext, -1) != -1):
                invertindextext[wtext]['index'].append(idx)
                invertindextext[wtext]['position'].append(pos)
            else:
                invertindextext[wtext] = {'index': [idx], 'position': [pos]}
        except KeyError:
            print('word: {}'.format(wtext))

def createDataInvertIndex(df: pd.DataFrame, 
                          titlecolumn : str = 'SName', 
                          textcolumn :str = 'Lyric'):
    """
    Args: df - preprocessed dataFrame of Songs, titlecolumn and textcolumn - columns  
    Return: Словарь {term: [[indexes of Corpus Documents], [first positions in this Corpus Documents]]}
    Aналог buildindex
    Мы создадим на базе корпуса документов 
    инвертированный индекс (словарь со словами и значениями в каких документах данное слово встретилось)
    """
    global invertindextitles, invertindextext
    # -- create inverted index - По сути есть просто термин и индекс документа, где он содержиться и также позиция в документе -- #
    # -- create dictionary of all Terms -- #
    for idx, row in tqdm(enumerate(df.itertuples())):
        title = getattr(row, titlecolumn)
        text = getattr(row, textcolumn)
        # -- Наверное выгодно, в случае песен, прежде всего искать совпадения среди названий песен -- #
        # -- record all new terms -- #
        record(title, text, idx)

    # -- Genres -- #
    uniqueGenres = df['Genre'].unique()
    groupGenres = df.groupby(by = ['Genre'])
    # -- {genre: list of indeces by order in df} -- #
    for genrename in uniqueGenres:
        invertIndexGenres[genrename] = groupGenres.get_group(genrename).index.values.tolist()

    # -- NameArtists -- #
    # -- {artistname : list of indeces by order in df} -- #
    uniqueNames = df['Artist'].unique()
    groupArtists = df.groupby(by = ['Artist'])
    for artistname in uniqueNames:
        invertIndexArtistName[artistname] = groupArtists.get_group(artistname).index.values.tolist()
    
    # -- index : Popularity -- #
    index_popularity = df['Popularity'].to_dict()



# -- Просто записывает не предобработанные данные для первичного вывода -- #
def build_index2(df: pd.DataFrame,
                 titlecolumn: str = 'SName',
                 textcolumn: str = 'Lyric',
                 new_columns = False):
    """
    df - not preprocessed data
    """
    # -- record all not preprocessed data -- #
    for _, row in enumerate(df.itertuples(), 0):
        
        if(not new_columns):
            index.append(Document(getattr(row, titlecolumn), 
                                  getattr(row, textcolumn)))
        else:
            index.append(Document(getattr(row, titlecolumn), 
                                  getattr(row, textcolumn),
                                  getattr(row, 'Artist'),
                                  getattr(row, 'Genre'),
                                  getattr(row, 'Popularity')))

def load_embedding_WV():
    """
    load and fill dictionary_fastWV for word: vector in R^{300} 
    """
    global dictionary_fastWV
    print('Load and fill dictionary_fastWV for {word : embeddings}\n')
    with open("wiki-news-300d-1M.vec", 'r') as f:
        with tqdm(position = 0, leave = True) as pbar:
            for line in tqdm(f, position = 0, 
                             desc = 'download progress: ', 
                             leave = True):
            
                wplusvec = line.split(" ")
                if wplusvec.__len__() == 301:
                    #print("here ", wplusvec[0], len(wplusvec[1:]))
                    #break
                    w = wplusvec[0]
                    vecd = np.array(wplusvec[1:], 
                                    dtype = np.float32)
                    dictionary_fastWV[w] = vecd
                if len(dictionary_fastWV) == 300000:
                    break
                pbar.update()

# -- we should load data from another file -- #

def loadandbuildindex2():
    global default
    nums = 1000
    
    df_artists, df_en_songs = load_data()
    df_en_songs = df_en_songs.iloc[:nums, :]

    # -- after load we should create corpus Documents for search system -- #
    # -- preprocessed data -- #  
    tqdm_pandas(tqdm())
    
    clspreproccessed = ClassPreprocessed(text_column = 'Lyric',
                                               DRAREFREQ_words = True)
    df_new_songs = clspreproccessed.preprocessed(df_en_songs)
    # -- save -- #
    df_new_songs.to_csv('preprocessed_data.csv')

    print(df_new_songs['Lyric'][0])
    print('Most common words:\n')
    print(FREQWORDS)
    print('Most rare words:\n')
    print(RAREWORDS)

    # -- we should preprocessed title of songs -- #
    clspreproccessed2 = ClassPreprocessed(text_column = 'SName',
                                          DRAREFREQ_words = False, 
                                          stopwords = False,
                                          lemmatize = False)
    df_new_songs = clspreproccessed2.preprocessed(df_new_songs, verbose = 1)
    df_new_songs.to_csv('preprocessed_data.csv')    
    try:
        index = np.random.choice(df_new_songs.index, size = 1)[0]
        print('before: \n')
        print(df_en_songs.loc[index, ['Lyric', 'SName']].values)
        print('after: \n')
        print(df_new_songs.loc[index, ['Lyric', 'SName']].values)
    except:
        print('Choice works bad!')
    print('all Done!')
    # -- create invert index -- #
    # -- We have alseo information about Artist, Popularity, Genre of each music -- #
    # -- We can use it for getting more actual results -- #
    # -- We should use name artist and Genre and Popularity -- #
    # -- merge df_artist and df_songs -- #
    merge = pd.merge(df_new_songs, 
                             df_artists, 
                             how = 'inner', 
                             left_on = 'ALink',
                             right_on = 'Link')
    # -- У нас есть ещё жанр Genre и Artist_name -- #
    # -- Мы должны этим воспользоваться -- #
    # -- Сгруппировать песни по Жанру и Артисту -- #
    # -- Тоже сделать что-то вроде базы данных, которая содержит список песен для каждого отдельного взятого певца -- #                        
    createDataInvertIndex(merge)
    build_index2(merge, new_columns = True)
    # -- Заполняем наш словарь отображений слов -- #
    load_embedding_WV() 
    # -- По умолчанию вектор -- #
    default = sum(dictionary_fastWV.values())/len(dictionary_fastWV)



def score(query, document):
    # возвращает какой-то скор для пары запрос-документ
    # больше -- релевантнее
    return random.random()

def score2(query, document):
    # -- by fast vec -- #
    """Построим представление
     векторное (как было показано в материалах к работе, 
     будем строить нечно средне взвешанное всех представлений 
     слов входящих в каждый документ с весами, полученными из 
     tfidf) отдельно взятого документа и запроса и по 
     cos(vq, vd) = (vq, vd)/(|vq| * |vd|)"""
    # -- строим общий текст title + text -- #
    words_text = (document.title + document.text).split(" ")
    # -- Уберём все лишнее если есть и приведем к нормализованному стостоянию -- #
    query = Preprocess().cleantext(query)
    words_query = query.split(" ")
    vd = sum(list(map(lambda w: dictionary_fastWV.get(w, default), words_text)))/len(words_text)
    vq = sum(list(map(lambda w: dictionary_fastWV.get(w, default), words_query)))/len(words_query)
    # -- Само значение косинусной близости -- #
    return np.dot(vd, vq)/(np.linalg.norm(vd)*np.linalg.norm(vq))

def retrieve(query):
    # возвращает начальный список релевантных документов
    # (желательно, не бесконечный)
    # линейный поиск среди некоторого блока документов
    candidates = []
    for doc in index:
        try:
            if (query.lower() in doc.title.lower()) or (query in doc.text.lower()):
                candidates.append(doc)
        except AttributeError:
            print("quare: {} title: {} text : {}".format(query, doc.title, doc.text))
    return candidates[:50]

def findKeyInInvertIndexDict(II: dict, querylower):
    max_len = 0
    find_key = str()
    for key in II.keys():
        len_current = 0
        for pn in key.split(" "):
            s = re.search(pn.lower(), querylower)
            if(s is not None):
                len_current += len(s.group())
        if(max_len < len_current):
            max_len = len_current
            find_key = key
    return find_key

    
def retrieve2(query):
    """
    Получаем запорос, если не пустой, то с помощью инвертированного индекса 
    Записываем потенциальных претендетнов на вывод. Это должно быть достаточно быстро.
    Сперва смотрим на title, так как чаще всего люди ищут песню по её названию. Если в top 100 or 200 
    Не все по названию попали, то смотрим уже по содержанию в тексте песни. Хотя это моежт быть неправильно, ибо 
    общие слова есть в текстах всех песен, поэтому стоит попробовать строить для каждого текста его векторное представление
    Аналогично тому, как мы это делали с текстами и embeddings. 
    """
    # -- Введёт исполнителя -- #
    # -- Жанр -- #
    # -- Пытаться вспомнить имя песни (точно, неточно) -- #
    # -- Исполнитель + Песня + Жанр -- #
    # -- Пытаться кусочек фразы вспомнить из песни -- #
    N = 100
    # -- if empty enter first  or maybe by popularity -- #
    if(query.__len__() == 0):
        return index[:N]

    query_words = [w.lower() for w in query.split(" ")]
    query_lowercase = " ".join(query_words)
    # -- Поискать среди исполнителей -- #
    # -- [query1, query2, ...], [name1, name2, ...] -- #
    # -- Мы просто пробегаемся по именам исполнителей и ищем максимальное совпадение с запросом query -- #
    name_artist = findKeyInInvertIndexDict(invertIndexArtistName, query_lowercase)
    # -- we have find_name -- #
    # -- Можно поискать среди жанров -- # (Их мало)
    # -- Человек может просто ввести жанр и например по популярности логично вывезти слова -- #
    name_genre = findKeyInInvertIndexDict(invertIndexGenres, query_lowercase)
    # -- поискать по названию песни -- #
    # -- Надо query преобразовать по подобию, как мы делали с title -- #
    prep = Preprocess(lemmatize = True)
    query_prep_words = prep.cleantext(query_lowercase, stopwords = True).split(" ")
    # -- Предобработанный список слов query_prep_words -- #
    indeces_of_documents = []
    indeces_title = []
    for w in query_prep_words:
        if(invertindextitles.get(w, -1) != -1):
            indeces_title += invertindextitles[w]['index']

    indeces_text = []
    for w in query_prep_words:
        if(invertindextext.get(w, -1) != -1):
            indeces_text += invertindextext[w]['index']

    # -- Можно уже на самом деле выводить список, если мы смогли набрать за счёт indeces_title -- #
    indeces_of_documents += indeces_title    
    if(name_artist != str()):
        # -- we have some match -- #
        # -- need intersection with indeces_title -- #
        # -- indeces_title is list, invertArtist is list also -- #
        # -- Может быть и не нужно искать пересечение -- #
        indeces_of_documents += list(set(indeces_of_documents).intersection(invertIndexArtistName[name_artist]))

    # -- Если ничего до не нашли пройтись по тексту песен -- # 
    if(len(indeces_of_documents) == 0 and len(indeces_text) != 0):
        # -- Ничего не отобрали -- #
        indeces_of_documents += indeces_of_text 
        
    # -- Если претендентов больше скажем 100, то мы посмотрим на Popularity песен и по нему отсортируем -- #
    # -- sort by popularity всегда сортируем что-бы там не было по популярности -- #
    if(len(indeces_of_documents) != 0):
        return np.array(index)[sorted(indeces_of_documents, key = lambda x: index_popularity[x])][:N].tolist()
    else:
        # -- Вообще ничего не нашли, то надо что-то хоть отобрать -- #
        # -- Например по популярности -- #
        return np.array(index)[[ind for ind, _ in sorted(index_popularity, key = lambda x: x[1])]][: N].tolist()