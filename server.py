from flask import Flask, render_template, request
from search1 import score, retrieve, loadandbuildindex2, createDataInvertIndex, build_index, build_indexMy
from time import time
import os

app = Flask(__name__, template_folder='.')

# -- get df -- #

"""
try:
    file_path = str()
    for roots, dirs, files in os.walk('./'):
        for file in files:
            if file.find('lyric') != -1:
                print('find!')
                file_path = roots + '/' + file
    print('file path is : {}'.format(file_path))
    # -- Why this cannot load -- #
    df = pd.read_csv(file_path)
except:
    print('Did not read dataframe!')
"""
loadandbuildindex2()

@app.route('/', methods=['GET'])
def index():
    start_time = time()
    query = request.args.get('query')
    if query is None:
        query = ''
    # return some documents 
    documents = retrieve2(query)
    # тут уже определяет некоторое ранжирование (сортируем)
    # doc - some feature (and returned all data)
    documents = sorted(documents, key=lambda doc: -score(query, doc))
    # для каждого документа меняем его формат добавляя строку с присвоенным значением величины ранга
    #results = [doc.format(query)+['%.2f' % score(query, doc)] for doc in documents] 
    results = [doc.format(query)+['%.2f' % score(query, doc)] for doc in documents] 
    
    return render_template(
        'index.html',
        time="%.2f" % (time()-start_time),
        query=query,
        search_engine_name='Yandex',
        results=results
    )

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5004)
