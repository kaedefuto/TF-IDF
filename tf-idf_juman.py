from sklearn.feature_extraction.text import TfidfVectorizer

import logging
import os

import gensim

from pyknp import Juman
from gensim import corpora, models

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

name = 'test'
num = 0

#Jumanppオブジェクト

jumanpp = Juman()
train_texts = []
with open('./{0}.txt'.format(name), 'r', encoding = 'utf-8') as f:
  for lines in f:
    try:
      lines = lines.strip('\n')
      lines = lines.strip("#")
      #print(lines)
      text = []
      result = jumanpp.analysis(lines)
      #print(result)
      for mrph in result.mrph_list():
          if mrph.hinsi == '名詞' or mrph.hinsi == '形容詞':
              text.append(mrph.genkei)
      train_texts.append(text)
      print(num)
      num+=1
    except Exception as e:
      print(e)
      pass

print(train_texts)

docs=[]
tx=""
for i in train_texts:
    #print(i)
    tx=""
    for x in i:
        tx +=x
        tx +=" "
    docs.append(tx)
print(docs)
vectorizer = TfidfVectorizer(max_df=0.9) # tf-idfの計算
#                            ^ 文書全体の90%以上で出現する単語は無視する
X = vectorizer.fit_transform(docs)
print('feature_names:', vectorizer.get_feature_names())


words = vectorizer.get_feature_names()
for doc_id, vec in zip(range(len(docs)), X.toarray()):
    print('doc_id:', doc_id)
    for w_id, tfidf in sorted(enumerate(vec), key=lambda x: x[1], reverse=True):
        lemma = words[w_id]
        print('\t{0:s}: {1:f}'.format(lemma, tfidf))
